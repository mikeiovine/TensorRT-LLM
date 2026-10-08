#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bucket an nsys kernel trace of the SID-GR beam benchmark like the GR analyzer.

Produces the same split the recsys-examples ``analyze_nsys_gr_sglang.py``
reports for its engine (attention / GEMM / top-k & beam selection / other
kernels, plus CPU gaps inside the active window) so a TensorRT-LLM profile can
be compared bucket by bucket.

Capture (executor in-process, profiler range around the timed runs)::

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --cuda-graph-trace=node -o trtllm_sid_gr \\
        python examples/llm-api/bench_sid_gr_beam.py offline --single-process \\
            --cuda-profiler-range --context-lens 1000 --batch-sizes 1 --repeat 3 \\
            --out-dir benchmark_artifacts/sid_gr/profile

Export and analyze::

    nsys stats --report cuda_gpu_trace --format csv --output trtllm_sid_gr trtllm_sid_gr.nsys-rep
    python examples/llm-api/analyze_sid_gr_nsys.py trtllm_sid_gr_cuda_gpu_trace.csv --repeat 3

``--cuda-graph-trace=node`` is required so kernels replayed inside CUDA graphs
appear individually. ``--repeat`` divides totals by the number of timed runs
inside the profiler range to report per-request-batch numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# First match wins; mirrors the GR analyzer's substring buckets, extended with
# the TensorRT-LLM kernel names for the same roles.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("topk_beam_select", ("topk", "top_k", "radix", "beam", "logsoftmax", "log_softmax",
                          "softmax", "cumsum", "scatter", "gather", "index_put", "indexselect",
                          "index_select", "sort", "argsort")),
    ("attention", ("attention", "mmha", "masked_multihead", "cascade", "fmha", "flash",
                   "xqa", "prefill", "decode", "kv_cache", "kvcache", "applybiasrope",
                   "qknorm", "qk_norm", "rope", "mergestate", "merge_state")),
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "sm90_xmma", "sm80_xmma", "matmul",
              "splitk", "split_k", "ampere_", "hopper_")),
    ("norm_act_elementwise", ("rmsnorm", "rms_norm", "layernorm", "silu", "swiglu", "gelu",
                              "elementwise", "vectorized", "fill", "copy", "cast", "embedding",
                              "unrolled", "reduce", "add")),
]


def bucket_of(name: str) -> str:
    lower = name.lower()
    for bucket, needles in BUCKETS:
        if any(needle in lower for needle in needles):
            return bucket
    return "other"


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        # nsys prepends metadata lines before the header on some versions.
        lines = handle.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Start") or
                      line.startswith("Start (ns)"))
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def column(row: dict[str, str], *candidates: str) -> str | None:
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_csv", help="nsys stats --report cuda_gpu_trace CSV export")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Timed runs inside the profiler range (per-run numbers = totals / repeat)")
    parser.add_argument("--gap-us", type=float, default=50.0,
                        help="Idle gaps between consecutive kernels longer than this count as CPU overhead")
    parser.add_argument("--top", type=int, default=15, help="Kernels to list per bucket")
    args = parser.parse_args()

    rows = read_trace(Path(args.trace_csv))
    kernels = []
    for row in rows:
        name = column(row, "Name", "Kernel Name")
        start = column(row, "Start (ns)", "Start")
        duration = column(row, "Duration (ns)", "Duration")
        if name is None or start is None or duration is None:
            continue
        # Skip memcpy/memset rows (they have no grid) unless they carry a kernel name.
        if column(row, "GrdX", "Grid X") is None and "memcpy" in name.lower():
            continue
        kernels.append((int(float(start)), int(float(duration)), name))
    if not kernels:
        sys.exit("no kernel rows found; export with: nsys stats --report cuda_gpu_trace --format csv")
    kernels.sort()

    window_ns = kernels[-1][0] + kernels[-1][1] - kernels[0][0]
    kernel_ns = sum(d for _, d, _ in kernels)
    gap_ns = 0
    gap_count = 0
    threshold_ns = int(args.gap_us * 1000)
    prev_end = kernels[0][0]
    for start, duration, _ in kernels:
        gap = start - prev_end
        if gap > threshold_ns:
            gap_ns += gap
            gap_count += 1
        prev_end = max(prev_end, start + duration)

    per_bucket_ns: dict[str, int] = defaultdict(int)
    per_bucket_count: dict[str, int] = defaultdict(int)
    per_kernel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, duration, name in kernels:
        bucket = bucket_of(name)
        per_bucket_ns[bucket] += duration
        per_bucket_count[bucket] += 1
        per_kernel[name][0] += duration
        per_kernel[name][1] += 1

    rep = max(args.repeat, 1)
    ms = lambda ns: ns / 1e6 / rep  # noqa: E731

    print(f"timed runs: {rep}   kernels: {len(kernels)} ({len(kernels) / rep:.0f} per run)")
    print(f"{'metric':<34}{'ms / run':>12}")
    print(f"{'active window (first..last kernel)':<34}{ms(window_ns):>12.3f}")
    print(f"{'kernel time':<34}{ms(kernel_ns):>12.3f}")
    print(f"{f'CPU gaps > {args.gap_us:.0f} us ({gap_count} gaps)':<34}{ms(gap_ns):>12.3f}")
    print()
    print(f"{'bucket':<24}{'ms / run':>12}{'launches / run':>18}")
    for bucket in [b for b, _ in BUCKETS] + ["other"]:
        print(f"{bucket:<24}{ms(per_bucket_ns[bucket]):>12.3f}{per_bucket_count[bucket] / rep:>18.1f}")
    print()
    for bucket in [b for b, _ in BUCKETS] + ["other"]:
        names = sorted(((v[0], v[1], n) for n, v in per_kernel.items() if bucket_of(n) == bucket),
                       reverse=True)[:args.top]
        if not names:
            continue
        print(f"[{bucket}]")
        for total, count, name in names:
            print(f"  {ms(total):>9.3f} ms  {count / rep:>8.1f} x  {name[:110]}")
        print()


if __name__ == "__main__":
    main()
