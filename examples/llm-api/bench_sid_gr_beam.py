#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SID-GR style beam-search benchmark for TensorRT-LLM.

Reproduces the measurement methodology of recsys-examples
``examples/sid-gr-inference`` (long context, short decode, large beam) against
the TensorRT-LLM LLM API and ``trtllm-serve``, without any SGLang dependency.

Three sub-commands:

``offline``
    Build one ``LLM``, then for each (context_len, beam_width, batch) case:
    submit ``batch`` requests in a single ``generate`` call, run until all
    finish, and record the wall time. ``--warmup-runs`` untimed runs precede
    ``--repeat`` timed runs; the median is reported. Workloads are the same
    deterministic token-id sequences the GR tool generates with
    ``--no-tokenizer`` (or load theirs with ``--workload-jsonl``).

``online``
    Closed-loop HTTP client against ``trtllm-serve`` ``/v1/completions`` with
    ``--max-concurrency`` in-flight requests, mirroring the GR online recipe
    (ctx 5000, 3 output tokens, beam 256, 64 requests, concurrency 4).

``summarize``
    Rebuild ``summary.md`` / ``summary.csv`` from an output directory. Pass
    ``--gr-dir`` pointing at a GR sweep directory (their ``gr_ctx*_beam*_req*.json``
    files) to add a side-by-side GR column.

Profiling: run with ``--single-process`` (or ``TLLM_WORKER_USE_SINGLE_PROCESS=1``)
so the executor is in-process, then either ``--cuda-profiler-range`` (brackets
the timed runs with cudaProfilerStart/Stop for
``nsys --capture-range=cudaProfilerApi``) or the iteration-scoped
``TLLM_PROFILE_START_STOP`` env var. ``analyze_sid_gr_nsys.py`` buckets the
resulting kernel trace the way the GR analyzer does.

Decode-attention variants for wide beams (``offline`` flags, all become
environment variables that must be set before TensorRT-LLM is imported, which
this script does):

``--cascade-mmha`` (default on)
    ``TRTLLM_ENABLE_CASCADE_MMHA=1``: the C++ cascade kernels read the shared
    prompt KV once per request instead of once per beam.
``--fmha-libs +beam_shared_prefix``
    ``TLLM_FMHA_LIBS``: the Python FlashInfer-based shared-prefix library for
    generation-only beam batches (see
    ``tensorrt_llm/_torch/attention/backends/fmha/beam_shared_prefix.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

# Matches recsys-examples tools/make_qwen3_beam_workload.py --no-tokenizer.
QWEN3_VOCAB_SIZE = 151936


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #
def deterministic_token_ids(idx: int, *, context_len: int,
                            vocab_size: int) -> list[int]:
    """Same formula as make_qwen3_beam_workload._deterministic_token_ids."""
    if vocab_size <= 1024:
        raise ValueError("vocab_size must be > 1024 for deterministic workload")
    start = 1024 + idx * 17
    span = vocab_size - 1024
    return [1024 + ((start + pos * 13) % span) for pos in range(context_len)]


def build_workload(requests: int, context_len: int,
                   vocab_size: int) -> list[list[int]]:
    return [
        deterministic_token_ids(idx,
                                context_len=context_len,
                                vocab_size=vocab_size)
        for idx in range(requests)
    ]


def load_workload_jsonl(path: Path, *, requests: int,
                        context_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids = row.get("input_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"{path}:{line_no} has no input_ids list")
            if len(ids) != context_len:
                raise ValueError(f"{path}:{line_no} has {len(ids)} tokens, "
                                 f"expected context_len={context_len}")
            rows.append([int(t) for t in ids])
            if len(rows) == requests:
                break
    if len(rows) < requests:
        raise ValueError(f"{path} has {len(rows)} usable rows, need {requests}")
    return rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_int_list(value: str) -> list[int]:
    return [int(v) for v in value.replace(",", " ").split()]


def median(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[index]


def git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "trtllm_commit": git_commit(Path(__file__).resolve().parents[2]),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
    except ImportError:
        pass
    try:
        import tensorrt_llm
        info["tensorrt_llm"] = tensorrt_llm.__version__
    except ImportError:
        pass
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def apply_env_knobs(args) -> dict[str, str]:
    """Translate the engine knobs into environment variables.

    Must run before anything imports ``tensorrt_llm``: the C++ cascade gate
    and the FMHA library list are read once per process.
    """
    knobs: dict[str, str] = {}
    if getattr(args, "single_process", False):
        knobs["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    if getattr(args, "cascade_mmha", None) is not None:
        knobs["TRTLLM_ENABLE_CASCADE_MMHA"] = "1" if args.cascade_mmha else "0"
    if getattr(args, "fmha_libs", None):
        knobs["TLLM_FMHA_LIBS"] = args.fmha_libs
    if getattr(args, "beam_max_tail", None) is not None:
        knobs["TLLM_BEAM_SHARED_PREFIX_MAX_TAIL"] = str(args.beam_max_tail)
    for item in getattr(args, "env", None) or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        knobs[key] = value
    if "tensorrt_llm" in sys.modules and knobs:
        print("WARNING: tensorrt_llm already imported; env knobs may not take effect",
              file=sys.stderr)
    os.environ.update(knobs)
    return knobs


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #
def build_llm(args, *, max_context_len: int, max_beam_width: int,
              max_batch_size: int):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    max_num_tokens = args.max_num_tokens or max_batch_size * max_context_len
    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_beam_width": max_beam_width,
        "max_batch_size": max_batch_size,
        "max_num_tokens": max_num_tokens,
        "max_seq_len": max_context_len + args.output_len + 1,
        "kv_cache_config": KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=args.kv_cache_free_gpu_mem_fraction,
        ),
        "disable_overlap_scheduler": args.disable_overlap_scheduler,
        "enable_chunked_prefill": False,
        "skip_tokenizer_init": True,
    }
    if args.no_cuda_graph:
        kwargs["cuda_graph_config"] = None
    else:
        kwargs["cuda_graph_config"] = CudaGraphConfig(
            batch_sizes=sorted(set(args.batch_sizes)))
    if args.attn_backend:
        kwargs["attn_backend"] = args.attn_backend
    if args.iter_stats:
        kwargs["enable_iter_perf_stats"] = True
    if args.prefill_cuda_graph != "disabled":
        # One bucket per (batch, context length) the sweep will prefill in a
        # single step; the engine pads a context batch up to the next bucket.
        buckets = sorted({
            batch * ctx
            for batch in args.batch_sizes for ctx in args.context_lens
            if batch * ctx <= max_num_tokens
        })
        kwargs["prefill_cuda_graph_backend"] = args.prefill_cuda_graph
        kwargs["prefill_capture_num_tokens"] = buckets
        if args.prefill_cuda_graph == "piecewise":
            from tensorrt_llm.llmapi import TorchCompileConfig
            kwargs["torch_compile_config"] = TorchCompileConfig(
                enable_fullgraph=False, enable_inductor=False)
    if args.extra_llm_api_options:
        import yaml
        extra = yaml.safe_load(Path(args.extra_llm_api_options).read_text()) or {}
        kwargs.update(extra)

    print("LLM kwargs:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    return LLM(**kwargs), kwargs


def make_sampling_params(beam_width: int, output_len: int,
                         perf_metrics: bool = True):
    from tensorrt_llm import SamplingParams
    return SamplingParams(
        use_beam_search=True,
        n=beam_width,
        max_tokens=output_len,
        ignore_eos=True,
        end_id=-1,
        return_perf_metrics=perf_metrics,
    )


def _timing_split(outputs, start_mono: Optional[float] = None,
                  end_mono: Optional[float] = None) -> dict[str, Optional[float]]:
    """Prefill / decode split from RequestPerfMetrics timing (ms).

    Valid when every request in the run is scheduled in the same batch, which
    is the case for the offline sweep (batch <= max_batch_size).

    The executor stamps its times with the C++ steady clock, which on Linux is
    the clock behind ``time.monotonic()``; given the monotonic start/end of the
    ``generate`` call this also reports the client-side legs: ``submit_ms``
    (call to executor arrival) and ``response_ms`` (last token to return).
    """
    arrival, scheduled, first, last = [], [], [], []
    for out in outputs:
        # The PyTorch result path attaches the per-request metrics to each
        # CompletionOutput (identical across beams), not to the RequestOutput.
        pm = getattr(out, "request_perf_metrics", None)
        if pm is None and out.outputs:
            pm = getattr(out.outputs[0], "request_perf_metrics", None)
        tm = getattr(pm, "timing_metrics", None) if pm is not None else None
        if tm is None:
            continue
        arrival.append(tm.arrival_time.total_seconds())
        scheduled.append(tm.first_scheduled_time.total_seconds())
        first.append(tm.first_token_time.total_seconds())
        last.append(tm.last_token_time.total_seconds())
    if not first:
        return {"queue_ms": None, "prefill_ms": None, "decode_ms": None,
                "submit_ms": None, "response_ms": None}
    split: dict[str, Optional[float]] = {
        "queue_ms": (min(scheduled) - min(arrival)) * 1000.0,
        "prefill_ms": (max(first) - min(scheduled)) * 1000.0,
        "decode_ms": (max(last) - max(first)) * 1000.0,
        "submit_ms": None,
        "response_ms": None,
    }
    if start_mono is not None and end_mono is not None:
        submit = (min(arrival) - start_mono) * 1000.0
        response = (end_mono - max(last)) * 1000.0
        # Only meaningful when both clocks agree (same steady clock); a
        # negative leg means they do not, and the legs are left unset.
        if submit >= 0 and response >= 0:
            split["submit_ms"] = submit
            split["response_ms"] = response
    return split


def _beam_results(output) -> list[dict[str, Any]]:
    beams = []
    for beam in output.outputs:
        beams.append({
            "token_ids": list(beam.token_ids),
            "score": beam.cumulative_logprob,
        })
    beams.sort(key=lambda b: (b["score"] is None, -(b["score"] or 0.0)))
    for rank, beam in enumerate(beams):
        beam["rank"] = rank
    return beams


def run_offline_case(llm, args, *, context_len: int, beam_width: int,
                     requests: int) -> dict[str, Any]:
    if args.cuda_profiler_range:
        import torch

    if args.workload_jsonl:
        prompts = load_workload_jsonl(Path(args.workload_jsonl),
                                      requests=requests,
                                      context_len=context_len)
    else:
        prompts = build_workload(requests, context_len, args.vocab_size)
    sampling = make_sampling_params(beam_width, args.output_len,
                                    perf_metrics=not args.no_perf_metrics)

    # generate() blocks until every request has finished and its tokens are
    # on the host, so wall time needs no device synchronize. Avoiding CUDA
    # calls here also keeps the parent process from allocating a CUDA context
    # when the executor runs in its own worker process.
    def run_once() -> tuple[float, list, float, float]:
        start_mono = time.monotonic()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return wall_ms, outputs, start_mono, time.monotonic()

    def drain_iter_stats() -> list[dict]:
        if not args.iter_stats:
            return []
        stats = []
        # Stats are queued per executor iteration; collect everything emitted
        # so far (the queue is empty once a short timeout expires).
        while True:
            chunk = llm.get_stats(timeout=0.2)
            if not chunk:
                return stats
            stats.extend(s if isinstance(s, dict) else json.loads(s) for s in chunk)

    for _ in range(args.warmup_runs):
        run_once()
    drain_iter_stats()

    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    runs = []
    try:
        for _ in range(args.repeat):
            wall_ms, outputs, start_mono, end_mono = run_once()
            run = {"wall_ms": wall_ms, **_timing_split(outputs, start_mono, end_mono)}
            run["generated_tokens"] = sum(
                len(b.token_ids) for o in outputs for b in o.outputs)
            if args.iter_stats:
                iters = drain_iter_stats()
                run["iterations"] = [{
                    "iter": s.get("iter"),
                    "latency_ms": s.get("iterLatencyMS"),
                    "queue_latency_ms": s.get("newActiveRequestsQueueLatencyMS"),
                    "num_active": s.get("numActiveRequests"),
                    "num_ctx": s.get("inflightBatchingStats", {}).get("numContextRequests"),
                    "num_gen": s.get("inflightBatchingStats", {}).get("numGenRequests"),
                } for s in iters]
                run["iter_latency_sum_ms"] = sum(
                    float(s.get("iterLatencyMS") or 0.0) for s in iters)
            runs.append((run, outputs))
    finally:
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()

    last_outputs = runs[-1][1]
    for out in last_outputs:
        if len(out.outputs) != beam_width:
            raise RuntimeError(
                f"request {out.request_id} returned {len(out.outputs)} beams, "
                f"expected {beam_width}")
    walls = [r["wall_ms"] for r, _ in runs]
    wall_median = median(walls)
    result: dict[str, Any] = {
        "framework": "tensorrt_llm",
        "context_len": context_len,
        "beam_width": beam_width,
        "requests": requests,
        "output_len": args.output_len,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "wall_ms_samples": walls,
        "wall_ms_median": wall_median,
        "qps": requests / (wall_median / 1000.0) if wall_median else None,
        "runs": [r for r, _ in runs],
    }
    for key in ("queue_ms", "prefill_ms", "decode_ms", "submit_ms", "response_ms"):
        result[f"{key}_median"] = median(r.get(key) for r, _ in runs)
    if args.iter_stats:
        result["iter_latency_sum_ms_median"] = median(
            r.get("iter_latency_sum_ms") for r, _ in runs)
        result["num_iterations_median"] = median(
            len(r.get("iterations", [])) for r, _ in runs)
    if args.record_outputs:
        result["outputs"] = [{
            "request_index": idx,
            "prompt_tokens": len(prompts[idx]),
            "beam_results": _beam_results(out),
        } for idx, out in enumerate(last_outputs)]
    return result


def cmd_offline(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    knobs = apply_env_knobs(args)
    if args.cuda_profiler_range and os.environ.get(
            "TLLM_WORKER_USE_SINGLE_PROCESS") != "1":
        print("WARNING: --cuda-profiler-range only brackets GPU work when "
              "the executor is in-process (pass --single-process).",
              file=sys.stderr)

    llm, llm_kwargs = build_llm(
        args,
        max_context_len=max(args.context_lens),
        max_beam_width=max(args.beam_widths),
        max_batch_size=max(args.batch_sizes),
    )
    env = environment_info()
    env["env_knobs"] = knobs
    env["variant"] = args.variant
    try:
        for context_len in args.context_lens:
            for beam_width in args.beam_widths:
                for requests in args.batch_sizes:
                    suffix = f"ctx{context_len}_beam{beam_width}_req{requests}"
                    print(f"== {suffix} ==", flush=True)
                    result = run_offline_case(llm,
                                              args,
                                              context_len=context_len,
                                              beam_width=beam_width,
                                              requests=requests)
                    result["environment"] = env
                    result["llm_kwargs"] = llm_kwargs
                    write_json(out_dir / f"trtllm_{suffix}.json", result)
                    print(f"  wall_ms_median={fmt(result['wall_ms_median'])} "
                          f"submit_ms={fmt(result['submit_ms_median'])} "
                          f"queue_ms={fmt(result['queue_ms_median'])} "
                          f"prefill_ms={fmt(result['prefill_ms_median'])} "
                          f"decode_ms={fmt(result['decode_ms_median'])} "
                          f"response_ms={fmt(result['response_ms_median'])} "
                          f"iter_sum_ms={fmt(result.get('iter_latency_sum_ms_median'))} "
                          f"iters={result.get('num_iterations_median', '')} "
                          f"samples={[round(w, 3) for w in result['wall_ms_samples']]}",
                          flush=True)
    finally:
        llm.shutdown()
    summarize(out_dir, Path(args.gr_dir) if args.gr_dir else None)


# --------------------------------------------------------------------------- #
# Online
# --------------------------------------------------------------------------- #
async def _online(args) -> dict[str, Any]:
    import aiohttp

    base = f"http://{args.host}:{args.port}"
    prompts = build_workload(args.requests, args.context_len, args.vocab_size)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.served_model_name
        if model is None:
            async with session.get(f"{base}/v1/models") as resp:
                resp.raise_for_status()
                model = (await resp.json())["data"][0]["id"]

        def payload(ids: list[int]) -> dict[str, Any]:
            body = {
                "model": model,
                "prompt": ids,
                "max_tokens": args.output_len,
                "n": args.beam_width,
                "use_beam_search": True,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": False,
            }
            if args.extra_body:
                body.update(json.loads(args.extra_body))
            return body

        async def one(ids: list[int]) -> dict[str, Any]:
            start = time.perf_counter()
            async with session.post(f"{base}/v1/completions",
                                    json=payload(ids)) as resp:
                text = await resp.text()
                latency_ms = (time.perf_counter() - start) * 1000.0
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:400]}")
                data = json.loads(text)
            choices = data.get("choices", [])
            return {
                "latency_ms": latency_ms,
                "num_choices": len(choices),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
            }

        sem = asyncio.Semaphore(args.max_concurrency)

        async def guarded(ids):
            async with sem:
                return await one(ids)

        for _ in range(args.warmup_requests):
            await one(prompts[0])

        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(p) for p in prompts))
        wall_s = time.perf_counter() - start

    latencies = [r["latency_ms"] for r in results]
    bad = [r for r in results if r["num_choices"] != args.beam_width]
    return {
        "framework": "tensorrt_llm",
        "mode": "online",
        "endpoint": f"{base}/v1/completions",
        "model": model,
        "context_len": args.context_len,
        "output_len": args.output_len,
        "beam_width": args.beam_width,
        "requests": args.requests,
        "max_concurrency": args.max_concurrency,
        "warmup_requests": args.warmup_requests,
        "wall_s": wall_s,
        "request_throughput": args.requests / wall_s,
        "input_tokens_per_s": args.requests * args.context_len / wall_s,
        "output_tokens_per_s": args.requests * args.output_len / wall_s,
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p90": percentile(latencies, 0.90),
        "latency_ms_p99": percentile(latencies, 0.99),
        "requests_with_wrong_beam_count": len(bad),
        "per_request": results,
        "environment": environment_info(),
    }


def cmd_online(args) -> None:
    summary = asyncio.run(_online(args))
    out = Path(args.out_dir) / (
        f"trtllm_online_ctx{args.context_len}_beam{args.beam_width}"
        f"_req{args.requests}_mc{args.max_concurrency}.json")
    write_json(out, summary)
    print(f"req/s={summary['request_throughput']:.2f} "
          f"p50={summary['latency_ms_p50']:.2f}ms "
          f"p90={summary['latency_ms_p90']:.2f}ms "
          f"p99={summary['latency_ms_p99']:.2f}ms "
          f"input_tok/s={summary['input_tokens_per_s']:.0f} "
          f"wrong_beam_count={summary['requests_with_wrong_beam_count']}")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _gr_wall_ms(gr_dir: Optional[Path], suffix: str) -> Optional[float]:
    if gr_dir is None:
        return None
    path = gr_dir / f"gr_{suffix}.json"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["wall_ms_median"])
    except (KeyError, TypeError, ValueError):
        return None


def summarize(out_dir: Path, gr_dir: Optional[Path]) -> None:
    rows = []
    for path in sorted(out_dir.glob("trtllm_ctx*_beam*_req*.json")):
        data = json.loads(path.read_text())
        suffix = path.stem.removeprefix("trtllm_")
        trt = data.get("wall_ms_median")
        gr = _gr_wall_ms(gr_dir, suffix)
        rows.append({
            "variant": data.get("environment", {}).get("variant", ""),
            "context_len": data["context_len"],
            "beam_width": data["beam_width"],
            "batch_requests": data["requests"],
            "trtllm_wall_ms": trt,
            "trtllm_prefill_ms": data.get("prefill_ms_median"),
            "trtllm_decode_ms": data.get("decode_ms_median"),
            "trtllm_qps": data.get("qps"),
            "gr_wall_ms": gr,
            "trtllm_over_gr": (trt / gr) if trt and gr else None,
        })
    rows.sort(key=lambda r: (r["context_len"], r["beam_width"], r["batch_requests"]))
    if not rows:
        print(f"no trtllm_ctx*_beam*_req*.json files in {out_dir}")
        return

    fields = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    variants = sorted({r["variant"] for r in rows if r["variant"]})
    lines = [
        "# TensorRT-LLM SID-GR beam benchmark",
        "",
        f"variant: {', '.join(variants) if variants else 'n/a'}",
        "",
        "| ctx | beam | batch | TRT-LLM ms | prefill ms | decode ms | qps | GR ms | TRT-LLM/GR |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append("| {ctx} | {beam} | {batch} | {trt} | {pre} | {dec} | {qps} | {gr} | {ratio} |".format(
            ctx=r["context_len"],
            beam=r["beam_width"],
            batch=r["batch_requests"],
            trt=fmt(r["trtllm_wall_ms"]),
            pre=fmt(r["trtllm_prefill_ms"]),
            dec=fmt(r["trtllm_decode_ms"]),
            qps=fmt(r["trtllm_qps"]),
            gr=fmt(r["gr_wall_ms"]),
            ratio=fmt(r["trtllm_over_gr"]),
        ))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_summarize(args) -> None:
    summarize(Path(args.out_dir), Path(args.gr_dir) if args.gr_dir else None)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    off = sub.add_parser("offline", help="LLM API batch wall-time sweep")
    off.add_argument("--model", default="Qwen/Qwen3-1.7B")
    off.add_argument("--context-lens", type=parse_int_list, default=[1000, 5000])
    off.add_argument("--beam-widths", type=parse_int_list, default=[256])
    off.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8])
    off.add_argument("--output-len", type=int, default=3,
                     help="Total generated tokens per beam (GR: prefill token + 2 decode steps)")
    off.add_argument("--warmup-runs", type=int, default=1)
    off.add_argument("--repeat", type=int, default=3)
    off.add_argument("--vocab-size", type=int, default=QWEN3_VOCAB_SIZE)
    off.add_argument("--workload-jsonl",
                     help="Load input_ids from a GR make_qwen3_beam_workload.py JSONL instead of generating")
    off.add_argument("--record-outputs", action="store_true",
                     help="Store beam token ids and scores of the last timed run")
    off.add_argument("--out-dir", default="benchmark_artifacts/sid_gr/trtllm_offline")
    off.add_argument("--gr-dir", help="GR sweep dir with gr_ctx*_beam*_req*.json for side-by-side")
    # Engine knobs.
    off.add_argument("--max-num-tokens", type=int,
                     help="Default: max(batch) * max(context_len) so one prefill batch covers the case")
    off.add_argument("--kv-cache-free-gpu-mem-fraction", type=float, default=0.85)
    off.add_argument("--disable-overlap-scheduler", action="store_true")
    off.add_argument("--no-cuda-graph", action="store_true")
    off.add_argument("--attn-backend", help="e.g. TRTLLM, FLASHINFER")
    off.add_argument("--extra-llm-api-options", help="YAML merged into LLM kwargs last")
    off.add_argument("--no-perf-metrics", action="store_true",
                     help="Skip return_perf_metrics (loses prefill/decode split, removes its overhead)")
    off.add_argument("--prefill-cuda-graph",
                     choices=["disabled", "breakable", "piecewise"],
                     default="disabled",
                     help="Capture the prefill (context) forward in a CUDA graph per "
                     "(batch x context_len) token bucket: 'breakable' is the native runner, "
                     "'piecewise' goes through torch.compile. The eager prefill step is "
                     "host-launch bound at these sizes.")
    off.add_argument("--iter-stats", action="store_true",
                     help="Enable executor iteration stats and record per-iteration latency "
                     "(sum vs wall time separates executor time from API/response overhead)")
    off.add_argument("--cuda-profiler-range", action="store_true",
                     help="cudaProfilerStart/Stop around timed runs (needs --single-process)")
    # Process / attention variants (become env vars before tensorrt_llm is imported).
    off.add_argument("--variant", default="default",
                     help="Free-form label stored in every result JSON")
    off.add_argument("--single-process", action="store_true",
                     help="Run the executor in-process (TLLM_WORKER_USE_SINGLE_PROCESS=1); "
                     "removes the IPC hop, matches how the GR engine is timed")
    off.add_argument("--cascade-mmha", dest="cascade_mmha", action="store_true", default=True,
                     help="Shared-prefix cascade decode attention (TRTLLM_ENABLE_CASCADE_MMHA=1, default)")
    off.add_argument("--no-cascade-mmha", dest="cascade_mmha", action="store_false",
                     help="Legacy per-beam MMHA decode attention")
    off.add_argument("--fmha-libs",
                     help="TLLM_FMHA_LIBS value, e.g. '+beam_shared_prefix' for the FlashInfer "
                     "shared-prefix beam library")
    off.add_argument("--beam-max-tail", type=int,
                     help="TLLM_BEAM_SHARED_PREFIX_MAX_TAIL: generated positions per beam the "
                     "beam_shared_prefix library plans for under CUDA graphs (>= output_len - 1)")
    off.add_argument("--env", action="append", metavar="KEY=VALUE",
                     help="Extra environment variable (repeatable), applied before import")
    off.set_defaults(func=cmd_offline)

    on = sub.add_parser("online", help="HTTP client against trtllm-serve")
    on.add_argument("--host", default="127.0.0.1")
    on.add_argument("--port", type=int, default=8000)
    on.add_argument("--served-model-name", help="Default: first id from /v1/models")
    on.add_argument("--context-len", type=int, default=5000)
    on.add_argument("--output-len", type=int, default=3)
    on.add_argument("--beam-width", type=int, default=256)
    on.add_argument("--requests", type=int, default=64)
    on.add_argument("--max-concurrency", type=int, default=4)
    on.add_argument("--warmup-requests", type=int, default=0)
    on.add_argument("--vocab-size", type=int, default=QWEN3_VOCAB_SIZE)
    on.add_argument("--timeout", type=float, default=600.0)
    on.add_argument("--extra-body", help="JSON merged into each request body")
    on.add_argument("--out-dir", default="benchmark_artifacts/sid_gr/trtllm_online")
    on.set_defaults(func=cmd_online)

    summ = sub.add_parser("summarize", help="Rebuild summary.md/csv")
    summ.add_argument("--out-dir", required=True)
    summ.add_argument("--gr-dir")
    summ.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
