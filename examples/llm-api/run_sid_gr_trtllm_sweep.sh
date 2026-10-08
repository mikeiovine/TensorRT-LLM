#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A/B sweep of the TensorRT-LLM SID-GR beam benchmark across the decode-attention
# variants, one output directory per variant, then a combined comparison table.
#
#   examples/llm-api/run_sid_gr_trtllm_sweep.sh                 # quick: ctx 1000, batch 1
#   FULL=1 examples/llm-api/run_sid_gr_trtllm_sweep.sh          # ctx 1000/5000, batch 1 2 4 8
#   VARIANTS="cascade flashinfer" examples/llm-api/run_sid_gr_trtllm_sweep.sh
#   GR_DIR=/path/to/gr_offline examples/llm-api/run_sid_gr_trtllm_sweep.sh   # side-by-side with GR
#
# Variants:
#   mmha            legacy per-beam MMHA decode attention, worker in a separate process
#   cascade         TRTLLM_ENABLE_CASCADE_MMHA=1 (shared-prefix C++ kernels)
#   flashinfer      cascade + TLLM_FMHA_LIBS=+beam_shared_prefix (FlashInfer shared-prefix library)
#   flashinfer_pcg  flashinfer + piecewise (torch.compile) prefill CUDA graphs: attention stays
#                   eager, works at every batch size
#   flashinfer_bcg  flashinfer + breakable prefill CUDA graphs: attention captured too (fastest
#                   context step); prototype runner, validated at batch 1 only so far
# Every variant except "mmha" runs the executor in-process (--single-process), which is how
# the GR engine is timed. Set SINGLE_PROCESS=0 to keep the worker process everywhere.
#
# Knobs: MODEL, CONTEXT_LENS, BEAM_WIDTHS, BATCH_SIZES, OUTPUT_LEN, WARMUP_RUNS, REPEAT,
#        OUT_ROOT, EXTRA_ARGS (appended to every run, e.g. "--no-perf-metrics").
# Debugging the flashinfer variant: CHECK=1 runs it with
# TLLM_BEAM_SHARED_PREFIX_CHECK=1 CUDA_LAUNCH_BLOCKING=1 (synchronizes after every stage
# and validates the page table / prefix attention; numbers from that run are not timings).
# FLASHINFER_BACKEND=fa2|fa3|auto selects the FlashInfer prefill kernels (default fa2).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${HERE}/bench_sid_gr_beam.py"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
if [[ "${FULL:-0}" == "1" ]]; then
  CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
  BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
  REPEAT="${REPEAT:-3}"
else
  CONTEXT_LENS="${CONTEXT_LENS:-1000}"
  BATCH_SIZES="${BATCH_SIZES:-1}"
  REPEAT="${REPEAT:-3}"
fi
BEAM_WIDTHS="${BEAM_WIDTHS:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
OUT_ROOT="${OUT_ROOT:-benchmark_artifacts/sid_gr/trtllm_sweep_$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-mmha cascade flashinfer flashinfer_pcg flashinfer_bcg}"
SINGLE_PROCESS="${SINGLE_PROCESS:-1}"
GR_DIR="${GR_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUT_ROOT}"
echo "== TensorRT-LLM SID-GR sweep =="
echo "model=${MODEL} ctx=${CONTEXT_LENS} beam=${BEAM_WIDTHS} batch=${BATCH_SIZES} output_len=${OUTPUT_LEN}"
echo "variants=${VARIANTS} out=${OUT_ROOT}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true

common=(
  --model "${MODEL}"
  --context-lens "${CONTEXT_LENS}"
  --beam-widths "${BEAM_WIDTHS}"
  --batch-sizes "${BATCH_SIZES}"
  --output-len "${OUTPUT_LEN}"
  --warmup-runs "${WARMUP_RUNS}"
  --repeat "${REPEAT}"
)
# ITER_STATS=0 drops the per-iteration executor stats (they add a little host work).
if [[ "${ITER_STATS:-1}" == "1" ]]; then
  common+=(--iter-stats)
fi
# Per-request perf metrics make the client record metrics for every one of the 256
# beams on the final response; off by default for timing runs (PERF_METRICS=1 enables
# them for the queue/prefill/decode split).
if [[ "${PERF_METRICS:-0}" != "1" ]]; then
  common+=(--no-perf-metrics)
fi
# Tail positions a beam carries before its last token.
max_tail=$(( OUTPUT_LEN > 1 ? OUTPUT_LEN - 1 : 1 ))

for variant in ${VARIANTS}; do
  out_dir="${OUT_ROOT}/${variant}"
  args=("${common[@]}" --out-dir "${out_dir}" --variant "${variant}")
  case "${variant}" in
    mmha)
      args+=(--no-cascade-mmha)
      ;;
    cascade)
      args+=(--cascade-mmha)
      [[ "${SINGLE_PROCESS}" == "1" ]] && args+=(--single-process)
      ;;
    flashinfer|flashinfer_pcg|flashinfer_bcg)
      args+=(--cascade-mmha --fmha-libs "+beam_shared_prefix" --beam-max-tail "${max_tail}")
      [[ "${SINGLE_PROCESS}" == "1" ]] && args+=(--single-process)
      if [[ "${variant}" != "flashinfer" ]]; then
        if [[ "${variant}" == "flashinfer_pcg" ]]; then
          args+=(--prefill-cuda-graph piecewise)
        else
          args+=(--prefill-cuda-graph breakable)
        fi
        # SPEC_BEAM_D2H=0 keeps the per-step beam-history snapshot.
        [[ "${SPEC_BEAM_D2H:-1}" == "1" ]] && args+=(--speculative-beam-d2h)
      fi
      if [[ -n "${FLASHINFER_BACKEND:-}" ]]; then
        args+=(--env "TLLM_BEAM_SHARED_PREFIX_BACKEND=${FLASHINFER_BACKEND}")
      fi
      if [[ "${CHECK:-0}" == "1" ]]; then
        args+=(--env TLLM_BEAM_SHARED_PREFIX_CHECK=1 --env CUDA_LAUNCH_BLOCKING=1 --warmup-runs 1 --repeat 1)
      fi
      ;;
    *)
      echo "unknown variant ${variant}" >&2
      exit 2
      ;;
  esac
  # shellcheck disable=SC2206
  args+=(${EXTRA_ARGS})
  echo
  echo "== variant: ${variant} =="
  if ! python "${BENCH}" offline "${args[@]}" 2>&1 | tee "${OUT_ROOT}/${variant}.log"; then
    echo "variant ${variant} FAILED (see ${OUT_ROOT}/${variant}.log); continuing" >&2
  fi
done

echo
echo "== comparison =="
python - "${OUT_ROOT}" "${GR_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import os

root = Path(sys.argv[1])
gr_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
# recsys-examples examples/sid-gr-inference README, section 6 (H100 80GB HBM3, Qwen3-1.7B,
# beam 256, 3 output tokens, radix cache off): wall ms by (ctx, beam, batch).
GR_README_MS = {
    (1000, 256, 1): 17.611, (1000, 256, 2): 27.768, (1000, 256, 4): 47.736, (1000, 256, 8): 93.230,
    (5000, 256, 1): 42.255, (5000, 256, 2): 80.904, (5000, 256, 4): 154.224, (5000, 256, 8): 307.917,
}
use_readme = gr_dir is None and os.environ.get("GR_README", "1") == "1"
rows = {}
variants = []
for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    variants.append(variant_dir.name)
    for path in sorted(variant_dir.glob("trtllm_ctx*_beam*_req*.json")):
        data = json.loads(path.read_text())
        key = (data["context_len"], data["beam_width"], data["requests"])
        rows.setdefault(key, {})[variant_dir.name] = data

def gr_ms(key):
    if gr_dir is None:
        return GR_README_MS.get(key) if use_readme else None
    path = gr_dir / f"gr_ctx{key[0]}_beam{key[1]}_req{key[2]}.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["wall_ms_median"])

gr_label = "GR ms (README)" if use_readme and gr_dir is None else "GR ms"
header = ["ctx", "beam", "batch"] + [f"{v} ms" for v in variants] + [gr_label, "best / GR"]
lines = ["| " + " | ".join(header) + " |", "|" + " ---: |" * len(header)]
for key in sorted(rows):
    cells = [str(k) for k in key]
    best = None
    for v in variants:
        d = rows[key].get(v)
        if d is None or d.get("wall_ms_median") is None:
            cells.append("")
        else:
            wall = float(d["wall_ms_median"])
            best = wall if best is None else min(best, wall)
            pre = d.get("prefill_ms_median")
            dec = d.get("decode_ms_median")
            queue = d.get("queue_ms_median")
            split = ""
            if pre is not None and dec is not None:
                split = f" (q {queue:.1f} + p {pre:.1f} + d {dec:.1f})" if queue is not None \
                    else f" (p {pre:.1f} + d {dec:.1f})"
            iters = d.get("iter_latency_sum_ms_median")
            if iters is not None:
                split += f" [iters {iters:.1f}]"
            cells.append(f"{d['wall_ms_median']:.2f}{split}")
    gr = gr_ms(key)
    cells.append(f"{gr:.2f}" if gr is not None else "")
    cells.append(f"{best / gr:.2f}x" if gr is not None and best is not None else "")
    lines.append("| " + " | ".join(cells) + " |")
text = "\n".join(lines)
if use_readme and gr_dir is None:
    text += ("\n\nGR column: recsys-examples sid-gr-inference README (H100 80GB HBM3), not "
             "measured on this machine.")
(root / "comparison.md").write_text(text + "\n", encoding="utf-8")
print(text)
print(f"\nwrote {root / 'comparison.md'}  (cells: wall ms (q queue + p prefill + d decode) "
      "[sum of executor iteration latencies])")
PY
