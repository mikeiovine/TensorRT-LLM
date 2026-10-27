#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test Helix CP in aggregated serving, optionally against a plain-TP baseline.

Runs short prompts plus a long "needle" prompt whose passphrase sits near the
start of the context, so answering it needs KV blocks that other CP ranks
stored during the prefill. With --baseline, the same inputs run again as plain
TP over the same GPUs (tp * cp) and the greedy outputs are compared.

Examples:
  # DeepSeek-V3-Lite on 4 GPUs, Helix CP4, 16K-token needle prompt
  python run_helix_agg.py --model_dir $LLM_MODELS_ROOT/DeepSeek-V3-Lite/bf16 --cp 4

  # Chunked prefill + MTP, compare with TP4
  python run_helix_agg.py --model_dir $LLM_MODELS_ROOT/DeepSeek-V3-Lite/bf16 \\
      --tp 2 --cp 2 --chunk 2048 --mtp 2 --prompt_len 32768 --baseline
"""
import argparse
import random
import time

from transformers import AutoTokenizer

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

SHORT_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "Explain why the sky is blue in two sentences.",
]
FILLER = [
    "The committee reviewed the quarterly report and noted steady progress.",
    "Rain fell softly over the hills while the village slept.",
    "Engineers replaced the bearings and recalibrated the turbine.",
    "The library extended its hours for the examination period.",
    "A flock of starlings wheeled above the harbor at dusk.",
]


def needle_prompt(tokenizer, num_tokens, passphrase):
    """A ~num_tokens prompt whose answer is stated near its start."""
    rng = random.Random(0)
    head = f"Remember this: the secret passphrase is {passphrase}.\n"
    question = "\nQuestion: what is the secret passphrase? Answer: the secret passphrase is"
    budget = num_tokens - len(tokenizer.encode(head + question))
    lines, used = [], 0
    while used < budget:
        line = f"{len(lines)}. {rng.choice(FILLER)}\n"
        used += len(tokenizer.encode(line, add_special_tokens=False))
        lines.append(line)
    return head + "".join(lines) + question


def run(args, prompts, helix):
    world = args.tp * args.cp
    max_seq_len = args.prompt_len + args.max_tokens + 1024
    kwargs = dict(
        tensor_parallel_size=args.tp if helix else world,
        moe_expert_parallel_size=world,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=args.free_mem,
            tokens_per_block=args.tokens_per_block,
            enable_block_reuse=False,
        ),
        enable_chunked_prefill=args.chunk is not None,
        max_num_tokens=args.chunk or max(8192, max_seq_len),
        max_seq_len=max_seq_len,
        max_batch_size=args.batch,
        disable_overlap_scheduler=args.no_overlap,
        cuda_graph_config=None if args.no_cuda_graph else CudaGraphConfig(
            enable_padding=True, max_batch_size=args.batch),
        speculative_config=MTPDecodingConfig(max_draft_len=args.mtp) if args.mtp else None,
        trust_remote_code=True,
    )
    if helix:
        kwargs.update(
            context_parallel_size=args.cp,
            cp_config={"cp_type": "HELIX", "tokens_per_block": args.tokens_per_block},
        )
    label = f"helix tp{args.tp}cp{args.cp}" if helix else f"baseline tp{world}"
    with LLM(model=args.model_dir, **kwargs) as llm:
        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)
        llm.generate(prompts[:1], sampling)  # warm up outside the timing
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling)
        dt = time.perf_counter() - t0
    num_out = sum(len(o.outputs[0].token_ids) for o in outs)
    line = f"=== {label}: {dt:.2f}s, {num_out / dt:.1f} output tok/s"
    if args.mtp:
        al = sum(o.avg_decoded_tokens_per_iter for o in outs) / len(outs)
        line += f", AL={al:.2f} tokens/iter"
    print(line + " ===")
    for o in outs:
        prompt = o.prompt
        if len(prompt) > 80:
            prompt = f"<{len(o.prompt_token_ids)}-token needle prompt>"
        print(f"{prompt!r} -> {o.outputs[0].text!r}")
    return [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].text for o in outs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cp", type=int, default=4, help="Helix context-parallel size (>= 2)")
    p.add_argument("--prompt_len", type=int, default=16384, help="Needle prompt length in tokens")
    p.add_argument("--num_needles", type=int, default=2, help="Needle prompts in the batch")
    p.add_argument("--chunk", type=int, default=None, help="Enable chunked prefill with this chunk")
    p.add_argument("--mtp", type=int, default=0, help="MTP draft length (0 disables)")
    p.add_argument("--max_tokens", type=int, default=48)
    p.add_argument("--batch", type=int, default=16, help="max_batch_size")
    p.add_argument("--tokens_per_block", type=int, default=32)
    p.add_argument("--free_mem", type=float, default=0.5,
                   help="KV cache fraction; helix does not profile prefill activations")
    p.add_argument("--no_overlap", action="store_true", help="Disable the overlap scheduler")
    p.add_argument("--no_cuda_graph", action="store_true")
    p.add_argument("--baseline", action="store_true", help="Also run plain TP and compare")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    passphrases = ["copper-417", "violet-82", "harbor-9031", "maple-26"]
    passphrases = (passphrases * args.num_needles)[:args.num_needles]
    prompts = SHORT_PROMPTS + [needle_prompt(tokenizer, args.prompt_len, s) for s in passphrases]

    helix_ids, helix_text = run(args, prompts, helix=True)
    found = [s in t for s, t in zip(passphrases, helix_text[len(SHORT_PROMPTS):])]
    print(f"Needle recall: {sum(found)}/{len(found)}")
    ok = all(found)

    if args.baseline:
        base_ids, _ = run(args, prompts, helix=False)
        for i, (h, b) in enumerate(zip(helix_ids, base_ids)):
            same = next((j for j, (x, y) in enumerate(zip(h, b)) if x != y), min(len(h), len(b)))
            status = "match" if h == b else f"diverges at token {same}"
            print(f"prompt {i}: {status}")
        # Greedy outputs may legitimately diverge late (different reduction
        # order); an early divergence on many prompts points at a bug.
        ok = ok and sum(h == b for h, b in zip(helix_ids, base_ids)) >= len(prompts) // 2
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
