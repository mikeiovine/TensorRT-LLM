# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the opt-in beam_shared_prefix FMHA library.

The merge math is covered on the CPU against a dense reference; the
end-to-end test runs Qwen3-0.6B beam search with and without the library and
requires a GPU plus the model checkpoint.
"""

import math
import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import (
    LIB_NAME, merge_prefix_and_tail)

_LOG2E = 1.4426950408889634


def _reference_prefix_state(q: torch.Tensor, k_prefix: torch.Tensor,
                            v_prefix: torch.Tensor, num_kv_heads: int,
                            sm_scale: float):
    """Dense prefix attention plus its base-2 LSE, FlashInfer's convention."""
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    q_grouped = q.view(rows, num_kv_heads, group, head_dim).float()
    scores = torch.einsum("rkgd,rtkd->rkgt", q_grouped, k_prefix.float())
    scores = scores * sm_scale
    out = torch.einsum("rkgt,rtkd->rkgd", torch.softmax(scores, dim=-1),
                       v_prefix.float())
    lse2 = torch.logsumexp(scores, dim=-1) * _LOG2E
    return out.view(rows, num_heads, head_dim), lse2.view(rows, num_heads)


@pytest.mark.parametrize("tail_len, with_mask", [(1, False), (3, False),
                                                  (4, True)])
@pytest.mark.parametrize("group", [1, 2, 4])
def test_merge_prefix_and_tail_matches_dense_attention(tail_len, with_mask,
                                                       group):
    gen = torch.Generator().manual_seed(1234)
    rows, num_kv_heads, head_dim, prefix_len = 6, 2, 64, 37
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(rows, num_heads, head_dim, generator=gen)
    k_prefix = torch.randn(rows, prefix_len, num_kv_heads, head_dim,
                           generator=gen)
    v_prefix = torch.randn(rows, prefix_len, num_kv_heads, head_dim,
                           generator=gen)
    k_tail = torch.randn(rows, tail_len, num_kv_heads, head_dim, generator=gen)
    v_tail = torch.randn(rows, tail_len, num_kv_heads, head_dim, generator=gen)
    tail_valid = None
    if with_mask:
        # Ragged tails: row r keeps its last (r % tail_len) + 1 positions.
        keep = torch.tensor([(r % tail_len) + 1 for r in range(rows)])
        tail_valid = torch.arange(tail_len).view(1, -1) >= (tail_len -
                                                            keep).view(-1, 1)

    prefix_out, prefix_lse = _reference_prefix_state(q, k_prefix, v_prefix,
                                                     num_kv_heads, sm_scale)
    merged = merge_prefix_and_tail(q,
                                   k_tail,
                                   v_tail,
                                   tail_valid,
                                   prefix_out,
                                   prefix_lse,
                                   num_kv_heads=num_kv_heads,
                                   sm_scale=sm_scale)

    # Dense reference over the concatenated prefix + (valid) tail.
    k_all = torch.cat([k_prefix, k_tail], dim=1)
    v_all = torch.cat([v_prefix, v_tail], dim=1)
    q_grouped = q.view(rows, num_kv_heads, group, head_dim)
    scores = torch.einsum("rkgd,rtkd->rkgt", q_grouped, k_all) * sm_scale
    if tail_valid is not None:
        valid = torch.cat(
            [torch.ones(rows, prefix_len, dtype=torch.bool), tail_valid],
            dim=1)
        scores = scores.masked_fill(~valid.view(rows, 1, 1, -1), float("-inf"))
    expected = torch.einsum("rkgt,rtkd->rkgd", torch.softmax(scores, dim=-1),
                            v_all).view(rows, num_heads, head_dim)
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-4)


def _check_fused_tail_merge(tail_len, group, pool, page_lo, page_hi):
    """Run the Triton tail+merge kernel against merge_prefix_and_tail on
    ``pool`` ([pages, 2, kv_heads, page_size, head_dim], any page stride),
    drawing tail pages from [page_lo, page_hi)."""
    from tensorrt_llm._torch.attention.kernels.beam_tail_merge import \
        beam_tail_merge

    torch.manual_seed(0)
    device = pool.device
    rows = 10
    _, _, num_kv_heads, page_size, head_dim = pool.shape
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)
    # Packed QKV row layout, like the attention input the library receives.
    qkv = torch.randn(rows, (num_heads + 2 * num_kv_heads) * head_dim,
                      device=device, dtype=torch.bfloat16)
    q = qkv[:, :num_heads * head_dim].view(rows, num_heads, head_dim)
    k = qkv[:, num_heads * head_dim:(num_heads + num_kv_heads) *
            head_dim].view(rows, num_kv_heads, head_dim)
    v = qkv[:, (num_heads + num_kv_heads) * head_dim:].view(
        rows, num_kv_heads, head_dim)
    prefix_out = torch.randn(rows, num_heads, head_dim, device=device,
                             dtype=torch.bfloat16)
    prefix_lse = torch.randn(rows, num_heads, device=device) * 3 + 10
    prefix_empty = torch.zeros(rows, dtype=torch.bool, device=device)
    prefix_empty[-1] = True
    if tail_len > 0:
        # int32, like the KV manager's block-offset table the library derives
        # them from; the kernel must widen before forming pool offsets.
        tail_page = torch.randint(page_lo, page_hi, (rows, tail_len),
                                  device=device, dtype=torch.int32)
        tail_slot = torch.randint(0, page_size, (rows, tail_len),
                                  device=device, dtype=torch.int32)
        tail_valid = torch.rand(rows, tail_len, device=device) > 0.3
        k_tail = pool[tail_page, 0, :, tail_slot, :]
        v_tail = pool[tail_page, 1, :, tail_slot, :]
        k_all = torch.cat([k_tail, k.unsqueeze(1)], dim=1)
        v_all = torch.cat([v_tail, v.unsqueeze(1)], dim=1)
        valid = torch.cat(
            [tail_valid, torch.ones(rows, 1, dtype=torch.bool, device=device)],
            dim=1)
    else:
        tail_page = tail_slot = tail_valid = None
        k_all, v_all, valid = k.unsqueeze(1), v.unsqueeze(1), None
    expected = merge_prefix_and_tail(q,
                                     k_all,
                                     v_all,
                                     valid,
                                     prefix_out,
                                     prefix_lse,
                                     num_kv_heads=num_kv_heads,
                                     sm_scale=sm_scale,
                                     prefix_empty=prefix_empty)
    out = torch.empty(rows, num_heads, head_dim, device=device,
                      dtype=torch.bfloat16)
    beam_tail_merge(q, k, v, pool, tail_page, tail_slot, tail_valid,
                    prefix_out, prefix_lse, prefix_empty, out,
                    sm_scale=sm_scale)
    torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("tail_len", [0, 1, 2, 5])
@pytest.mark.parametrize("group", [1, 2])
def test_fused_tail_merge_matches_torch_path(tail_len, group):
    """The Triton tail+merge kernel reproduces merge_prefix_and_tail, reading
    the tail K/V straight from a paged pool through page/slot indices."""
    torch.manual_seed(0)
    num_kv_heads, head_dim, page_size, pages = 2, 128, 32, 7
    pool = torch.randn(pages, 2, num_kv_heads, page_size, head_dim,
                       device="cuda", dtype=torch.bfloat16)
    _check_fused_tail_merge(tail_len, group, pool, 0, pages)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fused_tail_merge_large_page_offsets():
    """Pool offsets past 2**31 elements must not wrap.

    The per-layer pool the library reads is a slice of the manager's
    [pages, layers, 2, ...] buffer, so its page stride covers every layer and
    page * stride leaves int32 range after about a thousand pages on a real
    model. Reproduce that with a layer-strided view whose referenced pages sit
    beyond the overflow point.
    """
    num_kv_heads, head_dim, page_size, layers = 2, 128, 32, 4
    page_elems = layers * 2 * num_kv_heads * page_size * head_dim
    pages = 2**31 // page_elems + 64
    needed = pages * page_elems * 2
    free, _ = torch.cuda.mem_get_info()
    if free < needed + (1 << 30):
        pytest.skip(f"needs {needed / 2**30:.1f} GiB of free GPU memory")
    big = torch.empty(pages, layers, 2, num_kv_heads, page_size, head_dim,
                      device="cuda", dtype=torch.bfloat16)
    page_lo = pages - 64
    big[page_lo:].normal_()
    pool = big[:, 1]
    assert pool.stride(0) == page_elems
    assert page_lo * page_elems > 2**31
    _check_fused_tail_merge(2, 2, pool, page_lo, pages)


def _qwen3_small_path():
    root = llm_models_root(check=False)
    if root is None:
        return None
    path = root / "Qwen3" / "Qwen3-0.6B"
    return path if path.exists() else None


def _run_beam_search(model_path, prompts, beam_width, max_tokens, env):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        llm = LLM(
            model=str(model_path),
            max_beam_width=beam_width,
            max_batch_size=len(prompts),
            max_num_tokens=max(len(p) for p in prompts) * len(prompts),
            max_seq_len=max(len(p) for p in prompts) + max_tokens + 1,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                          free_gpu_memory_fraction=0.3),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[len(prompts)]),
            skip_tokenizer_init=True,
        )
        with llm:
            sampling = SamplingParams(use_beam_search=True,
                                      n=beam_width,
                                      max_tokens=max_tokens,
                                      ignore_eos=True,
                                      end_id=-1,
                                      logprobs=1)
            outputs = llm.generate(prompts, sampling_params=sampling)
        return [[(tuple(beam.token_ids), beam.cumulative_logprob)
                 for beam in out.outputs] for out in outputs]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None,
                    reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_matches_fallback_e2e():
    """Beam search with the library reproduces the fallback kernel's beams.

    Three output tokens over a prompt that spans several KV blocks, two
    requests in one batch, CUDA graphs on: the second and third decode steps
    exercise the indirection-gathered tail, the first the empty tail.
    """
    model_path = _qwen3_small_path()
    vocab = 150000
    prompts = [[1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
               for i, n in enumerate((300, 173))]
    beam_width, max_tokens = 16, 3

    base_env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    reference = _run_beam_search(model_path, prompts, beam_width, max_tokens,
                                 base_env)
    # Check mode synchronizes after every stage and validates the page table
    # and the prefix attention against dense references inside the library.
    candidate = _run_beam_search(
        model_path, prompts, beam_width, max_tokens, base_env | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            "TLLM_BEAM_SHARED_PREFIX_CHECK": "1",
            "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL": str(max_tokens - 1),
        })

    for ref_beams, cand_beams in zip(reference, candidate):
        assert len(cand_beams) == beam_width
        ref_tokens = [tokens for tokens, _ in ref_beams]
        cand_tokens = [tokens for tokens, _ in cand_beams]
        # Kernel numerics differ slightly, so allow the ranking of a few
        # near-tied beams to move; the sets must agree almost entirely.
        overlap = len(set(ref_tokens) & set(cand_tokens)) / beam_width
        assert overlap >= 0.9, (overlap, ref_tokens, cand_tokens)
        assert cand_tokens[0] == ref_tokens[0]
        for (tokens, ref_lp), (_, cand_lp) in zip(
                ref_beams, cand_beams):
            if ref_lp is not None and cand_lp is not None:
                assert abs(ref_lp - cand_lp) < 0.1 + 0.02 * abs(ref_lp)
