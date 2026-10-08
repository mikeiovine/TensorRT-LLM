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
"""Fused beam-tail attention and prefix merge for the beam_shared_prefix FMHA library.

One program per (beam row, KV head) gathers that row's few generated tail
positions straight from the paged KV pool (page/slot indices resolved through
the beam cache indirection by the caller), attends the row's query heads to
them plus the current token, and merges the result with the shared-prefix
partial state (normalized output + base-2 log-sum-exp) into the final output.
It replaces a dozen small eager ops per layer; the unfused torch path in
``fmha/beam_shared_prefix.py`` (``merge_prefix_and_tail``) stays as the
reference and fallback.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _beam_tail_merge_kernel(
    q_ptr, q_stride_r, q_stride_h,
    kcur_ptr, vcur_ptr, cur_stride_r, cur_stride_h,
    pool_k_ptr, pool_v_ptr, pool_stride_page, pool_stride_h, pool_stride_n,
    tail_page_ptr, tail_slot_ptr, tail_valid_ptr, tail_stride_r,
    prefix_out_ptr, po_stride_r, po_stride_h,
    prefix_lse_ptr, pl_stride_r,
    prefix_empty_ptr,
    out_ptr, o_stride_r, o_stride_h,
    scale_log2e,
    TAIL: tl.constexpr,
    TAIL_PAD: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Offsets are formed in 64 bits: the pool page stride spans every layer of
    # the KV pool (layers * 2 * heads * page_size * head_dim elements), so
    # page * stride exceeds int32 after roughly a thousand pages.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, HEAD_DIM).to(tl.int64)

    # Current token's K/V for this KV head (always attended, never masked).
    k_cur = tl.load(kcur_ptr + row * cur_stride_r + kv_head * cur_stride_h + d).to(tl.float32)
    v_cur = tl.load(vcur_ptr + row * cur_stride_r + kv_head * cur_stride_h + d).to(tl.float32)

    if TAIL > 0:
        t = tl.arange(0, TAIL_PAD)
        in_tail = t < TAIL
        page = tl.load(tail_page_ptr + row * tail_stride_r + t, mask=in_tail, other=0).to(tl.int64)
        slot = tl.load(tail_slot_ptr + row * tail_stride_r + t, mask=in_tail, other=0).to(tl.int64)
        valid = tl.load(tail_valid_ptr + row * tail_stride_r + t, mask=in_tail, other=0)
        valid = (valid != 0) & in_tail
        kv_off = (page[:, None] * pool_stride_page + kv_head * pool_stride_h
                  + slot[:, None] * pool_stride_n + d[None, :])
        k_tail = tl.load(pool_k_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v_tail = tl.load(pool_v_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)

    empty = tl.load(prefix_empty_ptr + row) != 0

    for g in tl.static_range(GROUP):
        head = kv_head * GROUP + g
        q = tl.load(q_ptr + row * q_stride_r + head * q_stride_h + d).to(tl.float32)
        s_cur = tl.sum(q * k_cur, axis=0) * scale_log2e
        if TAIL > 0:
            s_tail = tl.sum(k_tail * q[None, :], axis=1) * scale_log2e
            s_tail = tl.where(valid, s_tail, float("-inf"))
            m_tail = tl.maximum(tl.max(s_tail, axis=0), s_cur)
            p_tail = tl.exp2(s_tail - m_tail)
            p_cur = tl.exp2(s_cur - m_tail)
            l_tail = tl.sum(p_tail, axis=0) + p_cur
            o_tail = tl.sum(p_tail[:, None] * v_tail, axis=0) + p_cur * v_cur
        else:
            m_tail = s_cur
            l_tail = 1.0
            o_tail = v_cur

        lse_pre = tl.load(prefix_lse_ptr + row * pl_stride_r + head)
        lse_pre = tl.where(empty, float("-inf"), lse_pre)
        o_pre = tl.load(prefix_out_ptr + row * po_stride_r + head * po_stride_h + d).to(tl.float32)
        o_pre = tl.where(empty, 0.0, o_pre)

        m_all = tl.maximum(lse_pre, m_tail)
        w_pre = tl.exp2(lse_pre - m_all)
        w_tail = tl.exp2(m_tail - m_all)
        out = (o_pre * w_pre + o_tail * w_tail) / (w_pre + l_tail * w_tail)
        tl.store(out_ptr + row * o_stride_r + head * o_stride_h + d,
                 out.to(out_ptr.dtype.element_ty))


def _next_pow2(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


def beam_tail_merge(
    q: torch.Tensor,
    k_cur: torch.Tensor,
    v_cur: torch.Tensor,
    pool: torch.Tensor,
    tail_page: Optional[torch.Tensor],
    tail_slot: Optional[torch.Tensor],
    tail_valid: Optional[torch.Tensor],
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    prefix_empty: torch.Tensor,
    out: torch.Tensor,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """Fused tail attention + prefix merge; writes ``out`` and returns it.

    Args:
        q: [rows, num_heads, head_dim] queries (any strides along rows/heads).
        k_cur, v_cur: [rows, num_kv_heads, head_dim] current token K/V.
        pool: [pages, 2, num_kv_heads, page_size, head_dim] KV pool view
            (K at index 0, V at index 1 of dim 1).
        tail_page, tail_slot: [rows, tail] int64 pool page/slot of each past
            generated position, or None when the tail is empty.
        tail_valid: [rows, tail] bool, position exists.
        prefix_out: [rows, num_heads, head_dim] normalized prefix attention.
        prefix_lse: [rows, num_heads] fp32 base-2 log-sum-exp of the prefix.
        prefix_empty: [rows] bool, rows whose prefix state must be ignored.
        out: [rows, num_heads, head_dim] destination.
        sm_scale: softmax scale applied to raw scores.
    """
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k_cur.shape[1]
    group = num_heads // num_kv_heads
    tail = 0 if tail_page is None else tail_page.shape[1]
    if tail > 0:
        assert tail_slot is not None and tail_valid is not None
        tail_page = tail_page.contiguous()
        tail_slot = tail_slot.contiguous()
        valid_i8 = tail_valid.contiguous().view(torch.int8)
        tail_stride = tail_page.stride(0)
    else:
        # Unused by the kernel when TAIL == 0; pass any valid pointer.
        tail_page = tail_slot = prefix_empty
        valid_i8 = prefix_empty.view(torch.int8)
        tail_stride = 0
    pool_k = pool[:, 0]
    pool_v = pool[:, 1]
    assert pool_k.stride(-1) == 1 and q.stride(-1) == 1 and out.stride(-1) == 1
    assert prefix_out.stride(-1) == 1
    grid = (rows, num_kv_heads)
    _beam_tail_merge_kernel[grid](
        q, q.stride(0), q.stride(1),
        k_cur, v_cur, k_cur.stride(0), k_cur.stride(1),
        pool_k, pool_v, pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
        tail_page, tail_slot, valid_i8, tail_stride,
        prefix_out, prefix_out.stride(0), prefix_out.stride(1),
        prefix_lse, prefix_lse.stride(0),
        prefix_empty.view(torch.int8),
        out, out.stride(0), out.stride(1),
        sm_scale * 1.4426950408889634,
        TAIL=tail,
        TAIL_PAD=_next_pow2(tail),
        GROUP=group,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out


__all__ = ["beam_tail_merge"]
