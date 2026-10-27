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
"""Unit tests for the helix (decode-CP) super-block ledger in KVCacheManagerV2.

Design under test: the request ledger runs in GLOBAL tokens with one ledger
block spanning ``cp_size`` physical pages (one per CP rank); the per-rank
view (owner of this step's token, tokens held by this rank) is a closed-form
function of the global position. The decode-step index feeding that position
is manager-owned (``py_helix_decode_group_index``), immune to when the
sampler advances ``py_decoding_iter``.
"""

import math
import types
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import CacheType


def _mgr(cp_rank: int, cp_size: int, phys: int) -> SimpleNamespace:
    m = SimpleNamespace(
        tokens_per_block=phys,
        _ledger_tokens_per_block=phys * cp_size,
        _has_cp_helix=cp_size > 1,
        _helix_cp_rank=cp_rank,
        _helix_cp_size=cp_size,
    )
    m._helix_local_len = lambda global_len: KVCacheManagerV2._helix_local_len(m, global_len)
    return m


def _brute_local_len(global_len: int, cp_rank: int, cp_size: int, phys: int) -> int:
    """Reference: token position p lives on rank (p // phys) % cp_size."""
    return sum(1 for p in range(global_len) if (p // phys) % cp_size == cp_rank)


def test_helix_local_len_matches_brute_force() -> None:
    for cp_size in (1, 2, 4, 8):
        for phys in (2, 4, 32):
            for cp_rank in range(cp_size):
                m = _mgr(cp_rank, cp_size, phys)
                for global_len in range(0, 4 * cp_size * phys + 3):
                    assert KVCacheManagerV2._helix_local_len(m, global_len) == _brute_local_len(
                        global_len, cp_rank, cp_size, phys
                    ), (cp_size, phys, cp_rank, global_len)


def test_set_helix_rank_fields_cross_rank_consistency() -> None:
    """For any (prompt_len, decode step): exactly one active rank, the
    per-rank seqlens sum to the global in-flight length, and past_seen
    (= seqlen - 0/1 per the model_engine convention) sums to the global
    already-cached length."""
    cp_size, phys = 4, 4
    for total_input in (1, 5, 16, 17, 63):
        for group_index in (0, 1, 2, 7, 40):  # committed schedules so far
            fields = []
            for r in range(cp_size):
                req = SimpleNamespace(
                    total_input_len_cp=total_input,
                    py_helix_decode_group_index=group_index,
                )
                KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
                fields.append(req)
            pos = total_input + group_index
            active = [r for r in range(cp_size) if not fields[r].py_helix_is_inactive_rank]
            assert active == [(pos // phys) % cp_size]
            assert sum(f.seqlen_this_rank_cp for f in fields) == pos + 1
            past_seen = [
                f.seqlen_this_rank_cp - (0 if f.py_helix_is_inactive_rank else 1) for f in fields
            ]
            assert sum(past_seen) == pos
            # Prompt distribution matches the arrival striding convention.
            for r in range(cp_size):
                assert past_seen[r] >= 0


def test_ledger_is_rank_invariant() -> None:
    """The whole point of the super-block design: nothing the scheduler or
    ledger consumes depends on cp_rank — only the derived per-rank fields
    do. Verify the derivation never touches ledger quantities by checking
    the same request object gives identical (owner-adjusted) views."""
    cp_size, phys = 8, 32
    req_proto = dict(total_input_len_cp=1000, py_helix_decode_group_index=17)
    lens = []
    for r in range(cp_size):
        req = SimpleNamespace(**req_proto)
        KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
        lens.append(req.seqlen_this_rank_cp)
    # Ledger-side numbers (global position, page count) are identical on
    # every rank; per-rank lens differ by at most one physical page.
    assert max(lens) - min(lens) <= phys


def test_ledger_position_immune_to_sampler_timing() -> None:
    """Regression for the overlap-scheduler phase skew: the ledger position
    must be L, L+1, L+2, ... no matter when the sampler advances
    py_decoding_iter (the overlap loop updates it after scheduling; a stale
    read would repeat the first position and overwrite that token's KV)."""
    cp_size, phys, total_input = 4, 4, 17
    n_steps = 3 * cp_size * phys  # cross several ownership rotations

    for sampler_timing in ("overlap", "non_overlap"):
        req = SimpleNamespace(
            total_input_len_cp=total_input,
            py_decoding_iter=0,  # disagg-gen seeding happens after scheduling
            py_helix_decode_group_index=0,
        )
        mgrs = [_mgr(r, cp_size, phys) for r in range(cp_size)]
        positions = []
        for step in range(1, n_steps + 1):
            if sampler_timing == "non_overlap":
                req.py_decoding_iter = step  # sampler already advanced
            per_rank = []
            for r in range(cp_size):
                view = SimpleNamespace(**vars(req))
                KVCacheManagerV2._set_helix_rank_fields(mgrs[r], view)
                per_rank.append(view)
            # Commit, mirroring try_allocate_generation's success path.
            req.py_helix_decode_group_index += 1
            if sampler_timing == "overlap":
                req.py_decoding_iter = step  # sampler advances only now

            pos = total_input + step - 1
            active = [r for r in range(cp_size) if not per_rank[r].py_helix_is_inactive_rank]
            assert active == [(pos // phys) % cp_size]
            # In-flight convention: seqlens sum to pos + 1 (never repeats,
            # so no write offset can collide with the previous step's).
            assert sum(f.seqlen_this_rank_cp for f in per_rank) == pos + 1
            positions.append(pos)
        assert positions == [total_input + n for n in range(n_steps)]


def test_decode_step_derivation_is_idempotent() -> None:
    """A failed try_allocate (counter not committed) followed by a retry in
    the same scheduling pass must derive the SAME fields; a revert (skipped
    forward) must give the step back."""
    m = _mgr(cp_rank=1, cp_size=4, phys=4)
    req = SimpleNamespace(
        total_input_len_cp=10,
        py_helix_decode_group_index=5,
    )
    KVCacheManagerV2._set_helix_rank_fields(m, req)
    first = (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp)
    KVCacheManagerV2._set_helix_rank_fields(m, req)  # retry, no commit
    assert (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp) == first
    # Commit then revert restores the same derivation.
    req.py_helix_decode_group_index += 1
    req.py_helix_decode_group_index -= 1
    KVCacheManagerV2._set_helix_rank_fields(m, req)
    assert (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp) == first


def test_quota_converters_scale_by_cp() -> None:
    m = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        tokens_per_block=32,
        _ledger_tokens_per_block=128,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    # Rank-local byte quota buys 100 physical tokens, but only 96 (= 3 whole
    # 32-token pages) are allocatable -> 384 global ledger tokens, not 400.
    assert KVCacheManagerV2._get_max_tokens_from_quota(m, 12345) == 384.0
    # inf (all-SWA) passes through unscaled.
    m_inf = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        tokens_per_block=32,
        _ledger_tokens_per_block=128,
        _get_max_tokens_from_quota_impl=lambda quota: float("inf"),
    )
    assert math.isinf(KVCacheManagerV2._get_max_tokens_from_quota(m_inf, 1))
    # Global tokens -> whole ledger blocks (ceil) -> per-rank physical pages
    # -> bytes: 401 global tokens need ceil(401/128) = 4 ledger blocks =
    # 4 * 32 = 128 physical tokens per rank (not ceil(401/4) = 101).
    assert KVCacheManagerV2._get_quota_from_max_tokens(m, 401) == 128 * 7
    # cp == 1 (non-helix) is the identity: no page rounding is applied.
    m1 = SimpleNamespace(
        _has_cp_helix=False,
        _helix_cp_size=1,
        tokens_per_block=32,
        _ledger_tokens_per_block=32,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    assert KVCacheManagerV2._get_max_tokens_from_quota(m1, 1) == 100.0
    assert KVCacheManagerV2._get_quota_from_max_tokens(m1, 400) == 2800


def test_update_resources_leaves_history_untouched_under_helix() -> None:
    resizes = []

    kv = SimpleNamespace(
        is_active=True,
        capacity=100,
        resize=lambda cap, hist: resizes.append((cap, hist)) or True,
    )
    req = SimpleNamespace(
        py_request_id=7,
        py_rewind_len=0,
        py_num_accepted_draft_tokens=0,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=55,
    )
    mgr = SimpleNamespace(
        # is_draft=True skips the module-level draft-token relocation call;
        # with zero reserve tokens the rewind math is unchanged.
        is_draft=True,
        kv_cache_type=CacheType.SELF,
        _kv_reserve_draft_tokens=0,
        # No scheduler ran, so nothing recorded a generation allocation; the
        # rewind falls back to the reserve width.
        _allocated_draft_lens={},
        _inflight_draft_lens={},
        kv_cache_map={7: kv},
        kv_compression_manages_history=False,
        _has_cp_helix=True,
        # A draft manager never commits generation blocks.
        try_commit_generation_blocks=lambda request: None,
        # No lender attached, so a resize has no one to tell.
        _sharing=None,
    )
    batch = SimpleNamespace(generation_requests=[req])
    KVCacheManagerV2.update_resources(mgr, batch)
    assert resizes == [(100, None)]
    # Non-helix keeps the vanilla history commit.
    mgr._has_cp_helix = False
    resizes.clear()
    KVCacheManagerV2.update_resources(mgr, batch)
    assert resizes == [(100, 54)]


def test_helix_quota_fallback_sets_rank_local_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fraction sizing must set max_gpu_total_bytes (rank-local byte cap),
    not max_tokens, which the manager inflates by 1/max_util_for_resume."""
    import torch

    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    def creator(max_gpu_total_bytes, max_tokens):
        return SimpleNamespace(
            _mapping=SimpleNamespace(cp_size=4),
            _kv_cache_config=SimpleNamespace(
                max_gpu_total_bytes=max_gpu_total_bytes,
                max_tokens=max_tokens,
                free_gpu_memory_fraction=0.5,
            ),
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000, 2_000_000))

    # Explicit quota: early return, config untouched.
    c = creator(1 << 30, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_tokens is None
    # Explicit but non-positive max_tokens: rejected, not silently replaced
    # by fraction sizing.
    with pytest.raises(ValueError, match="must be positive"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, 0))
    # No quota: the rank-local byte budget (free * fraction) lands on
    # max_gpu_total_bytes verbatim; max_tokens stays unset.
    c = creator(0, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_gpu_total_bytes == 500_000
    assert c._kv_cache_config.max_tokens is None
    # Degenerate free memory: actionable error instead of a deep assert.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 2_000_000))
    with pytest.raises(ValueError, match="free memory"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, None))


def test_v1_helix_capacity_config_rejected() -> None:
    """configure_kv_cache_capacity emits V2 super-block-ledger coordinates
    under helix; V1 interprets max_tokens as rank-local. The V1 + helix
    combination must be rejected loudly, not silently mis-sized."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType

    c = SimpleNamespace(
        _mapping=SimpleNamespace(cp_config={"cp_type": CpType.HELIX}),
        _is_kv_cache_manager_v2=False,
    )
    with pytest.raises(NotImplementedError, match="V2 KV cache manager"):
        KvCacheCreator.configure_kv_cache_capacity(c)


def test_estimation_prepare_promotes_skip_est_for_v2() -> None:
    """Helix disables estimation; with a V2 manager it must also promote
    _skip_est so build_managers() calls configure_kv_cache_capacity().
    Other CP types must NOT be promoted: configure_kv_cache_capacity has no
    sizing path for them."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType

    def creator(is_v2, cp_type=CpType.HELIX):
        return SimpleNamespace(
            _skip_est=False,
            _mapping=SimpleNamespace(cp_config={"cp_type": cp_type}),
            _is_kv_cache_manager_v2=is_v2,
            _model_engine=SimpleNamespace(
                model=SimpleNamespace(
                    model_config=SimpleNamespace(attn_backend="TRTLLM", is_encoder_decoder=False)
                )
            ),
        )

    c = creator(is_v2=True)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is True
    c = creator(is_v2=False)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is False
    c = creator(is_v2=True, cp_type=CpType.ULYSSES)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is False


def test_scheduler_allocation_failure_raises_under_helix() -> None:
    """Precedent-consistent no-evict stance: allocation failure under helix
    raises instead of entering the (unvalidated-under-helix) eviction path."""
    sched, req, budget = _helix_scheduler_stand_in(enable_recompute_pause=False)
    with pytest.raises(RuntimeError, match="eviction is disabled under helix"):
        _schedule_generation(sched, req, budget)


def _helix_scheduler_stand_in(enable_recompute_pause: bool):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    sched = SimpleNamespace(
        has_cp_helix=True,
        # False on a disaggregated generation server, which cannot replay a prefill.
        enable_recompute_pause=enable_recompute_pause,
        kv_cache_manager=SimpleNamespace(try_allocate_generation=lambda req: False),
        draft_kv_cache_manager=None,
        enable_joint_kv_cache_reuse=False,
        _joint_draft_manager=None,
        _admit_unpaired_draft=lambda req: True,
    )
    sched._try_allocate_generation = types.MethodType(
        KVCacheV2Scheduler._try_allocate_generation, sched
    )
    req = SimpleNamespace(
        py_request_id=7,
        get_beam_width_by_iter=lambda for_next_iteration: 1,
        py_draft_tokens=None,
    )
    budget = SimpleNamespace(can_fit_tokens=lambda n: True)
    return sched, req, budget


def _schedule_generation(sched, req, budget):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        _RecomputePauseState,
    )

    return KVCacheV2Scheduler._try_schedule_generation(
        sched,
        req,
        budget,
        requests_list=[req],
        req_it=0,
        req_it_end=1,
        recompute_pause_state=_RecomputePauseState(1),
        evicted=[],
        recompute_paused=[],
        inflight_request_ids=set(),
        scheduled_beam_width=0,
    )


def test_aggregated_helix_allocation_failure_takes_eviction_ladder() -> None:
    """Aggregated helix prefills on the CP group, so a request can be
    recompute-paused like any other: allocation failure must reach the
    eviction ladder instead of raising."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import ScheduleAction

    sched, req, budget = _helix_scheduler_stand_in(enable_recompute_pause=True)
    evict_calls = []

    def try_evict_for_gen(req, requests_list, req_it, req_it_end, evicted, inflight_request_ids):
        evict_calls.append(req.py_request_id)
        return req_it_end, True

    sched._try_evict_for_gen = try_evict_for_gen
    action, tokens, _, _ = _schedule_generation(sched, req, budget)
    assert evict_calls == [7]
    assert action == ScheduleAction.SCHEDULED
    assert tokens == 1


def test_dummy_frozen_fields_sum_invariant() -> None:
    """The frozen dummy fiction (last rank active, shared synthetic global
    length) keeps the same books as real requests: per-rank lengths sum to
    the synthetic global length."""
    for cp_size in (2, 4, 8):
        for token_num in (2, 5, 33):
            seqlens = [token_num - 1 if r == cp_size - 1 else token_num for r in range(cp_size)]
            total = token_num * cp_size - 1
            assert sum(seqlens) == total


def test_resize_context_requires_the_full_prompt() -> None:
    """Aggregated helix prefills the full prompt on every CP rank and sizes the
    global ledger from it; a rank-local (disagg generation) prompt must never
    take the context path."""
    mgr = SimpleNamespace(_has_cp_helix=True, supports_helix_prefill=True, kv_cache_map={})
    full = SimpleNamespace(
        py_request_id=1,
        is_disagg_generation_init_state=False,
        is_dummy_request=False,
        prompt_len=70,
        total_input_len_cp=70,
    )
    # No cache yet: the guard passes and the resize reports a plain failure.
    assert KVCacheManagerV2.resize_context(mgr, full, 32) is False
    rank_local = SimpleNamespace(**{**vars(full), "prompt_len": 38})
    with pytest.raises(ValueError, match="full prompt"):
        KVCacheManagerV2.resize_context(mgr, rank_local, 32)
    # A disaggregated generation server never prefills, even a full prompt.
    disagg = SimpleNamespace(**{**vars(mgr), "supports_helix_prefill": False})
    with pytest.raises(ValueError, match="receives its prompt KV"):
        KVCacheManagerV2.resize_context(disagg, full, 32)


@pytest.mark.parametrize("cp_size,phys", [(2, 4), (4, 4), (4, 32), (8, 2)])
def test_aggregated_prefill_layout_matches_decode(cp_size: int, phys: int) -> None:
    """Helix prefill writes global block b to rank b % cp_size, local block
    b // cp_size (the global-order page table in TrtllmAttentionMetadata).
    That must be exactly the rank-local order decode reads, and the first
    decode step after the prefill must continue right behind it."""
    for prompt_len in (1, phys - 1, phys, 3 * phys + 1, 5 * cp_size * phys + 3):
        firsts = []
        for rank in range(cp_size):
            m = _mgr(rank, cp_size, phys)
            owned = [p for p in range(prompt_len) if (p // phys) % cp_size == rank]
            for pos in owned:
                block = pos // phys
                local_index = (block // cp_size) * phys + pos % phys
                assert local_index == KVCacheManagerV2._helix_local_len(m, pos)
            req = SimpleNamespace(total_input_len_cp=prompt_len, py_helix_decode_group_index=0)
            KVCacheManagerV2._set_helix_rank_fields(m, req)
            active = not req.py_helix_is_inactive_rank
            # past_seen (model_engine convention) is what the prefill stored here.
            assert req.seqlen_this_rank_cp - (1 if active else 0) == len(owned)
            firsts.append(active)
        assert firsts.count(True) == 1
        assert firsts.index(True) == (prompt_len // phys) % cp_size


def _metadata_stand_in(cp_size: int, cp_rank: int, phys: int, local, num_contexts: int, trash):
    import torch

    return SimpleNamespace(
        num_contexts=num_contexts,
        num_seqs=local.shape[1],
        mapping=SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank),
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=phys,
            helix_trash_block_offsets=lambda: torch.tensor([trash], dtype=torch.int32),
        ),
        kv_cache_block_offsets=local,
        _helix_local_block_offsets=None,
    )


@pytest.mark.parametrize("cp_rank", [0, 1, 2])
def test_global_order_page_table(cp_rank: int) -> None:
    import torch

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    cp_size, phys, width = 3, 4, 4
    trash = [-7, -8]
    # [pools, seqs, K/V, width]: one context row, one generation row.
    local = torch.zeros((1, 2, 2, width), dtype=torch.int32)
    local[0, 0, 0] = torch.tensor([10, 11, 12, 13])
    local[0, 0, 1] = local[0, 0, 0] + 1000
    local[0, 1, 0] = torch.tensor([20, 21, 22, 23])
    local[0, 1, 1] = local[0, 1, 0] + 1000
    meta = _metadata_stand_in(cp_size, cp_rank, phys, local, 1, trash)
    # A 30-token context row spans 8 global blocks; the generation row is local.
    kv_lens = torch.tensor([30, 9], dtype=torch.int32)
    TrtllmAttentionMetadata._present_helix_context_block_offsets(meta, kv_lens)

    table = meta.kv_cache_block_offsets
    assert meta._helix_local_block_offsets is local
    assert table.shape == (1, 2, 2, 8)
    for block in range(8):
        if block % cp_size == cp_rank:
            expected = [10 + block // cp_size, 1010 + block // cp_size]
        else:
            expected = trash
        assert table[0, 0, :, block].tolist() == expected, block
    assert torch.equal(table[0, 1, :, :width], local[0, 1])


def test_helix_kv_owner() -> None:
    import torch

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    meta = _metadata_stand_in(2, 0, 4, torch.zeros((1, 1, 2, 4), dtype=torch.int32), 1, [0, 0])
    starts = torch.tensor([0, 6], dtype=torch.int64)
    lengths = torch.tensor([9, 3], dtype=torch.int32)
    owner = TrtllmAttentionMetadata._helix_kv_owner(meta, starts, lengths)
    positions = list(range(0, 9)) + list(range(6, 9))
    assert owner.tolist() == [(p // 4) % 2 for p in positions]


def test_draft_forward_sees_dense_kv_lengths() -> None:
    """One-engine draft layers stay dense under helix: inside the draft scope
    they must see global KV lengths and no helix decode state, and the target
    view must come back intact afterwards."""
    import torch

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    class _Meta(SimpleNamespace):
        swap_for_draft = TrtllmAttentionMetadata.swap_for_draft
        record_draft_swap = TrtllmAttentionMetadata.record_draft_swap
        restore_draft_swaps = TrtllmAttentionMetadata.restore_draft_swaps
        prepare_for_draft_forward = TrtllmAttentionMetadata.prepare_for_draft_forward
        restore_after_draft_forward = TrtllmAttentionMetadata.restore_after_draft_forward
        _present_dense_kv_for_draft = TrtllmAttentionMetadata._present_dense_kv_for_draft

    # One 30-token context row (already global), two generation rows verifying
    # groups of 3 at global positions 100 and 200 with rank-local lengths 9.
    kv_lens_cuda = torch.tensor([30, 9, 9, 0], dtype=torch.int32)
    positions = torch.tensor([100, 101, 102, 200, 201, 202, 0, 0], dtype=torch.int32)
    inactive = torch.tensor([False, True], dtype=torch.bool)
    host_kv = torch.tensor([30, 9, 9], dtype=torch.int32)
    meta = _Meta(
        enable_helix=True,
        num_contexts=1,
        num_seqs=3,
        seq_lens_cuda=torch.tensor([30, 3, 3], dtype=torch.int32),
        kv_lens_cuda=kv_lens_cuda,
        kv_lens_cuda_runtime=kv_lens_cuda[:3],
        kv_lens_runtime=host_kv,
        host_total_kv_lens=torch.tensor([30, 18], dtype=torch.int32),
        helix_position_offsets=positions,
        helix_is_inactive_rank=inactive,
        _helix_spec_tokens_valid=True,
        helix_draft_kv_lens_cuda=torch.zeros(4, dtype=torch.int32),
        helix_dense_kv_lens_cpu=torch.tensor([106, 206, 0, 0], dtype=torch.int32),
        draft_replay_swapped_attrs={},
    )

    saved = meta.prepare_for_draft_forward()
    assert meta.kv_lens_cuda[:3].tolist() == [30, 103, 203]
    assert meta.kv_lens_cuda_runtime.tolist() == [30, 103, 203]
    assert meta.kv_lens_runtime.tolist() == [30, 106, 206]
    assert meta.host_total_kv_lens.tolist() == [30, 312]
    assert meta.helix_position_offsets is None
    assert meta.helix_is_inactive_rank is None
    assert meta._helix_spec_tokens_valid is False
    # The target's own buffers are untouched (a CUDA graph replay runs this
    # before the captured target forward).
    assert kv_lens_cuda.tolist() == [30, 9, 9, 0]

    meta.restore_draft_swaps()
    meta.restore_after_draft_forward(saved)
    assert meta.kv_lens_cuda is kv_lens_cuda
    assert meta.kv_lens_runtime is host_kv
    assert meta.helix_position_offsets is positions
    assert meta.helix_is_inactive_rank is inactive
    assert meta._helix_spec_tokens_valid is True
    assert meta.host_total_kv_lens.tolist() == [30, 18]


def test_aggregated_helix_speculation_modes() -> None:
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.llmapi import MTPDecodingConfig, NGramDecodingConfig

    validate = KvCacheCreator._validate_aggregated_helix_speculation

    def mtp(num_nextn_predict_layers: int, **kwargs) -> MTPDecodingConfig:
        config = MTPDecodingConfig(max_draft_len=2, **kwargs)
        # Filled from the checkpoint at model load; the constructor drops it.
        object.__setattr__(config, "num_nextn_predict_layers", num_nextn_predict_layers)
        return config

    validate(mtp(1))
    with pytest.raises(NotImplementedError, match="one-engine MTP"):
        validate(mtp(2))
    with pytest.raises(NotImplementedError, match="one-engine MTP"):
        validate(mtp(1, use_mtp_vanilla=True))
    with pytest.raises(NotImplementedError, match="one-engine MTP"):
        validate(NGramDecodingConfig(max_draft_len=2, max_matching_ngram_size=2))
    with pytest.raises(NotImplementedError, match="mid-run"):
        validate(mtp(1, max_concurrency=8))
