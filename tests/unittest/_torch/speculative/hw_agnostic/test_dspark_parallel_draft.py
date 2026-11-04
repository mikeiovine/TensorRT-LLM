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
"""CPU tests for experimental parallel DSpark drafting (dspark_parallel.py)."""

import pytest
import torch

import tensorrt_llm._torch.models.modeling_dspark as modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    dspark_attention_forward_batched,
    dspark_attention_forward_parallel,
    get_dspark_topk_idxs_batched,
    get_dspark_topk_idxs_parallel,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._torch.speculative import dspark_parallel as dp
from tensorrt_llm._torch.speculative.dspark_parallel import (
    DSparkParallelDraftConfig,
    anchor_rows,
    backfill_plan,
    broadcast_rank0_rows,
    context_extent,
    inflight_fill_valid,
    resolve_hits,
)

_ENV_VARS = (
    dp.ENV_ENABLE,
    dp.ENV_OVERLAP,
    dp.ENV_REPLICATE,
    dp.ENV_INFLIGHT_CTX,
    dp.ENV_SYNC_TOKENS,
    dp.ENV_STREAM_PRIORITY,
    dp.ENV_STATS_INTERVAL,
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def test_config_disabled_by_default(clean_env):
    assert DSparkParallelDraftConfig.from_env(world_size=4) == DSparkParallelDraftConfig()
    assert not DSparkParallelDraftConfig.from_env(world_size=1).enabled


def test_config_single_rank_overlaps_without_replication(clean_env):
    clean_env.setenv(dp.ENV_ENABLE, "1")
    cfg = DSparkParallelDraftConfig.from_env(world_size=1)
    assert cfg.enabled and cfg.overlap and not cfg.replicate
    assert cfg.inflight_ctx == "mask" and not cfg.fill_inflight


def test_config_multi_rank_overlap_replicates_by_default(clean_env):
    clean_env.setenv(dp.ENV_ENABLE, "1")
    cfg = DSparkParallelDraftConfig.from_env(world_size=4)
    assert cfg.overlap and cfg.replicate and cfg.sync_tokens


def test_config_rejects_sharded_multi_rank_overlap(clean_env):
    clean_env.setenv(dp.ENV_ENABLE, "1")
    clean_env.setenv(dp.ENV_REPLICATE, "0")
    with pytest.raises(ValueError, match="replicated drafter"):
        DSparkParallelDraftConfig.from_env(world_size=4)


def test_config_emulated_mode_keeps_sharded_drafter(clean_env):
    clean_env.setenv(dp.ENV_ENABLE, "1")
    clean_env.setenv(dp.ENV_OVERLAP, "0")
    cfg = DSparkParallelDraftConfig.from_env(world_size=8)
    assert not cfg.overlap and not cfg.replicate


def test_config_parses_fill_priority_and_stats(clean_env):
    clean_env.setenv(dp.ENV_ENABLE, "true")
    clean_env.setenv(dp.ENV_INFLIGHT_CTX, "FILL")
    clean_env.setenv(dp.ENV_STREAM_PRIORITY, "-1")
    clean_env.setenv(dp.ENV_STATS_INTERVAL, "50")
    cfg = DSparkParallelDraftConfig.from_env(world_size=1)
    assert cfg.fill_inflight and cfg.stream_priority == -1 and cfg.stats_interval == 50


@pytest.mark.parametrize(
    "name, value",
    [(dp.ENV_INFLIGHT_CTX, "zeros"), (dp.ENV_OVERLAP, "maybe"), (dp.ENV_STATS_INTERVAL, "-2")],
)
def test_config_rejects_bad_values(clean_env, name, value):
    clean_env.setenv(dp.ENV_ENABLE, "1")
    clean_env.setenv(name, value)
    with pytest.raises(ValueError):
        DSparkParallelDraftConfig.from_env(world_size=1)


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------


def test_anchor_rows():
    hit = torch.tensor([True, False, True])
    assert anchor_rows(hit, 5).tolist() == [4, 0, 4]
    assert anchor_rows(hit, 1).tolist() == [0, 0, 0]


def test_context_extent_keeps_sliding_window_relative_to_anchor():
    window = 8
    ctx_end = torch.tensor([20, 20, 20, 3])
    valid = torch.tensor([8, 8, 2, 3])
    anchor = torch.tensor([20, 24, 24, 3])
    # No gap: the whole window. Gap 4: 4 verified frames still in range.
    # Fewer written frames than that: all of them.
    assert context_extent(ctx_end, valid, anchor, window).tolist() == [8, 4, 2, 3]


def test_context_extent_is_zero_for_a_fresh_bootstrapped_slot():
    assert context_extent(torch.tensor([100]), torch.tensor([0]), torch.tensor([100]), 8).item() == 0


def test_inflight_fill_valid_selects_frames_after_ctx_end_through_anchor():
    # Previous block slot 0 at frame 10 (stash entries at frames 10..14);
    # verified through frame 10, anchor at position 14 -> frames 11..14 in flight.
    valid = inflight_fill_valid(
        torch.tensor([10, 0, 10]), 5, torch.tensor([10, 10, 10]), torch.tensor([14, 14, 10])
    )
    assert valid.tolist() == [
        [False, True, True, True, True],
        [False] * 5,  # empty stash
        [False] * 5,  # recovery: nothing in flight
    ]


def test_resolve_hits():
    num_accepted = torch.tensor([5, 5, 3, 1, 2])
    anchors = torch.tensor([4, 4, 4, 0, 0])
    bonus = torch.tensor([7, 7, 7, 9, 9])
    first = torch.tensor([7, 8, 7, 9, 9])
    # all accepted + guess right; guess wrong; partial accept; recovery hit;
    # filler accepted past the anchor.
    assert resolve_hits(num_accepted, anchors, bonus, first).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_backfill_plan():
    rows, frames, mask = backfill_plan(
        torch.tensor([2, 4]), 4, torch.tensor([3, 7]), torch.tensor([50, 60])
    )
    assert rows.tolist() == [[3, 4, 5, 6], [7, 8, 9, 10]]
    assert frames.tolist() == [[51, 52, 53, 54], [61, 62, 63, 64]]
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]


def test_broadcast_rank0_rows_takes_first_rank_slice():
    local = torch.tensor([[1, 2], [3, 4]])
    other = torch.tensor([[5, 6], [7, 8]])

    def fake_allgather(t, mapping, dim):
        assert dim == 0 and mapping == "tp"
        return torch.cat([local, other], dim=0)

    out = broadcast_rank0_rows(other, "tp", gather_fn=fake_allgather)
    assert out.tolist() == local.tolist()


# ---------------------------------------------------------------------------
# Token-level simulation of the hit / miss / recovery state machine
# ---------------------------------------------------------------------------


def _simulate(seq, width, num_steps, corrupt_steps=(), corrupt_first_only=False):
    """Drive the helpers with an oracle drafter and a strict-match target.

    The in-flight window is ``[committed bonus, drafts...]`` starting at
    position ``p0``. The oracle drafts ``seq[a + 1 : a + 1 + width]`` after an
    anchor at position ``a``. Steps in ``corrupt_steps`` get a wrong block (a
    real draft conditioned on a wrong first token is wrong throughout), or
    only a wrong first token with ``corrupt_first_only``. Returns per-step
    (committed tokens, hit) and the committed text.
    """
    block = width
    p0 = 0
    inflight_drafts = [-1] * (width - 1)  # filler after prefill
    hit = torch.tensor([False])
    committed = [seq[0]]
    log = []
    for step in range(num_steps):
        anchors = anchor_rows(hit, width)
        a = p0 + int(anchors.item())
        pending = list(seq[a + 1 : a + 1 + block])
        if step in corrupt_steps:
            pending = [-7] + pending[1:] if corrupt_first_only else [-7] * block
        # Strict verification of the in-flight drafts.
        n = 1
        for k, tok in enumerate(inflight_drafts):
            if tok != seq[p0 + 1 + k]:
                break
            n += 1
        bonus = seq[p0 + n]
        committed.extend(seq[p0 + 1 : p0 + n + 1])
        hit = resolve_hits(
            torch.tensor([n]), anchors, torch.tensor([bonus]), torch.tensor([pending[0]])
        )
        log.append((n, bool(hit.item())))
        inflight_drafts = pending[1:width]
        p0 += n
    return log, committed


@pytest.mark.parametrize("width", [2, 4, 6])
def test_oracle_drafter_commits_full_windows_after_first_recovery(width):
    seq = list(range(100, 200))
    log, committed = _simulate(seq, width, num_steps=8)
    # Step 0 verifies prefill filler: one token, and the recovery draft hits.
    assert log[0] == (1, True)
    assert all(entry == (width, True) for entry in log[1:])
    assert committed == seq[: len(committed)]


def test_wrong_guess_costs_one_recovery_step():
    seq = list(range(100, 200))
    width = 4
    log, committed = _simulate(seq, width, num_steps=6, corrupt_steps={2})
    # Step 2's draft has the wrong first token: step 2 still commits a full
    # window but misses, step 3 verifies filler (1 token) and recovers.
    assert log[2] == (width, False)
    assert log[3] == (1, True)
    assert log[4] == (width, True)
    assert committed == seq[: len(committed)]


def test_filler_accepted_past_the_anchor_is_a_miss():
    seq = list(range(100, 200))
    width = 4
    log, committed = _simulate(seq, width, num_steps=6, corrupt_steps={2}, corrupt_first_only=True)
    # Only the first token was wrong, so step 3's filler drafts are right and
    # verification accepts past the recovery anchor. That draft describes
    # already-committed positions, so step 3 misses and step 4 recovers.
    assert log[2] == (width, False)
    assert log[3] == (width, False)
    assert log[4] == (1, True)
    assert committed == seq[: len(committed)]


# ---------------------------------------------------------------------------
# Attention: parallel path vs the sequential batched path
# ---------------------------------------------------------------------------


def _attn_setup(seed=0):
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 4, 8, 4
    q_lora, o_lora, n_groups = 6, 5, 2
    window, block, rows = 8, 3, 3
    bf = torch.bfloat16
    weights = dict(
        wq_a=torch.randn(q_lora, dim, dtype=bf) * 0.1,
        q_norm_w=torch.ones(q_lora),
        wq_b=torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1,
        wkv=torch.randn(head_dim, dim, dtype=bf) * 0.1,
        kv_norm_w=torch.ones(head_dim),
        wo_a=torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1,
        wo_b=torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1,
        attn_sink=torch.randn(n_heads),
    )
    params = dict(
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rd,
        n_groups=n_groups,
        o_lora_rank=o_lora,
        window_size=window,
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
        freqs_cis=precompute_dspark_freqs_cis(rd, 32),
    )
    x = torch.randn(2, block, dim, dtype=bf)
    main_x = torch.randn(2, 1, dim, dtype=bf)
    kv_cache = torch.randn(rows, window, head_dim, dtype=bf)
    slots = torch.tensor([2, 0])
    return x, main_x, kv_cache, slots, weights, params


def test_parallel_topk_matches_batched_without_stand_ins():
    ctx_end = torch.tensor([3, 11, 20])
    valid = torch.tensor([3, 8, 5])
    torch.testing.assert_close(
        get_dspark_topk_idxs_parallel(8, 3, ctx_end, valid),
        get_dspark_topk_idxs_batched(8, 3, ctx_end, valid),
        rtol=0,
        atol=0,
    )


def test_parallel_topk_places_stand_ins_between_window_and_block():
    idx = get_dspark_topk_idxs_parallel(
        4,
        2,
        torch.tensor([5]),
        torch.tensor([2]),
        extra_valid=torch.tensor([[False, True, True]]),
    )
    assert idx.shape == (1, 2, 4 + 3 + 2)
    # Frames 5 and 4 live in window columns 1 and 0.
    assert idx[0, 0].tolist() == [0, 1, -1, -1, -1, 5, 6, 7, 8]


@pytest.mark.parametrize("seed", [0, 1])
def test_parallel_attention_matches_sequential_once_main_kv_is_in_window(monkeypatch, seed):
    """Recovery-state parallel draft == sequential draft (same verified context)."""
    monkeypatch.setattr(modeling_dspark, "IS_CUTLASS_DSL_AVAILABLE", False)
    x, main_x, kv_cache, slots, weights, params = _attn_setup(seed)
    start_pos = torch.tensor([6, 13])
    valid = torch.tensor([5, 8])

    written = kv_cache.clone()
    expected = dspark_attention_forward_batched(
        x, main_x, start_pos, written, slots, valid, persist=True, **weights, **params
    )
    # The sequential path wrote main_kv at frame start_pos; the parallel path
    # reads the same frame from the window as verified context.
    before = written.clone()
    actual, block_kv = dspark_attention_forward_parallel(
        x, start_pos, start_pos, valid, written, slots, **weights, **params
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(written, before, rtol=0, atol=0)  # read only
    assert block_kv.shape == (2, x.shape[1], params["head_dim"])


def test_parallel_attention_ignores_masked_stand_ins_and_uses_valid_ones(monkeypatch):
    monkeypatch.setattr(modeling_dspark, "IS_CUTLASS_DSL_AVAILABLE", False)
    x, _, kv_cache, slots, weights, params = _attn_setup(3)
    start_pos = torch.tensor([10, 12])
    ctx_end = torch.tensor([7, 9])
    num_ctx = torch.tensor([5, 5])
    extra_kv = torch.randn(2, 3, params["head_dim"], dtype=torch.bfloat16)

    plain, _ = dspark_attention_forward_parallel(
        x, start_pos, ctx_end, num_ctx, kv_cache, slots, **weights, **params
    )
    masked, _ = dspark_attention_forward_parallel(
        x,
        start_pos,
        ctx_end,
        num_ctx,
        kv_cache,
        slots,
        extra_kv,
        torch.zeros(2, 3, dtype=torch.bool),
        **weights,
        **params,
    )
    filled, _ = dspark_attention_forward_parallel(
        x,
        start_pos,
        ctx_end,
        num_ctx,
        kv_cache,
        slots,
        extra_kv,
        torch.ones(2, 3, dtype=torch.bool),
        **weights,
        **params,
    )
    torch.testing.assert_close(masked, plain, rtol=0, atol=0)
    assert not torch.equal(filled, plain)


def test_parallel_attention_requires_paired_stand_in_arguments():
    x, _, kv_cache, slots, weights, params = _attn_setup()
    with pytest.raises(ValueError, match="passed together"):
        dspark_attention_forward_parallel(
            x,
            torch.tensor([4, 4]),
            torch.tensor([4, 4]),
            torch.tensor([4, 4]),
            kv_cache,
            slots,
            torch.zeros(2, 1, params["head_dim"], dtype=torch.bfloat16),
            None,
            **weights,
            **params,
        )
