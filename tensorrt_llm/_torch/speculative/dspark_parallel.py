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
"""Experimental parallel drafting for the embedded DeepSeek-V4 DSpark drafter.

Sequential one-model DSpark runs ``target verify -> draft`` back to back, so the
draft sits on the critical path of every step. Parallel drafting runs the draft
for step ``t + 1`` *while* the target verifies step ``t``, assuming every token
in flight is accepted, and throws the draft away when that assumption fails.

The drafter conditions on target hidden states, and the ones for the tokens in
flight do not exist yet. The scheme therefore is:

* Each step the target consumes ``W = max_draft_len + 1`` tokens per request.
  The parallel draft anchors on one of them -- the last in-flight token in the
  *hit* state, the committed bonus (row 0) in the *recovery* state -- and
  drafts a full block of ``K = block_size = W`` tokens after it. The first of
  those tokens doubles as the guess for the target's next token.
* A request *hits* when the target accepted every in-flight token up to and
  including the anchor and its next token equals the first draft token. The
  remaining ``K - 1`` draft tokens are then the next step's drafts, and the
  next draft is anchored on the last of them.
* Otherwise the request *misses*: the accepted prefix and bonus are committed
  as usual, the next step verifies filler drafts, and the next parallel draft
  anchors on that step's committed bonus, where the drafter's context is
  complete. This is the pipelined recovery policy.
* The rolling captured-context window only ever holds verified target hidden
  states (back-filled after verification), so a miss needs no window rollback.
  The in-flight positions the window does not yet cover are either masked out
  (``mask``) or filled with the block K/V of the previous parallel draft, which
  sits at exactly those positions (``fill``).

Knobs are environment variables because this is an experiment, not a stable
API; they are read once at construction.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch

ENV_ENABLE = "TRTLLM_DSPARK_PARALLEL_DRAFT"
ENV_OVERLAP = "TRTLLM_DSPARK_PARALLEL_DRAFT_OVERLAP"
ENV_REPLICATE = "TRTLLM_DSPARK_PARALLEL_DRAFT_REPLICATE"
ENV_INFLIGHT_CTX = "TRTLLM_DSPARK_PARALLEL_DRAFT_INFLIGHT_CTX"
ENV_SYNC_TOKENS = "TRTLLM_DSPARK_PARALLEL_DRAFT_SYNC_TOKENS"
ENV_STREAM_PRIORITY = "TRTLLM_DSPARK_PARALLEL_DRAFT_STREAM_PRIORITY"
ENV_STATS_INTERVAL = "TRTLLM_DSPARK_PARALLEL_DRAFT_STATS_INTERVAL"

InflightContext = Literal["mask", "fill"]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={value!r} is not a boolean (use 0 or 1)")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f"{name}={value!r} is not an integer") from err


def parallel_draft_requested() -> bool:
    """Whether parallel DSpark drafting is switched on in this process."""
    return _env_flag(ENV_ENABLE, False)


@dataclass(frozen=True)
class DSparkParallelDraftConfig:
    """Resolved parallel-drafting settings.

    Attributes:
        enabled: Run the draft for the next step alongside target verification.
        overlap: Launch the draft on a side CUDA stream. When False the draft
            runs on the main stream before the target ("emulated" mode): same
            acceptance behavior, no concurrency. Useful to measure the
            acceptance cost of parallel drafting without any comm hazard.
        replicate: Build the drafter with a single-rank mapping on every rank
            (all draft experts and a full-vocab head per rank) so the draft has
            no collectives and can overlap with the target's collectives.
        inflight_ctx: How the draft attends to in-flight positions the window
            does not cover yet: ``mask`` drops them, ``fill`` uses the previous
            parallel draft's block K/V at those positions.
        sync_tokens: With a replicated drafter under attention TP, copy TP
            rank 0's draft tokens to every rank after the draft. Replicated
            drafts are computed independently per rank, and MoE kernels need
            not be bitwise deterministic, so ranks could otherwise disagree on
            the tokens fed to the next target step.
        stream_priority: CUDA priority of the side stream (lower = higher).
        stats_interval: Log the parallel-draft hit rate every N steps (0: off).
            Each log reads device counters, which synchronizes the host.
    """

    enabled: bool = False
    overlap: bool = True
    replicate: bool = False
    inflight_ctx: InflightContext = "mask"
    sync_tokens: bool = True
    stream_priority: int = 0
    stats_interval: int = 0

    @property
    def fill_inflight(self) -> bool:
        return self.inflight_ctx == "fill"

    @classmethod
    def from_env(cls, world_size: int) -> "DSparkParallelDraftConfig":
        """Resolve the settings for an engine of ``world_size`` ranks."""
        if not parallel_draft_requested():
            return cls(enabled=False)
        overlap = _env_flag(ENV_OVERLAP, True)
        multi_rank = int(world_size) > 1
        replicate = _env_flag(ENV_REPLICATE, overlap and multi_rank)
        if overlap and multi_rank and not replicate:
            raise ValueError(
                f"{ENV_OVERLAP}=1 on {world_size} ranks requires a replicated drafter "
                f"({ENV_REPLICATE}=1): a sharded drafter's collectives would run "
                "concurrently with the target's on shared communicators and "
                f"workspaces. Set {ENV_OVERLAP}=0 to run the draft serially instead."
            )
        inflight_ctx = os.environ.get(ENV_INFLIGHT_CTX, "mask").strip().lower() or "mask"
        if inflight_ctx not in ("mask", "fill"):
            raise ValueError(f"{ENV_INFLIGHT_CTX}={inflight_ctx!r} must be 'mask' or 'fill'")
        stats_interval = _env_int(ENV_STATS_INTERVAL, 0)
        if stats_interval < 0:
            raise ValueError(f"{ENV_STATS_INTERVAL} must be >= 0")
        return cls(
            enabled=True,
            overlap=overlap,
            replicate=replicate,
            inflight_ctx=inflight_ctx,
            sync_tokens=_env_flag(ENV_SYNC_TOKENS, True),
            stream_priority=_env_int(ENV_STREAM_PRIORITY, 0),
            stats_interval=stats_interval,
        )


def anchor_rows(hit: torch.Tensor, width: int) -> torch.Tensor:
    """Row of each request's in-flight window the parallel draft anchors on.

    ``hit`` requests anchor on the last in-flight token (row ``width - 1``);
    recovering requests anchor on the committed bonus (row 0).
    """
    return torch.where(
        hit, torch.full_like(hit, width - 1, dtype=torch.long), torch.zeros_like(hit, dtype=torch.long)
    )


def context_extent(
    ctx_end: torch.Tensor, valid_len: torch.Tensor, anchor_pos: torch.Tensor, window: int
) -> torch.Tensor:
    """Number of verified window frames, ending at ``ctx_end``, the draft attends.

    The sequential draft attends the ``window`` frames ending at the anchor's
    position. The parallel draft has verified frames only up to ``ctx_end``;
    the ``anchor_pos - ctx_end`` frames after it are in flight. Keeping the
    same sliding-window extent leaves ``window - gap`` verified frames, further
    bounded by how many frames were ever written.
    """
    gap = (anchor_pos - ctx_end).clamp(min=0)
    return torch.minimum(valid_len.long(), window - gap).clamp(min=0, max=window)


def inflight_fill_valid(
    prev_frame0: torch.Tensor, block: int, ctx_end: torch.Tensor, anchor_pos: torch.Tensor
) -> torch.Tensor:
    """Which stashed previous-block K/V entries stand in for in-flight frames.

    Entry ``e`` of the previous parallel draft's block sits at frame
    ``prev_frame0 + e``. It is used when that frame is in flight, i.e. after
    the last verified frame ``ctx_end`` and no later than the anchor's frame.
    ``prev_frame0 == 0`` marks an empty stash.
    """
    frames = prev_frame0.unsqueeze(1) + torch.arange(block, device=prev_frame0.device)
    return (
        (prev_frame0.unsqueeze(1) > 0)
        & (frames > ctx_end.unsqueeze(1))
        & (frames <= anchor_pos.unsqueeze(1))
    )


def resolve_hits(
    num_accepted: torch.Tensor,
    anchors: torch.Tensor,
    bonus: torch.Tensor,
    first_draft: torch.Tensor,
) -> torch.Tensor:
    """Whether each request's parallel draft is valid for the next step.

    Valid iff verification committed exactly the in-flight tokens through the
    anchor (``num_accepted - 1 == anchor``) and the target's next token (the
    bonus) equals the draft's first token. Accepting past the anchor -- filler
    drafts in a recovery step that happened to match -- is a miss: the draft
    then describes positions that are already committed.
    """
    return (num_accepted.long() - 1 == anchors) & (bonus.long() == first_draft.long())


def backfill_plan(
    num_accepted: torch.Tensor, width: int, base: torch.Tensor, ctx_end: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Captured rows and window frames of each request's accepted in-flight tokens.

    Row ``base + j`` holds the target hidden state of the in-flight token at
    position ``ctx_end + j``, which the window stores at frame ``ctx_end + 1 + j``
    (frame = position + 1). Rows ``j < num_accepted`` were accepted.

    Returns:
        ``(rows, frames, mask)``, each ``[G, width]``.
    """
    j = torch.arange(width, device=base.device)
    rows = base.unsqueeze(1) + j
    frames = ctx_end.unsqueeze(1) + 1 + j
    mask = j.unsqueeze(0) < num_accepted.long().unsqueeze(1)
    return rows, frames, mask


def broadcast_rank0_rows(
    rows: torch.Tensor, mapping, gather_fn: Optional[callable] = None
) -> torch.Tensor:
    """Return TP rank 0's copy of ``rows`` on every rank of the TP group."""
    if gather_fn is None:
        from ..distributed import allgather as gather_fn
    gathered = gather_fn(rows.contiguous(), mapping, dim=0)
    return gathered[: rows.shape[0]]
