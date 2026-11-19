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
"""Worker side of confidence-scheduled verification for step-by-step drafters.

The executor chooses how many draft tokens each request verifies from a
per-request, per-position acceptance confidence. It reaches the worker through
a small interface (``verify_planner``, ``staged_confidence_buffer``,
``bump_draft_seq``, ...), which :class:`ConfidenceScheduledWorkerMixin` serves
from one :class:`ConfidenceScheduleState`. The confidence comes from an
:class:`~.acceptance_estimator.OnlineAcceptanceEstimator` fitted on the draft
logits, so the drafter needs no trained confidence head.
"""

from collections import deque
from typing import Optional, Sequence

import torch

from ..._utils import prefer_pinned
from .acceptance_estimator import OnlineAcceptanceEstimator
from .dspark_schedule import NEUTRAL_CONFIDENCE_LOGIT, DSparkScheduleConfig
from .dspark_verify import DSparkVerifyPlanner

__all__ = ["ConfidenceScheduleState", "ConfidenceScheduledWorkerMixin"]


class ConfidenceScheduleState:
    """Slot-indexed confidence of every request's latest draft block.

    Rows ``[0, num_slots)`` belong to live requests. Row ``num_slots`` is a
    scratch row shared by padding and dummy requests, whose content is never
    read as a confidence. Row ``num_slots + 1`` stays neutral ("verify every
    draft") and stands in for a request without a slot.

    Args:
        num_slots: request slots of the executor's sequence-slot pool.
        max_draft_len: draft positions per request.
        device: device of the draft logits.
        device_windows: whether the planner serves device re-ranking
            (``enable_fused_confidence_scheduler``).
        is_dummy_request_id: host predicate for request ids that hold no real
            request (attention-DP idle rows, CUDA graph padding).
    """

    def __init__(
        self,
        *,
        num_slots: int,
        max_draft_len: int,
        device: torch.device | str,
        device_windows: bool,
        is_dummy_request_id,
    ) -> None:
        self.num_slots = int(num_slots)
        self.max_draft_len = int(max_draft_len)
        self.scratch_row = self.num_slots
        self.neutral_row = self.num_slots + 1
        self._is_dummy_request_id = is_dummy_request_id
        num_rows = self.num_slots + 2

        self.estimator = OnlineAcceptanceEstimator(num_rows, self.max_draft_len, device)
        self.verify_planner = DSparkVerifyPlanner(
            cfg=DSparkScheduleConfig(block_size=self.max_draft_len, min_verify_len=1),
            device_windows=device_windows,
        )
        # The executor updates these between forwards, outside inference mode,
        # which an inference tensor would reject.
        with torch.inference_mode(False):
            self.logits = torch.full(
                (num_rows, self.max_draft_len),
                NEUTRAL_CONFIDENCE_LOGIT,
                dtype=torch.float32,
                device=device,
            )
            # Draft pass that wrote each row of ``logits``.
            self.stamp = torch.zeros(num_rows, dtype=torch.int32, device=device)
            self.draft_seq = torch.zeros((), dtype=torch.int32, device=device)
            self.batch_to_slot = torch.full(
                (self.num_slots,), self.scratch_row, dtype=torch.long, device=device
            )
        # Windows the forward executed, in output-row order.
        self.output_verify_lens = torch.empty(self.num_slots, dtype=torch.int32, device=device)
        self._draft_seq_host = 0
        self._req_to_slot: dict[int, int] = {}
        self._free_slots = deque(range(self.num_slots))
        self._host_batch_to_slot: list[int] = []

    # Host side, between forwards.

    def assign_batch(self, request_ids: Sequence[int], paused_ids=frozenset()) -> None:
        """Map this step's requests to their slots and publish the map on device.

        A real request keeps one slot from its first step to its last. Slots of
        requests that left the batch are freed, except those paused in place.
        """
        req_to_slot = self._req_to_slot
        for request_id in req_to_slot.keys() - set(request_ids) - paused_ids:
            self._free_slots.append(req_to_slot.pop(request_id))
        slots = []
        for request_id in request_ids:
            slot = req_to_slot.get(request_id)
            if slot is None:
                if self._is_dummy_request_id(request_id):
                    slot = self.scratch_row
                else:
                    if not self._free_slots:
                        raise RuntimeError(
                            f"No free confidence slot for request {request_id}: all "
                            f"{self.num_slots} slots of the sequence-slot pool are in use"
                        )
                    slot = self._free_slots.popleft()
                    req_to_slot[request_id] = slot
                    # A recycled slot holds the previous occupant's block.
                    self.logits[slot].fill_(NEUTRAL_CONFIDENCE_LOGIT)
                    self.estimator.clear_row(slot)
            slots.append(slot)
        # Steady decode repeats the map already on the device.
        if slots != self._host_batch_to_slot:
            self.batch_to_slot[: len(slots)].copy_(
                torch.tensor(slots, dtype=torch.long, pin_memory=prefer_pinned()),
                non_blocking=True,
            )
            self._host_batch_to_slot = slots

    def bump_draft_seq(self) -> int:
        """Advance the draft-pass sequence and return the previous value."""
        previous = self._draft_seq_host
        self._draft_seq_host = (previous + 1) & ((1 << 31) - 1)
        if self._draft_seq_host == 0:
            # Recycled stamps must not authenticate an old score.
            self.stamp.zero_()
            self.logits.fill_(NEUTRAL_CONFIDENCE_LOGIT)
        self.draft_seq.fill_(self._draft_seq_host)
        return previous

    def confidence_row_for(self, request_id: int) -> int:
        """Row of ``logits`` holding the request's confidence, neutral if it has none."""
        return self._req_to_slot.get(request_id, self.neutral_row)

    # Inside the forward: fixed shapes and no host synchronization.

    def batch_rows(self, num_rows: int) -> torch.Tensor:
        """``[num_rows]`` slot of each batch row, in batch order."""
        return self.batch_to_slot[:num_rows]

    def observe(
        self,
        rows: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_verified: torch.Tensor | int,
    ) -> None:
        """Grade the block each generation request just verified.

        Args:
            rows: ``[G]`` slots of the generation requests.
            num_accepted_tokens: ``[G]`` tokens each request emits, bonus included.
            num_verified: draft tokens the target scored, ``[G]`` or one count.
        """
        self.estimator.observe(rows, num_accepted_tokens - 1, num_verified, rows != self.scratch_row)

    def score_step(self, logits: torch.Tensor, rows: torch.Tensor, position: int) -> None:
        """Score draft ``position`` of every request and stage its confidence.

        ``logits`` are the ``[N, vocab]`` full-vocabulary logits the greedy
        draft tokens of this step are taken from.
        """
        self.logits[rows, position] = self.estimator.predict_step(logits, rows, position)
        if position == 0:
            self.stamp[rows] = self.draft_seq

    def discard(self, rows: torch.Tensor) -> None:
        """Drop the ungraded blocks of ``rows``, which will not be verified."""
        self.estimator.discard(rows)

    def snapshot(self) -> tuple:
        """Copies of the state a forward changes, for a warmup pass to put back."""
        return self.logits.clone(), self.stamp.clone(), self.estimator.snapshot()

    def restore(self, saved: tuple) -> None:
        logits, stamp, estimator = saved
        self.logits.copy_(logits)
        self.stamp.copy_(stamp)
        self.estimator.restore(estimator)


class ConfidenceScheduledWorkerMixin:
    """The interface the executor and the engine read from a speculative worker.

    A worker that supports confidence scheduling builds ``_confidence`` in its
    first forward. Until then, and for a worker that leaves it ``None``, the
    executor sees no planner and keeps native verification.
    """

    _confidence: Optional[ConfidenceScheduleState] = None

    @property
    def verify_planner(self) -> Optional[DSparkVerifyPlanner]:
        return None if self._confidence is None else self._confidence.verify_planner

    def staged_confidence_buffer(self) -> Optional[torch.Tensor]:
        """The whole slot-indexed confidence buffer, read back by slot one step later."""
        return None if self._confidence is None else self._confidence.logits

    def confidence_stamp_buffer(self) -> Optional[torch.Tensor]:
        return None if self._confidence is None else self._confidence.stamp

    def bump_draft_seq(self) -> int:
        """Once per step from the executor's host path, before the forward is enqueued."""
        return self._confidence.bump_draft_seq()

    def advance_acceptance_estimator(self) -> None:
        """Same call contract as :meth:`bump_draft_seq`."""
        self._confidence.estimator.advance()

    def confidence_row_for(self, req_id: int) -> int:
        return self._confidence.confidence_row_for(req_id)

    def verified_draft_seq_cuda(self) -> Optional[torch.Tensor]:
        """Sequence of the pass that drafted the block being verified.

        ``draft_seq`` already holds the current step's sequence by launch time.
        """
        return None if self._confidence is None else self._confidence.draft_seq - 1

    def batch_slot_view(self, num_rows: int) -> Optional[torch.Tensor]:
        return None if self._confidence is None else self._confidence.batch_rows(num_rows)
