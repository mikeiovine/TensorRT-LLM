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
"""Worker-side confidence scheduling state of step-by-step drafters."""

import pytest
import torch

from tensorrt_llm._torch.speculative.confidence_schedule import (
    ConfidenceScheduledWorkerMixin,
    ConfidenceScheduleState,
)

pytestmark = pytest.mark.cpu_only

SLOTS = 4
K = 3
VOCAB = 32
DUMMY_ID = 1000
NEUTRAL = [30.0] * K


def _state(device_windows: bool = False) -> ConfidenceScheduleState:
    return ConfidenceScheduleState(
        num_slots=SLOTS,
        max_draft_len=K,
        device="cpu",
        device_windows=device_windows,
        is_dummy_request_id=lambda request_id: request_id == 0 or request_id >= DUMMY_ID,
    )


def _draft(state: ConfidenceScheduleState, num_rows: int) -> torch.Tensor:
    """Score one block of ``num_rows`` batch rows and return its logits ``[N, K, V]``."""
    logits = torch.randn(num_rows, K, VOCAB) * 4.0
    for position in range(K):
        state.score_step(logits[:, position], state.batch_rows(num_rows), position)
    return logits


def test_requests_keep_their_slot_and_dummies_share_the_scratch_row():
    state = _state()

    state.assign_batch([7, 9, DUMMY_ID, 0])
    assert state.batch_rows(4).tolist() == [0, 1, state.scratch_row, state.scratch_row]
    # The batch is reordered and one request leaves.
    state.assign_batch([9, 11])

    assert state.batch_rows(2).tolist() == [1, 2]
    assert state.confidence_row_for(9) == 1
    assert state.confidence_row_for(7) == state.neutral_row
    assert sorted(state._free_slots) == [0, 3]


def test_a_paused_request_keeps_its_slot_while_out_of_the_batch():
    state = _state()
    state.assign_batch([7, 9])

    state.assign_batch([9], paused_ids=frozenset({7}))
    state.assign_batch([7, 9])

    assert state.batch_rows(2).tolist() == [0, 1]


def test_a_full_slot_pool_is_reported():
    state = _state()
    state.assign_batch([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="No free confidence slot for request 5"):
        state.assign_batch([1, 2, 3, 4, 5])


def test_scores_land_in_the_request_slots_with_the_draft_pass_stamp():
    torch.manual_seed(0)
    state = _state()
    state.assign_batch([7, 9, DUMMY_ID])
    assert state.bump_draft_seq() == 0

    logits = _draft(state, 3)

    probs = torch.softmax(logits[:2].double(), dim=-1).amax(dim=-1)
    expected = (0.5 * torch.log(probs / (1.0 - probs)) - 0.2).float()
    torch.testing.assert_close(state.logits[:2], expected, atol=1e-3, rtol=1e-3)
    assert state.stamp.tolist() == [1, 1, 0, 0, 1, 0]
    assert state.logits[state.neutral_row].tolist() == NEUTRAL
    assert state.logits[2:SLOTS].flatten().tolist() == NEUTRAL * 2


def test_a_recycled_slot_starts_neutral_and_is_not_graded_on_the_old_block():
    state = _state()
    state.assign_batch([1, 2, 3, 4])
    _draft(state, SLOTS)

    # Request 1 leaves and request 5 takes its slot, the only free one.
    state.assign_batch([5, 2, 3, 4])
    assert state.batch_rows(SLOTS).tolist() == [0, 1, 2, 3]
    assert state.logits[0].tolist() == NEUTRAL
    assert state.logits[1].tolist() != NEUTRAL
    state.observe(state.batch_rows(1), torch.tensor([3]), K)

    assert not state.estimator.stats.any()


def test_observe_grades_real_rows_and_takes_the_bonus_token_off():
    state = _state()
    state.assign_batch([7, 9, DUMMY_ID])
    _draft(state, 3)

    # Request 7 accepts two drafts of a three-draft window. Request 9 verifies
    # one draft and accepts it. The padding row is never graded.
    state.observe(
        state.batch_rows(3),
        torch.tensor([3, 2, 4], dtype=torch.int32),
        torch.tensor([3, 1, 3], dtype=torch.int32),
    )

    assert state.estimator.stats[:, 5].tolist() == [2.0, 1.0, 1.0]


def test_the_draft_sequence_wrap_drops_every_staged_score():
    state = _state()
    state.assign_batch([7])
    _draft(state, 1)
    state._draft_seq_host = (1 << 31) - 1

    assert state.bump_draft_seq() == (1 << 31) - 1

    assert state.draft_seq.item() == 0
    assert not state.stamp.any()
    assert state.logits[0].tolist() == NEUTRAL


def test_restore_puts_back_the_state_a_warmup_forward_changes():
    state = _state()
    state.assign_batch([7, 9])
    state.bump_draft_seq()
    _draft(state, 2)
    expected = [state.logits.clone(), state.stamp.clone(), state.estimator.snapshot()]
    saved = state.snapshot()

    state.bump_draft_seq()
    state.observe(state.batch_rows(2), torch.tensor([2, 1]), K)
    _draft(state, 2)
    state.restore(saved)

    assert torch.equal(state.logits, expected[0])
    assert torch.equal(state.stamp, expected[1])
    for tensor, value in zip(state.estimator.snapshot(), expected[2]):
        assert torch.equal(tensor, value)


def test_a_worker_without_state_exposes_no_planner():
    worker = ConfidenceScheduledWorkerMixin()

    assert worker.verify_planner is None
    assert worker.staged_confidence_buffer() is None
    assert worker.confidence_stamp_buffer() is None
    assert worker.verified_draft_seq_cuda() is None
    assert worker.batch_slot_view(2) is None


def test_the_worker_interface_reads_the_state():
    worker = ConfidenceScheduledWorkerMixin()
    state = worker._confidence = _state(device_windows=True)
    state.assign_batch([7, 9])
    state.estimator.REFIT_INTERVAL = 1

    assert worker.verify_planner.device_windows
    assert worker.verify_planner.max_verify_len == K
    assert worker.staged_confidence_buffer() is state.logits
    assert worker.confidence_stamp_buffer() is state.stamp
    assert worker.confidence_row_for(9) == 1
    assert worker.batch_slot_view(2).tolist() == [0, 1]
    assert worker.bump_draft_seq() == 0
    assert worker.verified_draft_seq_cuda().item() == 0
    worker.advance_acceptance_estimator()
    assert state.estimator.num_refits == 1
