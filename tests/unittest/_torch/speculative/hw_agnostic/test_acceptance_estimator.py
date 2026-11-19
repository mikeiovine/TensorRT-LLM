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
"""Online acceptance estimator: scoring, grading and the refit."""

import pytest
import torch

from tensorrt_llm._torch.speculative.acceptance_estimator import (
    OnlineAcceptanceEstimator,
    max_token_log_odds,
)

pytestmark = pytest.mark.cpu_only

NUM_ROWS = 16
BLOCK = 4
VOCAB = 512


def _estimator(num_rows: int = NUM_ROWS) -> OnlineAcceptanceEstimator:
    estimator = OnlineAcceptanceEstimator(num_rows, BLOCK, "cpu")
    # Distinct per-position intercepts, so reading the wrong one shows up.
    estimator.slope.fill_(0.7)
    estimator.intercepts.copy_(torch.tensor([1.5, 1.0, 0.5, 0.0]))
    return estimator


def _reference_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """logit(q_max) by another route: mask the argmax and reduce the rest."""
    logits = logits.double()
    top = logits.max(dim=-1)
    rest = logits.scatter(-1, top.indices.unsqueeze(-1), float("-inf"))
    return (top.values - torch.logsumexp(rest, dim=-1)).clamp(-40.0, 40.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_feature_is_the_top_token_log_odds(dtype):
    torch.manual_seed(0)
    logits = (torch.randn(6, BLOCK, VOCAB) * 4.0).to(dtype)

    feature = max_token_log_odds(logits)

    assert feature.dtype == torch.float32
    assert feature.shape == (6, BLOCK)
    torch.testing.assert_close(
        feature.double(), _reference_log_odds(logits.float()), atol=2e-3, rtol=2e-3
    )


def test_feature_stays_finite_for_degenerate_rows():
    one_hot = torch.full((VOCAB,), float("-inf"))
    one_hot[3] = 0.0
    peaked = torch.zeros(VOCAB)
    peaked[5] = 200.0
    uniform = torch.zeros(VOCAB)

    feature = max_token_log_odds(torch.stack([one_hot, peaked, uniform]))

    assert feature[:2].tolist() == [40.0, 40.0]
    # Every token ties for the maximum: the others outweigh any one of them.
    assert feature[2].item() == pytest.approx(-torch.log(torch.tensor(VOCAB - 1.0)).item())


def test_predict_returns_batch_order_and_stores_by_slot():
    torch.manual_seed(0)
    estimator = _estimator()
    rows = torch.tensor([7, 2, 11, 5])
    logits = torch.randn(4, BLOCK, VOCAB) * 4.0

    confidence_logits = estimator.predict(logits, rows)

    feature = _reference_log_odds(logits).float()
    expected = 0.7 * feature + estimator.intercepts
    torch.testing.assert_close(confidence_logits, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(estimator.features[rows], feature, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        estimator.predictions[rows], torch.sigmoid(expected), atol=2e-3, rtol=2e-3
    )
    untouched = torch.ones(NUM_ROWS, dtype=torch.bool)
    untouched[rows] = False
    assert estimator.pending.tolist() == (~untouched).tolist()
    assert not estimator.features[untouched].any()
    assert not estimator.predictions[untouched].any()


def test_predict_step_scores_a_block_one_position_at_a_time():
    torch.manual_seed(0)
    by_block, by_step = _estimator(), _estimator()
    rows = torch.tensor([7, 2, 11, 5])
    logits = torch.randn(4, BLOCK, VOCAB) * 4.0

    expected = by_block.predict(logits, rows)
    stepped = []
    for position in range(BLOCK):
        assert by_step.pending[rows].all() == (position > 0)
        stepped.append(by_step.predict_step(logits[:, position], rows, position))

    torch.testing.assert_close(torch.stack(stepped, dim=1), expected)
    for name in ("features", "predictions", "pending"):
        assert torch.equal(getattr(by_step, name), getattr(by_block, name))


def _score(estimator, rows, features):
    """Store ``features`` as the latest scored block of ``rows``."""
    estimator.features[rows] = features
    estimator.predictions[rows] = torch.sigmoid(
        estimator.slope * features + estimator.intercepts
    )
    estimator.pending[rows] = True


def test_observe_labels_the_accepted_prefix_and_the_first_scored_rejection():
    estimator = _estimator()
    rows = torch.tensor([3, 9, 4, 12, 15])
    features = torch.arange(20, dtype=torch.float32).reshape(5, BLOCK) / 4.0 - 2.0
    _score(estimator, rows, features)
    estimator.pending[12] = False  # slot 12 holds no scored block
    predictions = estimator.predictions[rows].double()

    # Row 0: two accepted, the third rejected. Row 1: every scored draft
    # accepted, so the window holds no rejection. Row 2: rejected at once.
    # Row 3: nothing to grade. Row 4: a padding row.
    num_accepted = torch.tensor([2, 2, 0, 1, 1], dtype=torch.int32)
    num_verified = torch.tensor([4, 2, 3, 4, 4], dtype=torch.int32)
    valid = torch.tensor([True, True, True, True, False])
    estimator.observe(rows, num_accepted, num_verified, valid)

    graded = {(0, 0): 1.0, (0, 1): 1.0, (0, 2): 0.0, (1, 0): 1.0, (1, 1): 1.0, (2, 0): 0.0}
    expected = torch.zeros(BLOCK, 6, dtype=torch.float64)
    for (row, position), label in graded.items():
        f = float(features[row, position])
        p = float(predictions[row, position])
        w = p * (1.0 - p)
        expected[position] += torch.tensor(
            [w * f * f, w * f, w, (label - p) * f, label - p, 1.0], dtype=torch.float64
        )
    torch.testing.assert_close(estimator.stats, expected, atol=1e-5, rtol=1e-5)
    assert not estimator.pending.any()


def test_a_uniform_step_takes_one_verified_count_for_every_request():
    estimator = _estimator()
    rows = torch.tensor([0, 1])
    _score(estimator, rows, torch.ones(2, BLOCK))

    estimator.observe(rows, torch.tensor([BLOCK, 1]), BLOCK, torch.ones(2, dtype=torch.bool))

    assert estimator.stats[:, 5].tolist() == [2.0, 2.0, 1.0, 1.0]


def test_a_block_is_graded_once_and_dropped_blocks_never():
    estimator = _estimator()
    rows = torch.tensor([0, 1, 2])
    valid = torch.ones(3, dtype=torch.bool)
    accepted = torch.zeros(3, dtype=torch.int32)
    _score(estimator, rows, torch.ones(3, BLOCK))
    estimator.discard(torch.tensor([1]))
    estimator.clear_row(2)

    estimator.observe(rows, accepted, BLOCK, valid)
    estimator.observe(rows, accepted, BLOCK, valid)

    assert estimator.stats[:, 5].tolist() == [1.0, 0.0, 0.0, 0.0]


# A refit takes one Newton step per round, so it needs a realistic number of
# observations per round to be well conditioned.
REFIT_ROWS = 512


def _serve_round(estimator, rows, true_slope, true_intercepts):
    features = torch.rand(REFIT_ROWS, BLOCK) * 8.0
    _score(estimator, rows, features)
    accepted = torch.rand_like(features) < torch.sigmoid(true_slope * features + true_intercepts)
    # Verification stops at the first rejection.
    num_accepted = (~accepted).float().argmax(dim=1)
    num_accepted[accepted.all(dim=1)] = BLOCK
    estimator.observe(rows, num_accepted, BLOCK, torch.ones(REFIT_ROWS, dtype=torch.bool))
    estimator.advance()


def test_refit_recovers_a_shared_slope_and_per_position_intercepts():
    torch.manual_seed(0)
    true_slope = 0.6
    true_intercepts = torch.tensor([1.4, 0.9, 0.4, -0.1])
    estimator = OnlineAcceptanceEstimator(REFIT_ROWS, BLOCK, "cpu")
    estimator.REFIT_INTERVAL = 5
    rows = torch.arange(REFIT_ROWS)

    for _ in range(60):
        _serve_round(estimator, rows, true_slope, true_intercepts)

    assert estimator.num_refits == 12
    assert estimator.slope.item() == pytest.approx(true_slope, abs=0.1)
    torch.testing.assert_close(estimator.intercepts, true_intercepts, atol=0.3, rtol=0)


def test_refit_survives_a_position_with_no_observations():
    torch.manual_seed(0)
    estimator = OnlineAcceptanceEstimator(REFIT_ROWS, BLOCK, "cpu")
    estimator.REFIT_INTERVAL = 5
    rows = torch.arange(REFIT_ROWS)
    slope = estimator.slope.clone()
    intercepts = estimator.intercepts.clone()

    # Every request is rejected at position 1, so positions 2+ are never graded.
    for _ in range(20):
        _score(estimator, rows, torch.rand(REFIT_ROWS, BLOCK) * 8.0)
        estimator.observe(
            rows, torch.ones(REFIT_ROWS), BLOCK, torch.ones(REFIT_ROWS, dtype=torch.bool)
        )
        estimator.advance()

    assert torch.isfinite(estimator.slope)
    assert torch.isfinite(estimator.intercepts).all()
    assert not torch.equal(estimator.slope, slope)
    assert torch.equal(estimator.intercepts[2:], intercepts[2:])


def test_a_degenerate_round_leaves_the_coefficients_finite():
    """One feature value per position leaves the slope unidentified."""
    estimator = OnlineAcceptanceEstimator(REFIT_ROWS, BLOCK, "cpu")
    rows = torch.arange(REFIT_ROWS)
    for _ in range(3):
        _score(estimator, rows, torch.full((REFIT_ROWS, BLOCK), 40.0))
        estimator.observe(
            rows, torch.zeros(REFIT_ROWS), BLOCK, torch.ones(REFIT_ROWS, dtype=torch.bool)
        )
        estimator.refit()

    assert torch.isfinite(estimator.slope) and estimator.slope >= 0.0
    assert torch.isfinite(estimator.intercepts).all()


def test_step_damping_scales_with_the_evidence():
    """A round moves a coefficient in proportion to the samples behind it.

    Scaling a round's statistics leaves the Newton step unchanged, which
    isolates the factor ``n / (n + DAMPING_OBSERVATIONS)``. The slope/intercept
    coupling is zero so the two can be checked apart.
    """
    damping = OnlineAcceptanceEstimator.DAMPING_OBSERVATIONS
    per_position = 20.0

    def refit_with(scale):
        estimator = OnlineAcceptanceEstimator(NUM_ROWS, BLOCK, "cpu")
        estimator.L2 = 0.0
        estimator.stats[:, 0] = 30.0 * scale / BLOCK
        estimator.stats[:, 2] = 1.0 * scale
        estimator.stats[:, 3] = 2.0 * scale / BLOCK
        estimator.stats[:, 4] = 0.5 * scale
        estimator.stats[:, 5] = per_position * scale
        slope = estimator.slope.clone()
        intercepts = estimator.intercepts.clone()
        estimator.refit()
        assert not estimator.stats.any()
        return (estimator.slope - slope).item(), (estimator.intercepts - intercepts)[0].item()

    for scale in (1.0, 10.0, 100.0):
        total = per_position * scale * BLOCK
        slope_step, intercept_step = refit_with(scale)
        assert slope_step == pytest.approx((2.0 / 30.0) * total / (total + damping), rel=1e-4)
        assert intercept_step == pytest.approx(
            0.5 * per_position * scale / (per_position * scale + damping), rel=1e-4
        )


def test_a_round_without_observations_leaves_the_coefficients_alone():
    estimator = _estimator()
    slope = estimator.slope.clone()
    intercepts = estimator.intercepts.clone()

    estimator.refit()

    assert torch.equal(estimator.slope, slope)
    assert torch.equal(estimator.intercepts, intercepts)
    assert estimator.num_refits == 1


def test_a_step_toward_a_negative_slope_is_dropped():
    estimator = OnlineAcceptanceEstimator(NUM_ROWS, BLOCK, "cpu")
    estimator.stats[:, 0] = 1.0
    estimator.stats[:, 2] = 1.0
    estimator.stats[:, 3] = -1000.0
    estimator.stats[:, 5] = 1000.0

    estimator.refit()

    assert estimator.slope.item() == pytest.approx(0.5)


def test_advance_refits_once_per_interval():
    estimator = _estimator()
    estimator.REFIT_INTERVAL = 3

    for _ in range(7):
        estimator.advance()

    assert estimator.num_refits == 2


def test_restore_puts_back_everything_a_forward_changes():
    torch.manual_seed(0)
    estimator = _estimator()
    rows = torch.tensor([1, 4])
    estimator.predict(torch.randn(2, BLOCK, VOCAB), rows)
    saved = estimator.snapshot()
    state = (estimator.features, estimator.predictions, estimator.pending, estimator.stats)
    expected = [tensor.clone() for tensor in state]
    addresses = [tensor.data_ptr() for tensor in state]

    estimator.observe(rows, torch.tensor([1, 0]), BLOCK, torch.ones(2, dtype=torch.bool))
    estimator.predict(torch.randn(3, BLOCK, VOCAB), torch.tensor([1, 2, 3]))
    estimator.restore(saved)

    for tensor, value, address in zip(state, expected, addresses):
        assert torch.equal(tensor, value)
        assert tensor.data_ptr() == address


def test_host_refit_updates_buffers_the_forward_wrote_in_inference_mode():
    estimator = OnlineAcceptanceEstimator(NUM_ROWS, BLOCK, "cpu")
    rows = torch.arange(NUM_ROWS)
    with torch.inference_mode():
        estimator.predict(torch.randn(NUM_ROWS, BLOCK, VOCAB) * 4.0, rows)
        estimator.observe(
            rows, torch.ones(NUM_ROWS), BLOCK, torch.ones(NUM_ROWS, dtype=torch.bool)
        )
    intercepts = estimator.intercepts.clone()

    estimator.clear_row(0)
    estimator.refit()

    assert not torch.equal(estimator.intercepts, intercepts)
