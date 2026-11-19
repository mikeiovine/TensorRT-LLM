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
"""Per-position draft acceptance probabilities without a trained confidence head.

Confidence-scheduled verification ranks every ``(request, draft position)`` by
the probability that the target accepts it. A drafter with a trained confidence
head predicts that directly. :class:`OnlineAcceptanceEstimator` supplies it for
any drafter that materializes its draft logits, by fitting a logistic
regression while serving::

    p_accept(k) = sigmoid(slope * logit(q_max) + intercept[k])

``q_max`` is the probability the draft distribution gives its most likely
token. The slope is shared by all draft positions and each position keeps its
own intercept. The labels are the target's own verdicts on earlier drafts.

The feature scores the draft distribution, not the token drawn from it, so the
decision to verify a position never depends on the token at that position and
rejection sampling stays lossless.
"""

import torch

__all__ = ["OnlineAcceptanceEstimator", "max_token_log_odds"]

# A row with all of its mass on one token has infinite log-odds.
_MAX_LOG_ODDS = 40.0

# Cold-start coefficients. Log-loss is flat near the optimum, so one starting
# point serves every drafter until the first refits replace it.
_INIT_SLOPE = 0.5
_INIT_INTERCEPT = -0.2


def max_token_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Log-odds of the most likely token of each distribution.

    Args:
        logits: ``[..., vocab]`` logits of the distribution the draft token is
            drawn from, over the whole vocabulary.
    Returns:
        ``[...]`` float32 ``log(q_max / (1 - q_max))``, clamped to a finite range.
    """
    logits = logits.float()
    top = logits.amax(dim=-1, keepdim=True)
    # Shifted by the maximum, the most likely token contributes exp(0) = 1 to
    # the sum and every other token contributes its odds against it.
    others = torch.exp(logits - top).sum(dim=-1) - 1.0
    return (-torch.log(others.clamp_min(0.0))).clamp(-_MAX_LOG_ODDS, _MAX_LOG_ODDS)


class OnlineAcceptanceEstimator:
    """Predicts per-position acceptance and calibrates itself while serving.

    Rows are the caller's persistent request slots. Each decode step runs:

    1. :meth:`observe` inside the forward, after acceptance: folds the target's
       verdict on the block each request just verified into the fit statistics.
    2. :meth:`predict` inside the forward, after drafting: scores the new block
       and returns its confidence logits.
    3. :meth:`advance` on the host, between forwards: every
       ``REFIT_INTERVAL`` steps takes one damped Newton step on the
       coefficients and starts a new round of statistics.

    ``observe`` and ``predict`` use fixed shapes and never synchronize with the
    host, so both can be captured in a CUDA graph. Trimming can start at once:
    survival is a running product of sigmoids, so it decreases with draft
    position whatever the coefficients are.

    Args:
        num_rows: number of request slots, including any scratch rows.
        num_positions: draft block length.
        device: device of the draft logits.
    """

    # Decode steps between refits.
    REFIT_INTERVAL = 100
    # A Newton step is scaled by n / (n + DAMPING_OBSERVATIONS), so a
    # coefficient backed by this many observations moves half a full step and a
    # draft position nobody reached in a round does not move.
    DAMPING_OBSERVATIONS = 50.0
    # Ridge on the Newton solve.
    L2 = 1e-3

    def __init__(self, num_rows: int, num_positions: int, device: torch.device | str) -> None:
        self.num_positions = int(num_positions)
        self.num_refits = 0
        self._steps_since_refit = 0
        # The forward writes these in inference mode and the host updates them
        # outside it, which an inference tensor would reject.
        with torch.inference_mode(False):
            self.slope = torch.full((), _INIT_SLOPE, dtype=torch.float32, device=device)
            self.intercepts = torch.full(
                (self.num_positions,), _INIT_INTERCEPT, dtype=torch.float32, device=device
            )
            # Feature and predicted probability of each row's latest block.
            self.features = torch.zeros(
                (num_rows, self.num_positions), dtype=torch.float32, device=device
            )
            self.predictions = torch.zeros_like(self.features)
            # Rows whose latest block has been scored but not graded yet.
            self.pending = torch.zeros(num_rows, dtype=torch.bool, device=device)
            # Per-position sums of this round, over the graded drafts. With
            # feature f, prediction p, label y and weight w = p * (1 - p), the
            # columns are [w*f*f, w*f, w, (y-p)*f, (y-p), 1]. Float64 keeps the
            # arrowhead solve in refit() well conditioned.
            self.stats = torch.zeros((self.num_positions, 6), dtype=torch.float64, device=device)
            self._positions = torch.arange(self.num_positions, device=device)

    def predict(self, logits: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Score one draft block per request.

        Args:
            logits: ``[N, num_positions, vocab]`` logits each draft token is
                drawn from. A drafter that samples must pass the logits scaled
                by its sampling temperature.
            rows: ``[N]`` slot of each request.
        Returns:
            ``[N, num_positions]`` float32 confidence logits. Their sigmoid is
            the predicted conditional acceptance probability.
        """
        features = max_token_log_odds(logits)
        confidence_logits = self.slope * features + self.intercepts
        self.features[rows] = features
        self.predictions[rows] = torch.sigmoid(confidence_logits)
        self.pending.index_fill_(0, rows, True)
        return confidence_logits

    def predict_step(self, logits: torch.Tensor, rows: torch.Tensor, position: int) -> torch.Tensor:
        """Score one draft position per request, for a drafter that drafts step by step.

        A block starts at position 0 and the caller scores every later position
        of it in the same forward.

        Args:
            logits: ``[N, vocab]`` logits the draft tokens of ``position`` are
                drawn from, scaled like those of :meth:`predict`.
            rows: ``[N]`` slot of each request.
            position: draft position of this step.
        Returns:
            ``[N]`` float32 confidence logits of ``position``.
        """
        features = max_token_log_odds(logits)
        confidence_logits = self.slope * features + self.intercepts[position]
        self.features[rows, position] = features
        self.predictions[rows, position] = torch.sigmoid(confidence_logits)
        if position == 0:
            self.pending.index_fill_(0, rows, True)
        return confidence_logits

    def observe(
        self,
        rows: torch.Tensor,
        num_accepted: torch.Tensor,
        num_verified: torch.Tensor | int,
        valid: torch.Tensor,
    ) -> None:
        """Grade the block each request just verified.

        Verification stops at the first rejection, so a request labels its
        accepted drafts and the first rejected one, if the target scored it.

        Args:
            rows: ``[N]`` slot of each request.
            num_accepted: ``[N]`` accepted draft tokens, without the bonus token.
            num_verified: draft tokens the target scored, ``[N]`` or one count
                for every request.
            valid: ``[N]`` bool, False for rows that hold no real request.
        """
        graded = (valid & self.pending.index_select(0, rows)).unsqueeze(1)
        num_accepted = num_accepted.unsqueeze(1)
        if isinstance(num_verified, torch.Tensor):
            num_verified = num_verified.unsqueeze(1)
        positions = self._positions
        observed = graded & (positions <= num_accepted) & (positions < num_verified)
        labels = (positions < num_accepted).float()

        features = torch.where(observed, self.features.index_select(0, rows), 0.0)
        predictions = self.predictions.index_select(0, rows)
        weights = torch.where(observed, predictions * (1.0 - predictions), 0.0)
        residuals = torch.where(observed, labels - predictions, 0.0)
        self.stats += torch.stack(
            (
                weights * features * features,
                weights * features,
                weights,
                residuals * features,
                residuals,
                observed.float(),
            ),
            dim=-1,
        ).sum(dim=0)
        self.pending.index_fill_(0, rows, False)

    def discard(self, rows: torch.Tensor) -> None:
        """Drop the ungraded blocks of ``rows``, which will not be verified."""
        self.pending.index_fill_(0, rows, False)

    def clear_row(self, row: int) -> None:
        """Drop the ungraded block of one slot, when a new request takes it."""
        self.pending[row].zero_()

    def advance(self) -> None:
        """Count one decode step and refit every ``REFIT_INTERVAL`` steps.

        Runs on the host between two forwards, on the stream that runs them.
        """
        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0
        self.refit()

    @torch.no_grad()
    def refit(self) -> None:
        """Take one damped Newton step on this round's statistics and clear them.

        For coefficients ``[slope, intercept_0 .. intercept_{K-1}]`` the system
        ``(X^T W X + L2 * I) step = X^T (y - p)`` is an arrowhead: only the
        slope couples to every intercept. Eliminating the intercepts leaves a
        scalar equation for the slope step.
        """
        slope_info, coupling, intercept_info, slope_grad, intercept_grad, counts = (
            self.stats.unbind(dim=1)
        )
        intercept_info = intercept_info + self.L2
        slope_step = (slope_grad.sum() - (coupling * intercept_grad / intercept_info).sum()) / (
            slope_info.sum() + self.L2 - (coupling * coupling / intercept_info).sum()
        )
        # The slope learns from every position, so the whole round damps it.
        total = counts.sum()
        slope_step = slope_step * total / (total + self.DAMPING_OBSERVATIONS)
        # A negative slope would rank confident drafts last.
        keep = torch.isfinite(slope_step) & (self.slope + slope_step >= 0.0)
        slope_step = torch.where(keep, slope_step, torch.zeros_like(slope_step))

        intercept_step = (intercept_grad - coupling * slope_step) / intercept_info
        intercept_step = intercept_step * counts / (counts + self.DAMPING_OBSERVATIONS)
        intercept_step = torch.where(
            torch.isfinite(intercept_step), intercept_step, torch.zeros_like(intercept_step)
        )

        self.slope.add_(slope_step.to(self.slope.dtype))
        self.intercepts.add_(intercept_step.to(self.intercepts.dtype))
        self.stats.zero_()
        self.num_refits += 1

    def snapshot(self) -> tuple[torch.Tensor, ...]:
        """Copies of the state a forward changes, for :meth:`restore`."""
        return (
            self.features.clone(),
            self.predictions.clone(),
            self.pending.clone(),
            self.stats.clone(),
        )

    def restore(self, saved: tuple[torch.Tensor, ...]) -> None:
        """Put back a :meth:`snapshot` in place."""
        features, predictions, pending, stats = saved
        self.features.copy_(features)
        self.predictions.copy_(predictions)
        self.pending.copy_(pending)
        self.stats.copy_(stats)
