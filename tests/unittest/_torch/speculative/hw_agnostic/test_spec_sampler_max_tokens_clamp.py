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
"""SpecSamplerBase must clamp accepted tokens to the remaining max_new_tokens budget.

Eagle3 / MTP one-model verification can return accepted_drafts + 1 (bonus) tokens
in a single step. With overlap scheduling and OSL/max_tokens=1, that raw
acceptance must not commit a second output token.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSamplerBase,
)
from tensorrt_llm.bindings import SamplingConfig

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for SpecSamplerBase", allow_module_level=True)


def _make_request(*, prompt: list[int], max_new_tokens: int, seq_slot: int = 0) -> LlmRequest:
    return LlmRequest(
        request_id=seq_slot,
        seq_slot=seq_slot,
        input_tokens=prompt,
        max_new_tokens=max_new_tokens,
        end_id=None,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _make_sample_state(
    request: LlmRequest,
    *,
    accepted_tokens: list[int],
    draft_len: int,
) -> SampleStateSpec:
    """Build a SampleStateSpec with host tensors for update_requests."""
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    num_accepted = len(accepted_tokens)
    # new_tokens layout: [step, seq_slot, beam]
    new_tokens = torch.zeros((max(num_accepted, 1), seq_slot + 1, 1), dtype=torch.int32)
    for step, tok in enumerate(accepted_tokens):
        new_tokens[step, seq_slot, 0] = tok
    new_tokens_lens = torch.zeros((seq_slot + 1,), dtype=torch.int32)
    new_tokens_lens[seq_slot] = num_accepted
    next_draft_tokens = torch.zeros((seq_slot + 1, draft_len), dtype=torch.int32)

    host = SampleStateTensorsSpec(
        new_tokens=new_tokens,
        new_tokens_lens=new_tokens_lens,
        next_draft_tokens=next_draft_tokens,
    )
    device = SampleStateTensorsSpec(
        new_tokens=new_tokens.clone(),
        new_tokens_lens=new_tokens_lens.clone(),
        next_draft_tokens=next_draft_tokens.clone(),
    )
    sampler_event = MagicMock()
    sampler_event.synchronize = MagicMock()
    return SampleStateSpec(
        requests=[request],
        device=device,
        host=host,
        sampler_event=sampler_event,
        runtime_draft_len=draft_len,
    )


def _make_sampler(draft_len: int) -> SpecSamplerBase:
    return SpecSamplerBase(
        TorchSampler.Args(
            max_seq_len=128,
            max_draft_len=draft_len,
            max_total_draft_tokens=draft_len,
            max_num_sequences=4,
            max_beam_width=1,
            disable_overlap_scheduler=False,
        ),
        draft_len=draft_len,
    )


@pytest.mark.parametrize(
    "max_new_tokens,accepted,expected_generated",
    [
        # OSL=1 but verify returns bonus+draft → must emit exactly 1 token.
        (1, [101, 102, 103], 1),
        # Remaining budget 2 with 3 accepted → clamp to 2.
        (2, [101, 102, 103], 2),
        # Acceptance within budget → keep all.
        (4, [101, 102], 2),
    ],
)
def test_spec_sampler_clamps_to_max_new_tokens(
    max_new_tokens: int, accepted: list[int], expected_generated: int
):
    draft_len = 3
    prompt = [1, 2, 3]
    request = _make_request(prompt=prompt, max_new_tokens=max_new_tokens)
    assert request.state != LlmRequestState.GENERATION_COMPLETE

    sampler = _make_sampler(draft_len)
    state = _make_sample_state(request, accepted_tokens=accepted, draft_len=draft_len)
    sampler.update_requests(state)

    generated = request.get_num_tokens(0) - request.py_orig_prompt_len
    assert generated == expected_generated
    assert request.get_tokens(0)[len(prompt) :] == accepted[:expected_generated]
    # Rewind metadata must match tokens actually committed, not raw acceptance.
    assert request.py_num_accepted_draft_tokens == max(0, expected_generated - 1)
    assert request.py_rewind_len == draft_len - request.py_num_accepted_draft_tokens
    if expected_generated >= max_new_tokens:
        assert request.state == LlmRequestState.GENERATION_COMPLETE


def test_spec_sampler_updates_num_tokens_per_iteration_for_will_complete():
    """Overlap will_complete_next_iteration must see the tokens just committed.

    Without updating mNumTokensPerIteration, a 3-token speculative step leaves
    the default of 1, so will_complete under-estimates near the length budget.
    """
    draft_len = 2
    # After committing 3 tokens, remaining budget is 2; the next speculative
    # step can still emit up to draft_len+1, so will_complete must be True.
    request = _make_request(prompt=[1, 2], max_new_tokens=5)
    sampler = _make_sampler(draft_len)
    accepted = [10, 11, 12]
    state = _make_sample_state(request, accepted_tokens=accepted, draft_len=draft_len)
    sampler.update_requests(state)
    assert request.get_num_tokens(0) - request.py_orig_prompt_len == 3
    assert request.will_complete_next_iteration()
