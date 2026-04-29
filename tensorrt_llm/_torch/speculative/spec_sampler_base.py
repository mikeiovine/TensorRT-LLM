# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Base class for speculative decoding samplers.

This module provides a common base class for MTPSampler, SASampler, and
Eagle3OneModelSampler.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import FinishReason, LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    add_token,
    int_tensor,
)
from ..pyexecutor.scheduler import ScheduledRequests


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec


class SpecSamplerBase(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Base class for speculative decoding samplers (MTP, NGram, Eagle3, SA).

    Provides common functionality:
    - Pre-allocated GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    Subclasses can customize behavior by overriding:
    - _get_max_tokens(): How to calculate max_tokens for storage
    - _get_draft_tokens_storage_size(): Size of next_draft_tokens tensor
    - _add_dummy_draft_tokens(): Whether to add dummy drafts for context requests
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor

    def __init__(self, args: TorchSampler.Args, *, draft_len: int):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            draft_len: Maximum number of draft tokens per iteration.
        """
        self.mapping = None
        self.draft_len = draft_len
        self.max_seq_len = args.max_seq_len

        seq_slots = args.max_num_sequences
        max_tokens = self._get_max_tokens(args, draft_len)
        max_new_tokens = self._get_max_new_tokens(args, draft_len)
        draft_tokens_size = self._get_draft_tokens_storage_size(args, draft_len)
        self.max_beam_width = args.max_beam_width

        self.store = self.Store(
            new_tokens=int_tensor((max_new_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_tokens_size)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )

    def _get_max_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate max_tokens for storage allocation.

        Override in subclasses if needed. Default: draft_len + 1.
        MTP uses args.max_total_draft_tokens + 1 for tree-based speculation.
        """
        return draft_len + 1

    def _get_max_new_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """Max depth of accepted token path for new_tokens buffer.

        Defaults to _get_max_tokens (same size as next_new_tokens).
        Override when accepted path depth differs from total draft tokens,
        e.g. dynamic tree where max_draft_len < max_total_draft_tokens.
        """
        return self._get_max_tokens(args, draft_len)

    def _get_draft_tokens_storage_size(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate storage size for next_draft_tokens tensor.

        Override in subclasses if needed. Default: draft_len.
        MTP uses args.max_total_draft_tokens for tree-based speculation.
        """
        return draft_len

    def _add_dummy_draft_tokens(self) -> bool:
        """
        Whether to add dummy draft tokens for context requests.

        Override in subclasses. Default: True (needed for KV cache preparation).
        """
        return True

    @staticmethod
    def _handle_beam_stop_criteria(
        request: LlmRequest, new_token: int, *, max_seq_len: int, beam_idx: int
    ) -> bool:
        if new_token == request.py_end_id:
            request.set_finished_reason(FinishReason.END_ID, beam_idx)
            return True

        if TorchSampler._meet_max_token_stop_criteria(request, max_seq_len, beam_idx):
            request.set_finished_reason(FinishReason.LENGTH, beam_idx)
            return True

        if TorchSampler._meet_stop_token_criteria(request, new_token, beam_idx):
            request.set_finished_reason(FinishReason.STOP_WORDS, beam_idx)
            return True

        return False

    @staticmethod
    def _is_beam_finished(request: LlmRequest, beam_idx: int) -> bool:
        if hasattr(request, "get_finished_reason"):
            return request.get_finished_reason(beam_idx) != FinishReason.NOT_FINISHED
        return False

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        if request.py_return_context_logits:
            logger.warning(
                "return_context_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_generation_logits:
            logger.warning(
                "return_generation_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_log_probs:
            logger.warning(
                "return_log_probs not supported with speculative decoding, skipping for request %s",
                request.py_request_id,
            )
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    def update_requests(
        self,
        state: SampleStateSpec,
        resource_manager: Optional[BaseResourceManager] = None,
    ) -> None:
        """
        CPU-side request updates after GPU->CPU sync.

        Waits for async copy to complete, then updates request state with:
        - Accepted tokens
        - Stop criteria checks
        - Next iteration draft tokens
        """
        assert isinstance(state, SampleStateSpec)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        for req in state.requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            seq_slot = req.py_seq_slot
            assert seq_slot is not None
            req_beam_width = req.sampling_config.beam_width
            if req_beam_width > self.max_beam_width:
                raise ValueError(
                    f"request beam width {req_beam_width} exceeds max beam width {self.max_beam_width}"
                )

            num_new_tokens = new_tokens_lens_list[seq_slot]

            all_beams_finished = req_beam_width > 1
            for beam_idx in range(req_beam_width):
                beam_finished = self._is_beam_finished(req, beam_idx)
                for i in range(num_new_tokens):
                    new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                    if req_beam_width == 1:
                        beam_finished = TorchSampler._handle_stop_criteria(
                            req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                        )
                    elif self._handle_beam_stop_criteria(
                        req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                    ):
                        beam_finished = True
                        break
                all_beams_finished = all_beams_finished and beam_finished

            if req_beam_width > 1 and all_beams_finished:
                req.state = LlmRequestState.GENERATION_COMPLETE

            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            req.py_rewind_len = runtime_draft_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
    ) -> SampleStateSpec:
        """
        Async sampling - schedules GPU->CPU copy.
        Called after CUDA graph replay.

        Args:
            scheduled_requests: Batch of scheduled requests
            outputs: Dict from worker forward() containing:
                - new_tokens: [batch, max_draft_len + 1] accepted tokens
                - new_tokens_lens: [batch] number of accepted tokens
                - next_draft_tokens: [batch, max_draft_len] draft tokens for next iter
                - next_new_tokens: [batch, max_draft_len + 1] input for next iter
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests
        num_sampling_requests = len(sampling_requests)

        slots = torch.as_tensor([r.py_seq_slot for r in sampling_requests], dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]
        runtime_draft_len = o_next_draft_tokens.shape[1]

        if o_new_tokens.dim() == 2:
            o_new_tokens = o_new_tokens.unsqueeze(-1).expand(-1, -1, self.max_beam_width)
        if o_next_new_tokens.dim() == 2:
            o_next_new_tokens = o_next_new_tokens.unsqueeze(-1).expand(-1, -1, self.max_beam_width)
        if o_new_tokens_lens.dim() != 1:
            if o_new_tokens_lens.shape[-1] != self.max_beam_width:
                raise ValueError("unexpected new_tokens_lens width for speculative decoding")
            if not torch.equal(
                o_new_tokens_lens, o_new_tokens_lens[:, :1].expand_as(o_new_tokens_lens)
            ):
                raise ValueError(
                    "1-model speculative decoding requires equal accepted length across beams"
                )
            o_new_tokens_lens = o_new_tokens_lens[:, 0]

        # Pad or truncate to match fixed-size store buffers for index_copy_.
        # Use actual store buffer dimensions (which may differ from draft_len
        # when _get_max_new_tokens is overridden, e.g. dynamic tree mode).
        new_tokens_width = self.store.new_tokens.shape[0]
        next_new_tokens_width = self.store.next_new_tokens.shape[0]
        draft_tokens_width = self.store.next_draft_tokens.shape[1]
        if o_new_tokens.shape[1] < new_tokens_width:
            o_new_tokens = torch.nn.functional.pad(
                o_new_tokens, (0, new_tokens_width - o_new_tokens.shape[1])
            )
        elif o_new_tokens.shape[1] > new_tokens_width:
            o_new_tokens = o_new_tokens[:, :new_tokens_width]
        if o_next_draft_tokens.shape[1] < draft_tokens_width:
            o_next_draft_tokens = torch.nn.functional.pad(
                o_next_draft_tokens, (0, draft_tokens_width - o_next_draft_tokens.shape[1])
            )
        elif o_next_draft_tokens.shape[1] > draft_tokens_width:
            o_next_draft_tokens = o_next_draft_tokens[:, :draft_tokens_width]
        if o_next_new_tokens.shape[1] < next_new_tokens_width:
            o_next_new_tokens = torch.nn.functional.pad(
                o_next_new_tokens, (0, next_new_tokens_width - o_next_new_tokens.shape[1])
            )
        elif o_next_new_tokens.shape[1] > next_new_tokens_width:
            o_next_new_tokens = o_next_new_tokens[:, :next_new_tokens_width]

        # Use index_copy_ for efficient copying (slots are unique)
        self.store.new_tokens.permute(1, 0, 2).index_copy_(0, slots, o_new_tokens)
        self.store.next_new_tokens.permute(1, 0, 2).index_copy_(0, slots, o_next_new_tokens)
        self.store.new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        self.store.next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
        )

        host_tensors = SampleStateTensorsSpec(
            new_tokens=self._copy_to_host(self.store.new_tokens),
            new_tokens_lens=self._copy_to_host(self.store.new_tokens_lens),
            next_draft_tokens=self._copy_to_host(self.store.next_draft_tokens),
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        if self._add_dummy_draft_tokens():
            for request in finished_context_requests:
                request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
        )
