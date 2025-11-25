# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        get_draft_token_length)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig


class MambaCacheManager(BaseResourceManager):

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        spec_config: Optional["DecodingBaseConfig"] = None,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype
        self.spec_config = spec_config
        self.max_batch_size = max_batch_size

        # get tp size
        tp_size = mapping.tp_size

        # derive mamba parameters for conv and ssm states
        d_inner = head_dim * num_heads
        conv_dim = d_inner + 2 * n_groups * d_state
        nheads = num_heads

        # check that can be partitioned
        assert nheads % tp_size == 0, "nheads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"

        # partition conv_dim and nheads
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size

        # conv and ssm states device
        device = torch.device("cuda")

        pp_layers, num_layers = get_pp_layers(
            num_layers,
            mapping,
            layer_mask=layer_mask,
        )
        num_local_layers = len(pp_layers)
        self.mamba_layer_offsets = {
            idx: offset
            for offset, idx in enumerate(pp_layers)
        }

        # mamba conv states
        self.conv_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                conv_dim,
                d_conv - 1,
            ],
            dtype=dtype,
            device=device,
        )

        # mamba ssm states
        self.ssm_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                nheads,
                head_dim,
                d_state,
            ],
            dtype=self.mamba_ssm_cache_dtype,
            device=device,
        )

        # mamba cache available blocks
        self.mamba_cache_free_blocks = [i for i in range(max_batch_size)]

        # mamba cache index, maps request_id -> state indices
        self.mamba_cache_index: Dict[int, int] = {}

        # mamba cache state indices
        self.state_indices: torch.Tensor = torch.arange(max_batch_size,
                                                        device=device,
                                                        dtype=torch.int32)

        # For speculative decoding: saved mamba states before processing draft tokens
        # These are used to restore state when draft tokens are rejected
        self._needs_spec_dec_state_save = (
            spec_config is not None
            and spec_config.spec_dec_mode.needs_kv_cache_rewind())
        if self._needs_spec_dec_state_save:
            # Allocate backup buffers for state restoration
            self._saved_conv_states = torch.empty_like(self.conv_states)
            self._saved_ssm_states = torch.empty_like(self.ssm_states)
            # Track which request slots have saved state: request_id -> slot
            self._saved_state_slots: Dict[int, int] = {}
            # Track tokens that need to be reprocessed through mamba after state restore
            # request_id -> list of token ids
            self._tokens_to_reprocess: Dict[int, List[int]] = {}
        else:
            self._saved_conv_states = None
            self._saved_ssm_states = None
            self._saved_state_slots = {}
            self._tokens_to_reprocess = {}

    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        state_indices = []
        for r in request_ids:
            # cache hit
            if r in self.mamba_cache_index:
                state_indices.append(self.mamba_cache_index[r])
            # cache miss
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise Exception("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block
                state_indices.append(block)
        self.state_indices[:len(state_indices)] = torch.as_tensor(
            state_indices, dtype=torch.int32, device=self.ssm_states.device)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)

        # For speculative decoding: save mamba state before processing draft tokens
        # so we can restore it if tokens are rejected
        if self._needs_spec_dec_state_save:
            self._save_mamba_state_for_spec_dec(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id in self.mamba_cache_index:
            block = self.mamba_cache_index.pop(request_id)
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self) -> torch.Tensor:
        return self.state_indices

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.conv_states[layer_offset]

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.ssm_states[layer_offset]

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.mamba_ssm_cache_dtype

    def _save_mamba_state_for_spec_dec(
            self, scheduled_batch: ScheduledRequests) -> None:
        """Save mamba state for requests with draft tokens before forward pass.

        When draft tokens are rejected, we need to restore the mamba state to
        what it was before the rejected tokens were processed. This method saves
        a copy of the state for each request that has draft tokens.

        IMPORTANT: We do NOT clear _saved_state_slots here because in the overlap
        scheduler, prepare_resources(batch_N+1) is called BEFORE update_resources(batch_N).
        The saved state from batch_N must persist until update_resources(batch_N) is called.
        We only save state for requests that don't already have saved state waiting
        to be restored.
        """
        slots_to_save = []
        for request in scheduled_batch.generation_requests:
            # Only save state for requests that have draft tokens
            if get_draft_token_length(request) > 0:
                request_id = request.py_request_id
                # Only save if not already saved (preserve first save until restore)
                if request_id not in self._saved_state_slots:
                    if request_id in self.mamba_cache_index:
                        state_slot = self.mamba_cache_index[request_id]
                        self._saved_state_slots[request_id] = state_slot
                        slots_to_save.append(state_slot)

            # Save conv states: [num_layers, batch, conv_dim, d_conv-1]
            for slot in slots_to_save:
                self._saved_conv_states[:, slot].copy_(self.conv_states[:,
                                                                        slot])

                # Save ssm states: [num_layers, batch, nheads, head_dim, d_state]
                self._saved_ssm_states[:, slot].copy_(self.ssm_states[:, slot])

    def _restore_mamba_state_for_spec_dec(
            self, scheduled_batch: ScheduledRequests) -> None:
        """Restore mamba state for requests where ANY draft tokens were rejected.

        After verification, we ALWAYS restore the saved mamba state when any draft
        tokens were rejected (py_rewind_len > 0). This ensures the mamba state is
        clean and only reflects tokens up to before the draft tokens.

        For partial acceptance (some tokens accepted, some rejected):
        - We restore to the state before all draft tokens
        - We track the accepted tokens for reprocessing in the next forward pass
        - The next forward pass will include these tokens to update the mamba state

        This ensures model outputs are exactly the same as without speculation.
        """
        if not self._saved_state_slots:
            return

        slots_to_restore = []
        request_ids_to_remove = []

        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id

            if request_id not in self._saved_state_slots:
                continue

            request_ids_to_remove.append(request_id)

            # Restore state if ANY tokens were rejected
            if request.py_rewind_len > 0:
                slots_to_restore.append(self._saved_state_slots[request_id])

                # If there were accepted tokens, track them for reprocessing
                num_accepted = request.py_num_accepted_draft_tokens
                if num_accepted > 0:
                    # Get the accepted tokens from the request's token sequence
                    # The accepted tokens are the last num_accepted tokens added
                    all_tokens = list(request.get_tokens(0))
                    # The accepted tokens are at the end of the sequence
                    accepted_tokens = all_tokens[-num_accepted:]
                    self._tokens_to_reprocess[request_id] = accepted_tokens

        for slot in slots_to_restore:
            # Restore conv states
            self.conv_states[:, slot].copy_(self._saved_conv_states[:, slot])
            # Restore ssm states
            self.ssm_states[:, slot].copy_(self._saved_ssm_states[:, slot])

        # Remove processed entries from saved state tracking
        for rid in request_ids_to_remove:
            self._saved_state_slots.pop(rid, None)

    def update_resources(self, scheduled_batch: ScheduledRequests) -> None:
        """Update mamba resources after forward pass and verification.

        For speculative decoding, this restores mamba state for requests
        where draft tokens were rejected.
        """
        if self._needs_spec_dec_state_save:
            self._restore_mamba_state_for_spec_dec(scheduled_batch)

    def get_tokens_to_reprocess(self, request_id: int) -> List[int]:
        """Get tokens that need to be reprocessed through mamba for state consistency.

        After partial acceptance in speculative decoding, we restore mamba state to
        before the draft tokens. The accepted tokens need to be reprocessed through
        mamba to update the state correctly.

        Returns:
            List of token IDs that need reprocessing, or empty list if none.
            The returned tokens are removed from tracking after this call.
        """
        return self._tokens_to_reprocess.pop(request_id, [])

    def shutdown(self):
        # release tensor memory, keeping python references as tensors
        self.conv_states = torch.tensor([])
        self.ssm_states = torch.tensor([])
        self.state_indices = torch.tensor([])
        if self._saved_conv_states is not None:
            self._saved_conv_states = torch.tensor([])
        if self._saved_ssm_states is not None:
            self._saved_ssm_states = torch.tensor([])
        self._saved_state_slots.clear()
        self._tokens_to_reprocess.clear()
        torch.cuda.empty_cache()


class MambaHybridCacheManager(KVCacheManager, MambaCacheManager):

    def __init__(
        self,
        # mamba cache parameters
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_num_heads: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        mamba_ssm_cache_dtype: torch.dtype,

        # kv cache parameters
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        layer_mask: List[bool],
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        is_estimating_kv_cache: bool = False,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, "mamba hybrid cache requires block reuse to be disabled in KV cache config"

        # initialize mamba cache manager (with spec_config for speculative decoding support)
        MambaCacheManager.__init__(
            self,
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
            spec_config,
        )

        # initialize kv cache manager
        KVCacheManager.__init__(
            self,
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            is_estimating_kv_cache=is_estimating_kv_cache,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        """Update resources after forward pass and verification.

        For mamba hybrid models with speculative decoding, this handles:
        1. KV cache rewind for rejected tokens (via KVCacheManager)
        2. Mamba state restoration for rejected tokens (via MambaCacheManager)
        """
        # First restore mamba state for rejected tokens
        MambaCacheManager.update_resources(self, scheduled_batch)
        # Then handle KV cache rewind
        KVCacheManager.update_resources(self, scheduled_batch, attn_metadata,
                                        kv_cache_dtype_byte_size)

    def free_resources(self, request: LlmRequest):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request)

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)
