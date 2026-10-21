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
"""A separate draft KV pool travels with the target's, as extra layer groups of one page table.

A hidden-state drafter (DFlash, standalone DSpark) keeps per-prompt-token K/V in its own pool,
which a disaggregated generation worker cannot recompute without the target's hidden states.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from kv_cache_manager_v2_test_utils import allocate_request, initialize_cache, request_block_data

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import CacheKind, TokenRange
from tensorrt_llm._torch.disaggregation.native.extent import Chunk
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo, Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    DRAFT_GLOBAL_LAYER_ID_BASE,
    KVRegionExtractorV1,
    append_draft_page_table,
    is_draft_layer_group,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

TARGET_TPB = 128
DRAFT_TPB = 32


def _page_table(num_layers: int, tokens_per_block: int, base_address: int) -> KVCachePageTable:
    """One full-attention layer group of ``num_layers`` K/V layers in one pool."""
    entries = []
    for layer in range(num_layers):
        entries += [(layer, layer * 512, 256), (layer, layer * 512 + 256, 256)]
    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=2,
                local_layers=[
                    LocalLayer(local_layer_id=layer, global_layer_id=layer)
                    for layer in range(num_layers)
                ],
                pool_views=[
                    PoolView(pool_idx=0, buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE))
                ],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=base_address, slot_bytes=num_layers * 512, num_slots=16
                    )
                ]
            )
        ],
    )


@pytest.mark.cpu_only
def test_draft_groups_follow_the_target_groups_under_their_own_layer_ids():
    target = _page_table(num_layers=4, tokens_per_block=TARGET_TPB, base_address=0x1000)
    draft = _page_table(num_layers=2, tokens_per_block=DRAFT_TPB, base_address=0x9000)

    merged = append_draft_page_table(target, draft)

    assert merged.tokens_per_block == TARGET_TPB
    assert merged.layer_groups[0] is target.layer_groups[0]
    assert merged.pool_groups == target.pool_groups + draft.pool_groups
    draft_group = merged.layer_groups[1]
    assert draft_group.pool_group_idx == 1
    assert merged.pool_groups[draft_group.pool_group_idx].pools[0].base_address == 0x9000
    assert [ll.local_layer_id for ll in draft_group.local_layers] == [0, 1]
    assert [ll.global_layer_id for ll in draft_group.local_layers] == [
        DRAFT_GLOBAL_LAYER_ID_BASE,
        DRAFT_GLOBAL_LAYER_ID_BASE + 1,
    ]
    # The draft table itself is left as built.
    assert [ll.global_layer_id for ll in draft.layer_groups[0].local_layers] == [0, 1]
    # Peers see the draft groups as such after the wire round trip.
    received = KVCachePageTable.from_dict(merged.to_dict())
    assert [is_draft_layer_group(lg) for lg in received.layer_groups] == [False, True]


class _Adapter:
    """Reuse adapter double: fixed positional block tables and cached-prefix length."""

    def __init__(self, tokens_per_block, ordinals, cached_tokens=0):
        self.tokens_per_block = tokens_per_block
        self._ordinals = ordinals
        self._cached_tokens = cached_tokens
        self.cached_groups = None
        self.ordinal_groups = []

    def get_cached_token_count_per_layer_group(self, req, layer_groups):
        self.cached_groups = len(layer_groups)
        return [self._cached_tokens] * len(layer_groups)

    def get_block_ordinals(self, req, group_idx, lg):
        self.ordinal_groups.append(group_idx)
        return np.asarray(self._ordinals[group_idx], dtype=np.int64)


def _draft_transceiver(target_adapter, draft_adapter):
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._page_table = append_draft_page_table(
        _page_table(num_layers=4, tokens_per_block=TARGET_TPB, base_address=0x1000),
        _page_table(num_layers=2, tokens_per_block=DRAFT_TPB, base_address=0x9000),
    )
    transceiver._kv_cache_manager = SimpleNamespace(tokens_per_block=TARGET_TPB)
    transceiver._reuse_adapter = target_adapter
    transceiver._draft_reuse_adapter = draft_adapter
    transceiver._num_target_layer_groups = 1
    return transceiver


@pytest.mark.cpu_only
@pytest.mark.parametrize("generation_only", [False, True], ids=["context", "generation"])
def test_draft_blocks_are_read_from_the_draft_manager_in_its_own_block_size(generation_only):
    prompt_len = 100
    target = _Adapter(TARGET_TPB, {0: [5, 6]})
    # The draft pool reused one block on the generation side; a context side never masks.
    draft = _Adapter(DRAFT_TPB, {0: [7, 8, 9, 10, 11]}, cached_tokens=DRAFT_TPB)
    transceiver = _draft_transceiver(target, draft)
    req = SimpleNamespace(prompt_len=prompt_len, is_generation_only_request=generation_only)

    chunk = transceiver._create_chunk(req)

    target_blocks, draft_blocks = chunk.block_ids_per_layer_groups
    # ceil(100 / 128) target blocks; ceil(100 / 32) draft blocks, past the speculative tail.
    assert target_blocks.tolist() == [5]
    assert draft_blocks.tolist() == ([-1, 8, 9, 10] if generation_only else [7, 8, 9, 10])
    assert target.ordinal_groups == [0] and draft.ordinal_groups == [0]
    if generation_only:
        assert target.cached_groups == 1 and draft.cached_groups == 1
    assert chunk.token_range == TokenRange(start=0, end=prompt_len)


@pytest.mark.cpu_only
def test_pipelined_chunks_carry_the_same_token_span_in_draft_blocks():
    """Each slice sends the draft blocks its target span completes, so the slices tile both."""
    prompt_len = 500
    target_blocks = (prompt_len + TARGET_TPB - 1) // TARGET_TPB
    draft_blocks = (prompt_len + DRAFT_TPB - 1) // DRAFT_TPB
    whole = Chunk(
        block_ids_per_layer_groups=[
            np.arange(target_blocks, dtype=np.int64),
            np.arange(draft_blocks, dtype=np.int64) + 100,
        ],
        kind_per_layer_group=[CacheKind.PAGED, CacheKind.PAGED],
        token_range=TokenRange(start=0, end=prompt_len),
        is_last=True,
    )
    transceiver = _draft_transceiver(
        _Adapter(TARGET_TPB, {}), SimpleNamespace(tokens_per_block=DRAFT_TPB)
    )
    transceiver._describe_local = lambda req: whole
    transceiver._send_reqs = {}
    req = MagicMock()
    req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=42)
    req.py_beam_width = 1
    req.prompt_len = prompt_len
    req.prepopulated_prompt_len = 0

    sent = []
    for start, end in [(0, 100), (100, 300), (300, prompt_len)]:
        req.py_last_context_chunk = (start, end)
        req.context_remaining_length = prompt_len - end
        extent = KvCacheTransceiverV2._build_prefill_extent(transceiver, req)
        if extent is not None:
            sent.append(
                [ids[ids >= 0].tolist() for ids in extent.local.block_ids_per_layer_groups]
            )

    # The first chunk completes three draft blocks but no target block, so it waits.
    assert sent == [
        [[0, 1], list(range(100, 108))],
        [[2, 3], list(range(108, 116))],
    ]


def _manager(tokens_per_block: int, num_layers: int, is_draft: bool) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.HALF,
        is_draft=is_draft,
    )


def _request(request_id: int, prompt_len: int, context_only: bool) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if context_only
            else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        ),
    )


def _transceiver(page_table, target, draft):
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._page_table = page_table
    transceiver._kv_cache_manager = target
    transceiver._reuse_adapter = create_cache_reuse_adapter(target)
    transceiver._draft_reuse_adapter = create_cache_reuse_adapter(draft)
    transceiver._num_target_layer_groups = sum(
        not is_draft_layer_group(lg) for lg in page_table.layer_groups
    )
    return transceiver


def _bytes(pointer: int, count: int) -> torch.Tensor:
    return convert_to_torch_tensor(TensorWrapper(int(pointer), DataType.UINT8, [int(count)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_draft_pool_prompt_reaches_the_generation_draft_pool():
    """The sender's write list moves the prompt's draft pages, in the draft page size, into the
    pages the generation worker's draft mirror was admitted with."""
    prompt_len = 300
    device = torch.cuda.current_device()
    ctx_target = _manager(TARGET_TPB, num_layers=4, is_draft=False)
    ctx_draft = _manager(DRAFT_TPB, num_layers=2, is_draft=True)
    gen_target = _manager(TARGET_TPB, num_layers=4, is_draft=False)
    gen_draft = _manager(DRAFT_TPB, num_layers=2, is_draft=True)
    managers = (ctx_target, ctx_draft, gen_target, gen_draft)
    ctx_req, gen_req = _request(71, prompt_len, True), _request(72, prompt_len, False)
    try:
        allocate_request(ctx_target, ctx_req)
        allocate_request(ctx_draft, ctx_req)
        assert gen_draft.prepare_disagg_draft_gen_init(gen_req)
        assert gen_target.prepare_disagg_gen_init(gen_req)
        assert gen_draft.take_disagg_prompt_len(72) == prompt_len
        assert gen_draft.take_disagg_prompt_len(72) is None
        torch.cuda.synchronize()
        initialize_cache(ctx_target, seed=1)
        initialize_cache(ctx_draft, seed=2)
        for manager in (gen_target, gen_draft):
            initialize_cache(manager, seed=3)

        # The real wire serialization of both workers' merged page tables.
        ctx_info, gen_info = (
            RankInfo.from_bytes(
                RankInfo.from_kv_cache_manager(
                    name, target, device, draft_kv_cache_manager=draft
                ).to_bytes()
            )
            for name, target, draft in (
                ("context", ctx_target, ctx_draft),
                ("generation", gen_target, gen_draft),
            )
        )
        registrar = PeerRegistrar(ctx_info, KVRegionExtractorV1(ctx_info.page_table))
        registrar.register("generation", gen_info.instance_rank, gen_info)

        source = _transceiver(ctx_info.page_table, ctx_target, ctx_draft)._create_chunk(ctx_req)
        destination = _transceiver(gen_info.page_table, gen_target, gen_draft)._create_chunk(
            gen_req
        )
        sender = Sender.__new__(Sender)
        sender._registrar = registrar
        task = SimpleNamespace(
            _chunk=source, _prompt_len=prompt_len, _perf_timer=None, _unique_rid=991, slice_id=0
        )
        receiver = RecvReqInfo(
            sender_req_id=71,
            instance_name="generation",
            instance_rank=gen_info.instance_rank,
            block_ids_per_layer_groups=destination.block_ids_per_layer_groups,
            unique_rid=991,
        )
        write = sender._build_kv_write_meta(task, receiver)
        for src, dst, count in zip(write.src_ptrs, write.dst_ptrs, write.sizes):
            _bytes(dst, count).copy_(_bytes(src, count))
        torch.cuda.synchronize()

        for ctx_manager, gen_manager, tpb in (
            (ctx_target, gen_target, TARGET_TPB),
            (ctx_draft, gen_draft, DRAFT_TPB),
        ):
            prompt_blocks = (prompt_len + tpb - 1) // tpb
            torch.testing.assert_close(
                request_block_data(gen_manager, gen_req)[:prompt_blocks],
                request_block_data(ctx_manager, ctx_req)[:prompt_blocks],
                rtol=0,
                atol=0,
            )
    finally:
        for manager, req in zip(managers, (ctx_req, ctx_req, gen_req, gen_req)):
            manager.free_resources(req)
        for manager in managers:
            manager.shutdown()
