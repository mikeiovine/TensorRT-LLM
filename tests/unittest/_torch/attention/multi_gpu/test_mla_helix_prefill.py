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
"""Helix prefill (aggregated serving) for MLA against a single-GPU reference.

Every CP rank runs the whole prompt, attends with its slice of the heads and
stores only the KV blocks it owns. The prefill output must match the
reference, and the first helix decode step, which reads only the rank-local
blocks the prefill stored, must match the reference decode.
"""

import pickle
import sys
import weakref
from dataclasses import dataclass
from typing import Optional

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import _torch.attention.multi_gpu.test_mla_helix as mla_helix
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import (
    CACHE_TYPE_SELFKONLY,
    compute_mismatch_ratio,
    copy_weights_for_cp,
    run_helix_test,
)
from _torch.attention.multi_gpu.test_mla_helix import RopeConfig, Scenario, _generate_random_weights
from mpi4py import MPI
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    KVCacheParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.utils import helix_local_len, model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
cloudpickle.register_pickle_by_value(mla_helix)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@dataclass(kw_only=True, frozen=True)
class PrefillScenario:
    base: Scenario
    # Prefill chunk length; None runs the prompt in one context step.
    chunk_len: Optional[int] = None


test_scenarios = [
    PrefillScenario(base=Scenario(batch=1, ctx_len=64)),
    # Fewer blocks than CP ranks: the last rank stores nothing for this prompt.
    PrefillScenario(base=Scenario(batch=1, ctx_len=20)),
    # A partial last block and several requests per batch.
    PrefillScenario(base=Scenario(batch=4, ctx_len=1000)),
    PrefillScenario(base=Scenario(batch=2, ctx_len=4096)),
    # Chunked prefill reads earlier chunks back from the distributed cache.
    PrefillScenario(base=Scenario(batch=2, ctx_len=1024), chunk_len=256),
    PrefillScenario(base=Scenario(batch=1, ctx_len=2000), chunk_len=512),
]


def _make_consistent_absorption_weights(mla: MLA) -> None:
    """Derive k_b_proj_trans / v_b_proj from kv_b_proj, as a checkpoint does.

    Helix prefill expands latent KV with the absorption weights while the
    reference prefill uses kv_b_proj, so the two must describe the same model.
    """
    num_heads = mla.num_heads_tp
    weight = mla.kv_b_proj.weight
    k_nope = weight[: num_heads * mla.qk_nope_head_dim].view(
        num_heads, mla.qk_nope_head_dim, mla.kv_lora_rank
    )
    v = weight[num_heads * mla.qk_nope_head_dim :].view(
        num_heads, mla.v_head_dim, mla.kv_lora_rank
    )
    mla.k_b_proj_trans.data.copy_(k_nope.transpose(1, 2))
    mla.v_b_proj.data.copy_(v)


def _make_mla(scenario: Scenario, pos_embd_params, mapping: Optional[Mapping] = None) -> MLA:
    kwargs = {}
    if mapping is not None:
        kwargs["config"] = ModelConfig(mapping=mapping)
    return MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        **kwargs,
    ).cuda()


def _make_kv_cache_manager(scenario: Scenario, mapping: Mapping) -> KVCacheManagerV2:
    """A cache holding each prompt plus one generated token.

    Under helix the V2 ledger counts global tokens, one ledger block per
    tokens_per_block * cp_size, so the sizes below are global.
    """
    ledger_block = scenario.kv_cache_tokens_per_block * mapping.cp_size
    seq_len = scenario.ctx_len + 1
    ledger_tokens = (seq_len + ledger_block - 1) // ledger_block * ledger_block
    kv_cache_manager = KVCacheManagerV2(
        KvCacheConfig(
            # One spare ledger block for the helix prefill trash page.
            max_tokens=(ledger_tokens * scenario.batch + ledger_block),
            enable_block_reuse=False,
            host_cache_size=0,
        ),
        CACHE_TYPE_SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.kv_cache_tokens_per_block,
        max_seq_len=seq_len,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    requests = kv_cache_manager.add_dummy_requests(
        list(range(scenario.batch)), [seq_len] * scenario.batch
    )
    assert requests is not None
    return kv_cache_manager


def _context_metadata(
    scenario: Scenario,
    kv_cache_manager,
    mapping: Mapping,
    start: int,
    length: int,
    chunked: bool,
):
    """Metadata for one context step over prompt positions [start, start + length)."""
    runtime_features = None
    if chunked:
        runtime_features = AttentionRuntimeFeatures(
            chunked_prefill=True, chunk_size=length, chunked_prefill_buffer_batch_size=1
        )
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([length] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[length] * scenario.batch,
        max_num_tokens=length * scenario.batch,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[start] * scenario.batch,
        ),
        mapping=mapping,
        enable_context_mla_with_cached_kv=chunked,
        runtime_features=runtime_features,
    )
    attn_metadata.prepare()
    return attn_metadata


def _run_prefill(
    mla: MLA,
    scenario: Scenario,
    chunk_len: Optional[int],
    kv_cache_manager,
    mapping: Mapping,
    input_ctx: torch.Tensor,
) -> torch.Tensor:
    """Run the prompt through MLA in chunks; return outputs in input order."""
    batch, ctx_len = scenario.batch, scenario.ctx_len
    chunk_len = chunk_len or ctx_len
    input_bs = input_ctx.view(batch, ctx_len, scenario.hidden_size)
    output_bs = input_bs.new_empty(batch, ctx_len, scenario.hidden_size)
    for start in range(0, ctx_len, chunk_len):
        end = min(start + chunk_len, ctx_len)
        attn_metadata = _context_metadata(
            scenario, kv_cache_manager, mapping, start, end - start, chunked=chunk_len < ctx_len
        )
        hidden = input_bs[:, start:end].reshape(-1, scenario.hidden_size).contiguous()
        position_ids = torch.arange(start, end, dtype=torch.int, device="cuda").repeat(batch)
        with model_extra_attrs({"attention_metadata": weakref.ref(attn_metadata)}):
            output = mla(position_ids, hidden, attn_metadata)
        output_bs[:, start:end] = output.view(batch, end - start, scenario.hidden_size)
    return output_bs.view(-1, scenario.hidden_size)


def _reference_generation_metadata(scenario: Scenario, kv_cache_manager):
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=0,
        prompt_lens=[scenario.ctx_len] * scenario.batch,
        max_num_tokens=scenario.ctx_len,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[scenario.ctx_len] * scenario.batch,
        ),
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata.prepare()
    return attn_metadata


def _helix_generation_metadata(
    scenario: Scenario, kv_cache_manager, local_cached: int, is_inactive_rank: bool
):
    """One decode step at global position ctx_len over this rank's local KV."""
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=0,
        prompt_lens=[local_cached] * scenario.batch,
        max_num_tokens=scenario.ctx_len,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[local_cached] * scenario.batch,
        ),
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata.enable_helix = True
    attn_metadata.helix_is_inactive_rank = torch.tensor(
        [is_inactive_rank] * scenario.batch, dtype=torch.bool, device="cuda"
    )
    attn_metadata.helix_is_inactive_rank_cpu = attn_metadata.helix_is_inactive_rank.cpu()
    attn_metadata.helix_position_offsets = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )
    attn_metadata.helix_position_offsets_cpu = attn_metadata.helix_position_offsets.cpu()
    attn_metadata.prepare()
    return attn_metadata


def _broadcast_from_rank0(tensor: torch.Tensor, mapping: Mapping) -> torch.Tensor:
    gathered = cp_allgather(tensor, mapping=mapping, dim=0)
    return gathered.view(mapping.cp_size, *tensor.shape)[0]


@torch.inference_mode
def _full_test_multi_gpu(
    rank: int,
    world_size: int,
    prefill_scenario: PrefillScenario,
    use_nccl_for_alltoall: bool = False,
    fifo_version: int = 2,
):
    scenario = prefill_scenario.base
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=None,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    torch.manual_seed(42)
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    input_gen = torch.empty(
        scenario.batch, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )

    mla = _make_mla(scenario, pos_embd_params)
    _generate_random_weights(mla)
    _make_consistent_absorption_weights(mla)
    weights = mla.state_dict()

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )

    # Reference: one GPU, whole prompt in one context step, then one decode step.
    ref_ctx = input_ctx.new_empty(input_ctx.shape)
    ref_gen = input_gen.new_empty(input_gen.shape)
    if rank == 0:
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager = _make_kv_cache_manager(scenario, ref_mapping)
        ref_ctx = _run_prefill(mla, scenario, None, ref_kv_cache_manager, ref_mapping, input_ctx)
        ref_gen_metadata = _reference_generation_metadata(scenario, ref_kv_cache_manager)
        with model_extra_attrs({"attention_metadata": weakref.ref(ref_gen_metadata)}):
            ref_gen = mla(position_ids_gen, input_gen, ref_gen_metadata)
        ref_kv_cache_manager.shutdown()
    ref_ctx = _broadcast_from_rank0(ref_ctx, mapping)
    ref_gen = _broadcast_from_rank0(ref_gen, mapping)

    # Helix: every rank runs the whole prompt.
    helix_mla = _make_mla(scenario, pos_embd_params, mapping)
    copy_weights_for_cp(weights, "o_proj.weight", 1, rank, world_size)
    copy_weights_for_cp(weights, "v_b_proj", 0, rank, world_size)
    helix_mla.load_state_dict(weights)
    kv_cache_manager = _make_kv_cache_manager(scenario, mapping)
    assert kv_cache_manager.supports_helix_prefill
    ctx_output = _run_prefill(
        helix_mla, scenario, prefill_scenario.chunk_len, kv_cache_manager, mapping, input_ctx
    )
    ctx_mismatch = compute_mismatch_ratio(
        ctx_output, ref_ctx, scenario.atol, scenario.rtol, rank, world_size
    )

    # First decode step over the rank-local blocks the prefill stored.
    tokens_per_block = scenario.kv_cache_tokens_per_block
    local_cached = helix_local_len(scenario.ctx_len, tokens_per_block, world_size, rank)
    owner = (scenario.ctx_len // tokens_per_block) % world_size
    gen_metadata = _helix_generation_metadata(
        scenario, kv_cache_manager, local_cached, is_inactive_rank=rank != owner
    )
    with model_extra_attrs({"attention_metadata": weakref.ref(gen_metadata)}):
        gen_output = helix_mla(position_ids_gen, input_gen, gen_metadata)
    gen_mismatch = compute_mismatch_ratio(
        gen_output, ref_gen, scenario.atol, scenario.rtol, rank, world_size
    )
    kv_cache_manager.shutdown()
    return max(ctx_mismatch, gen_mismatch)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "prefill_scenario",
    test_scenarios,
    ids=lambda x: f"batch{x.base.batch}-ctx{x.base.ctx_len}-chunk{x.chunk_len}",
)
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v2"])
def test_mla_helix_prefill_distributed(prefill_scenario: PrefillScenario, comms_medium: str):
    run_helix_test(_full_test_multi_gpu, prefill_scenario, comms_medium)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "prefill_scenario",
    [test_scenarios[2], test_scenarios[4]],
    ids=lambda x: f"batch{x.base.batch}-ctx{x.base.ctx_len}-chunk{x.chunk_len}",
)
def test_mla_helix_prefill_distributed_cp4(prefill_scenario: PrefillScenario):
    run_helix_test(_full_test_multi_gpu, prefill_scenario, "fifo_v2", world_size=4)

