# Helix Parallelism

Helix is a context parallelism (CP) technique for the decode/generation phase of LLM inference. Unlike traditional attention-FFN disaggregation (AFD) techniques, which spatially separate attention and FFN blocks onto different GPUs, Helix temporally separates them by reconfiguring the same GPUs.

For all details, see the original paper:
[Helix Parallelism: Rethinking Sharding Strategies for
Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/pdf/2507.07120)

## How Helix Works

In Helix parallelism:

- **KV cache distribution**: The KV cache is partitioned across CP ranks in blocks of `tokens_per_block` tokens, round-robin: global block `b` lives on CP rank `b % context_parallel_size`
- **Attention computation**: Each rank computes partial attention over its local KV cache shard
- **Attention postprocessing**: Partial results are combined / corrected across ranks to produce the final attention output
- **FFN layers**: CP ranks are repurposed as tensor parallelism (TP) ranks for FFN/MoE layers, maximizing GPU utilization

Helix runs in two deployment modes:

- **Disaggregated serving**: the generation servers run Helix. A context server computes the prompt KV and the transceiver sends each CP rank the blocks it owns.
- **Aggregated serving**: the same CP group also runs the prefill. Every CP rank processes every prompt token and attends with its slice of the attention heads, as a tensor-parallel group of size `tensor_parallel_size * context_parallel_size` would, but stores only the KV blocks it owns. Prefill compute therefore matches a TP deployment of the same size, while the KV cache, and decode attention over it, stay sharded across the CP ranks. With chunked prefill, the KV of earlier chunks is gathered from its owning ranks when a later chunk attends to it.

## When to Use Helix

Helix parallelism provides performance benefits when the following conditions apply:

1. **Long input sequences**: Performance gains typically appear with input sequence lengths **>64K tokens** or more
2. **Low batch sizes**: Optimal for latency-sensitive workloads with high tokens/second/user requirements

Disaggregated serving lets the prefill run on a separately sized context server. Aggregated serving needs no second deployment: prefill runs with the same parallelism as a TP deployment of the whole CP group.

On a typical latency vs. throughput Pareto curve, Helix targets operating points toward the right side (low latency, high per-user throughput).

## Supported Models

Helix parallelism currently supports models using **Multi-head Latent Attention (MLA)** on Blackwell GPU architecture:

- DeepSeek-V3 / DeepSeek-V3-Lite

Aggregated serving additionally requires:

- Dense MLA on the `TRTLLM` attention backend (sparse MLA models such as DeepSeek-V3.2 run Helix only on disaggregated generation servers)
- KV cache manager V2 (`kv_cache_config.use_kv_cache_manager_v2=True`, the default)
- No attention data parallelism

Speculative decoding is supported in aggregated serving with one-engine MTP (the Eagle-style variant a single MTP layer resolves to, i.e. `use_mtp_vanilla=False`) and one-engine EAGLE3, with linear drafting. The target model verifies each draft group on the Helix-sharded KV cache, while the draft layers run as plain tensor parallelism over the CP group with their own unsharded KV cache. Options that turn speculation off mid-run (`acceptance_rate_window_size`, `acceptance_rate_threshold`, `max_concurrency`, `draft_len_schedule`) are not supported with Helix.

Block reuse is not supported with Helix. KV cache capacity is not profiled under Helix: the cache takes `kv_cache_config.free_gpu_memory_fraction` of the free memory after the model loads. In aggregated serving that remainder must also hold the prefill activations, so lower the fraction, or set `max_gpu_total_bytes`, if a large `max_num_tokens` runs out of memory.

## Configuration

### Configuration Parameters

Set the following parameters on the server that runs Helix: the generation servers in disaggregated mode, or the single server in aggregated mode. Examples can be seen in the e2e accuracy tests mentioned below.

| Parameter | Description | Required |
|-----------|-------------|----------|
| `context_parallel_size` | Number of GPUs for context parallelism (≥2 for Helix) | Yes |
| `cp_config.cp_type` | Must be `"HELIX"` or `CpType.HELIX` | Yes |
| `cp_config.tokens_per_block` | Tokens per KV cache block | Yes |
| `kv_cache_config.tokens_per_block` | Must match `cp_config.tokens_per_block` | Yes |

### JSON Configuration (for YAML/JSON configs)

```json
{
    "context_parallel_size": 2,
    "cp_config": {
        "cp_type": "HELIX",
        "tokens_per_block": 32
    },
    "kv_cache_config": {
        "tokens_per_block": 32
    }
}
```

## Testing Helix with TensorRT-LLM

### Unit Test: MLA Module Correctness

This developer test is not included in TensorRT-LLM release containers. Follow the
[build-from-source workflow](../installation/build-from-source.md), then install the test dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

The test requires at least two Blackwell GPUs; pytest skips it on unsupported hardware.

The simplest correctness test validates the [MLA attention module](../../../tensorrt_llm/_torch/attention/mla.py) with Helix enabled:

```bash
# Run the MLA Helix unit test
python3 -m pytest tests/unittest/_torch/attention/multi_gpu/test_mla_helix.py -v
```

This test verifies that attention outputs match between single-GPU and Helix-parallelized execution.

The Helix prefill used in aggregated serving has its own test, which compares the prefill output and the first decode step against a single-GPU reference:

```bash
python3 -m pytest tests/unittest/_torch/attention/multi_gpu/test_mla_helix_prefill.py -v
```

### End-to-End Accuracy test

For end-to-end validation, the accuracy benchmarks evaluate DeepSeek-V3-Lite on MMLU and GSM8K.

Disaggregated mode:

Test location: `tests/integration/defs/accuracy/test_disaggregated_serving.py`  
Test name: `TestDeepSeekV3Lite::test_auto_dtype_with_helix`

Aggregated mode, with and without MTP (requires four Blackwell GPUs):

Test location: `tests/integration/defs/accuracy/test_llm_api_pytorch.py`  
Test name: `TestDeepSeekV3Lite::test_bfloat16_helix`

### LLM API Example (aggregated)

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(
    "deepseek-ai/DeepSeek-V3-Lite",
    tensor_parallel_size=1,
    context_parallel_size=4,
    moe_expert_parallel_size=4,
    cp_config={"cp_type": "HELIX", "tokens_per_block": 32},
    kv_cache_config=KvCacheConfig(
        tokens_per_block=32,
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.5,
    ),
)
```
