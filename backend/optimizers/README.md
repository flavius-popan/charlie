# Optimizers

## Remote Setup (HuggingFace Inference Endpoints)

Model: `unsloth/Qwen3-4B-Instruct-2507-FP8`

Container args:
```
--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.95 --enable-chunked-prefill --max-num-batched-tokens 16384 --enable-prefix-caching
```

Endpoint settings:
- Max Batched Tokens: 16384
- Max Sequences: 32-64
- Tensor Parallel Size: 1

Thread config (in __init__.py):
- L4 GPU: 10-15 threads (has ~21x concurrency limit)
- A10G: 20-30 threads
- L40s: 40-50 threads

L4 "not enough SMs for max_autotune_gemm" warning is expected and harmless.

## Quantization Divergence

Local uses Q4_K_M (~4.5 bit), remote uses FP8 (8 bit).

For entity extraction: low risk - structured output, clear task.
For complex reasoning: higher risk - prompts may not transfer.

Recommendation: optimize on remote FP8 for speed, validate final prompts on local Q4_K_M. If metrics drop >5%, optimize locally instead.

## Usage

```bash
python -m backend.optimizers.extract_nodes_optimizer --remote      # fast, HF endpoint
python -m backend.optimizers.extract_nodes_optimizer               # slow, local llama.cpp
python -m backend.optimizers.extract_nodes_optimizer --remote --no-cache
```
