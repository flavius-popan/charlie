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

## GEPA Scaling

GEPA runs ~128 full evals in "light" mode. Each full eval runs through all examples:
- 3 examples × 128 evals = 384 metric calls
- 30 examples × 128 evals = 3,840 metric calls

### Dynamic Threads (`get_num_threads`)

GEPA parallelizes metric calls within each eval batch, but can only parallelize up to N calls where N = number of examples. With 3 examples and 20 threads, 17 threads sit idle waiting.

`get_num_threads(num_examples, remote)` returns `min(num_examples, 20)` for remote, avoiding wasted thread overhead. More examples = better thread utilization = faster wall-clock time despite more total calls.

### Train/Val Split (`split_examples`)

GEPA uses trainset for learning and valset only for Pareto tracking (selecting best candidates). Large valsets waste metric calls without improving optimization.

`split_examples` keeps valset small (1-3 examples, capped) and puts everything else in trainset:
- 10 examples → 9 train, 1 val
- 30 examples → 27 train, 3 val
- 100 examples → 97 train, 3 val

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

## Run Log

| Date | Optimizer | Examples | Mode | Time | Baseline | Optimized | Notes |
|------|-----------|----------|------|------|----------|-----------|-------|
| 2025-11-25 | extract_nodes | 3 (2t/1v) | remote/BF16/L4 | 60m | 66.7% | 66.7% | FP8, pre-dynamic threads, 20 threads hardcoded |
