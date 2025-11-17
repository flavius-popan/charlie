# Model Pool Benchmarking

## Objective

Test and benchmark parallel inference using multiple MLX model instances across separate processes.

## Background

**Current pattern**: Single model (~4GB), all threads serialized via `MLX_LOCK`
**Alternative pattern**: Multiple processes, each with own model, true parallelism

With 64GB RAM, can run 3-4 models in parallel for ~3-4x throughput improvement.

## Implementation Plan

### 1. Create Benchmark Test (`tests/benchmark_model_pool.py`)

```python
import multiprocessing as mp
from mlx_runtime import MLXDspyLM
import time

def worker_process(prompts, results_queue):
    """Each process loads its own model."""
    lm = MLXDspyLM()  # Separate 4GB model per process
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    predictor = dspy.Predict(Signature)

    start = time.time()
    for prompt in prompts:
        result = predictor(question=prompt)
        results_queue.put(result)
    duration = time.time() - start
    return duration

def benchmark_parallel_pool(num_processes=3, prompts_per_process=10):
    """Run inference with N processes in parallel."""
    # Split work across processes
    # Measure total throughput
    pass

def benchmark_serial(total_prompts=30):
    """Baseline: single model, serialized."""
    # Measure throughput with one model
    pass
```

### 2. Metrics to Capture

- **Throughput**: prompts/second for serial vs parallel
- **Memory**: actual RAM usage with N processes
- **Latency**: per-prompt time (should be similar)
- **Speedup**: parallel_throughput / serial_throughput (target: ~3x with 3 processes)

### 3. Test Configurations

- 1 process (baseline)
- 2 processes
- 3 processes
- 4 processes

### 4. Expected Results

With sufficient RAM:
- **Serial (1 model)**: ~X prompts/sec, 4GB RAM
- **Parallel (3 models)**: ~3X prompts/sec, 12GB RAM
- **Parallel (4 models)**: ~4X prompts/sec, 16GB RAM

### 5. Potential Issues to Test

- Metal/GPU resource contention with >4 models
- Memory bandwidth limits
- Context switching overhead

### 6. Production Use Case

If benchmarks show good speedup, implement process pool for Gradio:

```python
# gradio_app.py with pool
pool = mp.Pool(processes=3, initializer=init_worker)

def extract_graph(text):
    # Submit to pool instead of calling directly
    return pool.apply_async(worker_extract, (text,)).get()
```

## Running the Benchmark

The benchmark is implemented in `benchmarks/benchmark_model_pool.py` as a standalone Python script.

### Quick sanity check (3 prompts, 1 process, ~1 min)
```bash
python benchmarks/benchmark_model_pool.py --sanity
```

### Full benchmark (serial + 2-3 parallel processes, ~10-15 min)
```bash
python benchmarks/benchmark_model_pool.py
```

### Configuration

Edit constants at the top of `benchmarks/benchmark_model_pool.py`:
- `MAX_PROCESSES = 3`: Maximum number of parallel processes (adjust for available RAM)
- `PROMPTS_PER_PROCESS = 5`: Number of prompts per process

### Expected Output

The benchmark will print:
- Serial baseline: throughput (prompts/sec), avg latency
- Parallel results for 2-N processes: throughput, speedup, efficiency
- Estimated RAM usage per configuration

### Observations

**Memory allocation pattern (macOS Activity Monitor)**:
- 2 workers: Memory usage shows "stacked" pattern (bumps overlay)
- 3 workers: Memory usage shows "spread" pattern (bumps are separate/distributed)

This suggests macOS may be distributing memory pressure differently with 3+ concurrent model loads. Worth monitoring for performance implications.

### Next Steps

1. ✅ Benchmark script implemented
2. Run on hardware with 64GB RAM to measure actual speedup
3. Document production results in this file
4. If successful (>2x speedup), add pool option to production code
