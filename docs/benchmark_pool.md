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
from dspy_outlines import OutlinesLM, OutlinesAdapter
import time

def worker_process(prompts, results_queue):
    """Each process loads its own model."""
    lm = OutlinesLM()  # Separate 4GB model per process
    dspy.configure(lm=lm, adapter=OutlinesAdapter())
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

## Next Steps

1. Implement benchmark script
2. Run on hardware with 64GB RAM
3. Document results
4. If successful, add pool option to production code
