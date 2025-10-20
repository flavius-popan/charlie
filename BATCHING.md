# MLX Batching Implementation

## Overview

Batching implementation to improve throughput while respecting MLX thread-safety constraints.

**Problem:** MLX is not thread-safe, requiring serialization. Graphiti fires concurrent requests that were being processed one-at-a-time, wasting GPU capacity.

**Solution:** Request batching layer that collects concurrent requests and processes them together.

## Architecture

```
Graphiti fires 18 concurrent EntitySummary requests
    ↓
RequestBatcher (10ms collection window)
    ├─ Collects concurrent requests
    ├─ Waits batch_window for more requests
    └─ Processes batch together
    ↓
MLX_LOCK (single acquisition per batch)
    ↓
MLX operations (embeddings: true batching, LLM: serial in batch)
    ↓
Results distributed to individual futures
```

## Components

### 1. RequestBatcher (`app/llm/batcher.py`)

Generic batching coordinator:
- **Collects** requests arriving within `batch_window` (default 10ms)
- **Batches** up to `max_batch_size` requests (default 32)
- **Processes** via user-provided `batch_fn`
- **Distributes** results back to awaiting futures

### 2. MLXEmbedder Batching

**True batching** - single forward pass for multiple inputs:

```python
# Serial (before):
for text in texts:
    tokens = tokenize(text)           # N tokenizations
    embedding = model(tokens)         # N forward passes

# Batched (after):
batch_tokens = pad([tokenize(t) for t in texts])  # N tokenizations
batch_embeddings = model(batch_tokens)             # 1 forward pass
```

**Performance gain:** ~Nx speedup for N concurrent embeddings

### 3. GraphitiLM Batching

**Collection-based batching** - reduces lock overhead:

```python
# Serial (before):
for request in requests:
    async with MLX_LOCK:              # N lock acquisitions
        generate(request)

# Batched (after):
async with MLX_LOCK:                  # 1 lock acquisition
    for request in requests:
        generate(request)             # Still serial (Outlines limitation)
```

**Performance gain:** Reduced lock contention, better throughput

**Note:** Outlines doesn't support batched structured generation yet. When available, we can upgrade to true batched generation for additional speedup.

## Usage

### Enable Batching (Default)

```python
from app.llm.client import GraphitiLM
from app.llm.embedder import MLXEmbedder

llm_client = GraphitiLM(model, tokenizer, enable_batching=True)
embedder = MLXEmbedder(model, tokenizer, enable_batching=True)
```

### Disable Batching

```python
llm_client = GraphitiLM(model, tokenizer, enable_batching=False)
embedder = MLXEmbedder(model, tokenizer, enable_batching=False)
```

### Tuning Parameters

```python
from app.llm.batcher import RequestBatcher

# Custom batcher
llm_client.batcher = RequestBatcher(
    batch_fn=llm_client._generate_batch,
    batch_window=0.02,      # 20ms collection window (larger batches)
    max_batch_size=64,      # Allow bigger batches
)
```

## Performance Expectations

### Embeddings

**Test case:** 10 concurrent embedding requests

- **Before:** 10 × 50ms = 500ms total
- **After:** 1 × 80ms = 80ms total
- **Speedup:** ~6x

### LLM Generation

**Test case:** 18 concurrent EntitySummary requests (from logs)

- **Before:** 18 × 3s + lock overhead = ~55s
- **After:** 1 lock + (18 × 3s) = ~54s + better concurrency
- **Speedup:** Modest (10-20% from reduced lock contention)

**Future:** With batched Outlines generation → ~3-5x speedup possible

## Monitoring

Look for these log messages:

```
[Batcher] Processing batch of 10 requests
[MLXEmbedder] Batching 10 embeddings
[GraphitiLM] Processing batch of 18 generation requests
```

## Future Enhancements

### 1. True Batched LLM Generation

When Outlines supports batched structured generation:

```python
# Instead of:
for prompt in prompts:
    result = outlines_model(prompt, output_type=Schema)

# We can do:
results = outlines_model(prompts, output_type=Schema)  # Single batch
```

**Expected gain:** 3-5x speedup for concurrent generations

### 2. Adaptive Batch Windows

Dynamically adjust `batch_window` based on request arrival patterns:

- High concurrency → longer window (collect more)
- Low concurrency → shorter window (reduce latency)

### 3. Priority Batching

Process high-priority requests first while still batching:

```python
batcher.submit(prompt, priority=HIGH)
```

## Thread Safety

**MLX_LOCK remains critical:**
- Batching doesn't eliminate lock - it optimizes its usage
- Lock still acquired once per batch
- Prevents Metal command buffer crashes

**Async safety:**
- `asyncio.Lock` (not `threading.Lock`)
- No event loop blocking
- Futures coordinate result distribution

## Testing

### Test Batching Behavior

```python
import asyncio
from app.llm.embedder import MLXEmbedder

embedder = MLXEmbedder(model, tokenizer, enable_batching=True)

# Fire concurrent requests
tasks = [embedder.create(f"text {i}") for i in range(10)]
results = await asyncio.gather(*tasks)

# Check logs for: "[MLXEmbedder] Batching 10 embeddings"
```

### Benchmark

```bash
# Run with batching (default)
time python load_journals.py load --journal-file raw_data/2_days_journals.json --skip-verification

# Run without batching
# (modify code to set enable_batching=False)
time python load_journals.py load --journal-file raw_data/2_days_journals.json --skip-verification

# Compare times
```

## Design Decisions

### Why 10ms batch window?

Balance between:
- **Latency:** Longer window = more waiting
- **Batch size:** Longer window = more requests collected
- 10ms is imperceptible for most use cases

### Why max 32 batch size?

- Keeps memory usage reasonable
- Prevents one batch from monopolizing GPU
- Can be increased for high-throughput scenarios

### Why async.Lock not threading.Lock?

Async event loop compatibility - `threading.Lock` blocks the entire loop, causing deadlocks.

## Troubleshooting

### Batching not working

**Symptoms:** No "[Batcher] Processing batch" logs

**Causes:**
1. Batching disabled: Check `enable_batching=True`
2. Requests not concurrent: Graphiti may be serializing
3. Batch window too short: Requests don't overlap

### Deadlock

**Symptoms:** Hangs during processing

**Causes:**
1. Using `threading.Lock` instead of `asyncio.Lock`
2. Lock re-entrance (should be fixed with `_unlocked` methods)

### Poor performance

**Symptoms:** No speedup from batching

**Causes:**
1. Requests truly serial (no concurrency to batch)
2. Batch window too small (only 1 request per batch)
3. Overhead dominates (very fast operations)

## References

- MLX thread safety: https://github.com/ml-explore/mlx/issues/2133
- ParaLLM (batched MLX): https://github.com/willccbb/mlx_parallm
- Request batching pattern: Producer-consumer with futures
