# Prompt Caching in dspy_outlines

## Overview

MLX prompt caching speeds up generation by reusing previously computed key-value (KV) states from the transformer layers. The cache is updated in-place during generation, accumulating conversation history or repeated prompt prefixes.

**Performance**: ~11-15% speedup for prompts with shared prefixes (measured on Qwen3-4B-8bit).

## How It Works

The cache is created once per `OutlinesLM` instance and passed to every generation:

```python
from dspy_outlines import OutlinesLM

lm = OutlinesLM()  # Cache created automatically
# All subsequent calls reuse and extend this cache
```

Cache behavior:
- **Starts empty** (offset=0)
- **Grows with each generation** (accumulates KV states)
- **Works for both constrained and unconstrained generation**
- **Shared across all calls** on the same `OutlinesLM` instance

Example cache growth:
```
Call 1: "System prompt... Q1" → cache: 0 → 40 tokens
Call 2: "System prompt... Q2" → cache: 40 → 80 tokens (reuses first 40)
Call 3: "System prompt... Q3" → cache: 80 → 120 tokens (reuses first 40)
```

## Preventing Performance Degradation

### Problem: Growing Cache Overhead

As the cache grows, MLX must process larger KV tensors, increasing memory usage and eventually slowing generation.

### Solution: Reset Cache Periodically

```python
from mlx_lm.models.cache import make_prompt_cache

lm = OutlinesLM()

# After N generations or when switching contexts:
lm.prompt_cache = make_prompt_cache(lm.raw_mlx_model)
```

### Recommended Reset Triggers

1. **Conversation boundaries**: Reset between independent user sessions
2. **Context switches**: Reset when changing tasks/topics completely
3. **Token limit**: Reset when cache exceeds ~2048 tokens (model-dependent)

### Monitoring Cache Size

```python
cache_size = lm._get_cache_size()  # Returns number of cached tokens
print(f"Cache: {cache_size} tokens")

# Reset if too large
if cache_size > 2048:
    lm.prompt_cache = make_prompt_cache(lm.raw_mlx_model)
```

## Best Practices

**✅ DO:**
- Use caching for repeated prompt prefixes (system prompts, few-shot examples)
- Reset cache between independent conversations
- Monitor cache size in long-running applications

**❌ DON'T:**
- Share one `OutlinesLM` instance across completely different tasks without resetting
- Let cache grow unbounded in production (set a max token limit)
- Create new `OutlinesLM` instances per request (defeats caching, wastes 4GB RAM each)

## Thread Safety

The cache is accessed within `MLX_LOCK`, making it thread-safe when multiple threads share the same `OutlinesLM` instance. However, the cache will accumulate tokens from all threads' prompts, so reset more frequently in multi-threaded scenarios.

## Testing

See `tests/test_mlx.py` for cache verification tests:
- `test_prompt_cache_created`
- `test_cache_size_helper`
- `test_prompt_caching_with_repeated_prefix`
- `test_cache_works_with_constrained_generation`
