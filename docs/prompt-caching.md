# Prompt Caching in dspy_outlines

## Overview

MLX prompt caching speeds up generation by reusing previously computed key-value (KV) states from the transformer layers. The cache is updated in-place during generation, accumulating conversation history or repeated prompt prefixes.

**Default**: Prompt caching is **disabled**. We observed degraded graph extraction quality when stale KV states leaked across requests, so new `OutlinesLM` instances start without a cache.

**Performance** (when enabled): ~11-15% speedup for prompts with shared prefixes (measured on Qwen3-4B-8bit).

## How It Works

Enable caching explicitly when you control the prompt lifecycle:

```python
from dspy_outlines import OutlinesLM

# Option 1: enable at construction time
lm = OutlinesLM(enable_prompt_cache=True)

# Option 2: lazily enable on an existing instance
lm = OutlinesLM()
lm.enable_prompt_cache()

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

## Preventing Cross-Request Contamination

### Problem: Stale KV States

Caching can bleed contextual hints across unrelated graph extraction calls. If you enable caching, plan to reset frequently.

### Solution: Reset Cache Periodically

```python
lm.enable_prompt_cache()  # ensures cache exists

# After N generations or when switching contexts:
lm.enable_prompt_cache(reset=True)  # rebuild fresh cache
```

Alternatively, call `lm.disable_prompt_cache()` to revert to uncached generation.

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
    lm.enable_prompt_cache(reset=True)
```

## Best Practices

**✅ DO:**
- Enable caching only when you manage request isolation or reuse identical prefixes
- Reset cache between independent conversations
- Monitor cache size in long-running applications

**❌ DON'T:**
- Leave caching on for user-facing workflows that send unrelated prompts
- Let cache grow unbounded in production (set a max token limit)
- Assume caching is beneficial without measuring latency and quality

## Thread Safety

The cache is accessed within `MLX_LOCK`, making it thread-safe when multiple threads share the same `OutlinesLM` instance. However, the cache will accumulate tokens from all threads' prompts, so reset more frequently in multi-threaded scenarios.

## Testing

See `tests/test_mlx.py` for cache verification tests:
- `test_prompt_cache_disabled_by_default`
- `test_prompt_cache_created_on_demand`
- `test_cache_size_helper`
- `test_prompt_caching_with_repeated_prefix`
- `test_cache_works_with_constrained_generation`
