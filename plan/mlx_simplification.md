# MLX-LM Simplification Plan

## Executive Summary

**Objective: Simplify codebase by removing custom batching, keeping only essential MLX_LOCK for thread safety**

The current implementation has batching complexity that adds maintenance overhead. Since the prompt layer will be redone with DSPy soon, we should simplify now by:
1. Removing all custom batching logic
2. Keeping MLX_LOCK (prevents Metal segfaults)
3. Testing Qwen-specific configurations for correctness

---

## Phase 1: Remove Custom Batching System

### Current Architecture to Remove

**Files to modify/remove:**
- `app/llm/batcher.py` - Delete entire file
- `app/llm/client.py` - Remove batching integration
- `app/llm/embedder.py` - Remove batch processing, keep simple serial

**Current complexity:**
```python
# In GraphitiLM.__init__:
self.batcher = RequestBatcher(
    batch_fn=self._generate_batch,
    batch_window=0.01,
    max_batch_size=32,
)

# In _direct_generate:
if self.batcher:
    return await self.batcher.submit(messages, response_model, max_tokens)
```

**Simplified version:**
```python
# Just direct generation with MLX_LOCK
async with MLX_LOCK:
    return await self._generate_single(messages, response_model, max_tokens)
```

### Changes Required

#### 1. `app/llm/client.py`

**Remove:**
- Import of `RequestBatcher` from batcher
- `enable_batching` parameter from `__init__`
- `self.batcher` initialization
- `_generate_batch()` method
- Batching logic in `_direct_generate()`

**Keep:**
- `MLX_LOCK` (critical for Metal thread safety)
- `_generate_single()` method
- All core generation logic

**Simplified flow:**
```python
async def _direct_generate(
    self,
    messages: list[Message],
    response_model: type[BaseModel],
    max_tokens: int
) -> dict[str, Any]:
    """Direct generation via Outlines (no batching)."""
    if response_model is None:
        raise ValueError("response_model is required for structured generation")

    # Simple serial processing with MLX lock
    async with MLX_LOCK:
        return await self._generate_single_unlocked(messages, response_model, max_tokens)
```

#### 2. `app/llm/embedder.py`

**Remove:**
- Batch embedding methods
- Complex padding and attention mask logic
- Batching parameters

**Keep:**
- Simple serial embedding with MLX_LOCK
- Last-token pooling
- L2 normalization
- MRL truncation

**Simplified embedding:**
```python
async def embed(self, text: str) -> list[float]:
    """Generate embedding for single text (no batching)."""
    async with MLX_LOCK:
        return await asyncio.to_thread(self._embed_sync, text)

def _embed_sync(self, text: str) -> list[float]:
    """Synchronous embedding generation."""
    # Tokenize
    tokens = self.tokenizer.encode(text)

    # Forward pass
    outputs = self.model(mx.array([tokens]))

    # Last token pooling
    embedding = outputs[0, -1, :]

    # L2 normalize
    embedding = embedding / mx.linalg.norm(embedding)

    # MRL truncate if needed
    if embedding.shape[0] > self.target_dim:
        embedding = embedding[:self.target_dim]

    return embedding.tolist()
```

#### 3. Delete `app/llm/batcher.py`

Remove entire file - no longer needed.

#### 4. Update initialization in `load_journals.py` and other entry points

**Remove:**
```python
llm_client = GraphitiLM(
    outlines_model,
    mlx_tokenizer,
    config=llm_config,
    enable_batching=True  # Remove this parameter
)
```

**Simplified:**
```python
llm_client = GraphitiLM(
    outlines_model,
    mlx_tokenizer,
    config=llm_config
)
```

### Why This Simplification?

**Before (complex):**
- Custom batching system with request queuing
- Batch window timing logic
- Request collection and distribution
- Still processes serially due to Outlines limitation
- ~200 lines of batching code

**After (simple):**
- Direct serial processing
- MLX_LOCK for thread safety
- ~50 lines of generation code
- Same actual behavior (Outlines requires serial processing anyway)

**Key insight:** The batching system was collecting concurrent requests and processing them serially under one lock acquisition. This added complexity without true parallelism since Outlines doesn't support batched structured generation.

---

## Phase 2: Test Qwen Configuration

### Objective
Verify if explicit Qwen-specific settings improve tokenization/generation quality.

### Create Test Script

**File:** `test_qwen_config.py`

```python
"""
Test Qwen model configuration with/without explicit settings.
"""

import mlx_lm
from app.settings import settings

def test_default_config():
    """Test current behavior (relying on model defaults)."""
    print("Testing DEFAULT configuration...")
    model, tokenizer = mlx_lm.load(settings.MLX_MODEL_NAME)

    # Test chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Chat template output:\n{prompt}\n")

    # Test generation
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=50,
        verbose=True
    )
    print(f"Generation: {response}\n")

    # Check EOS token
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    return model, tokenizer, response

def test_explicit_config():
    """Test with explicit Qwen settings."""
    print("\nTesting EXPLICIT Qwen configuration...")
    model, tokenizer = mlx_lm.load(
        settings.MLX_MODEL_NAME,
        tokenizer_config={
            "eos_token": "<|endoftext|>",
            "trust_remote_code": True
        }
    )

    # Same tests as above
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Chat template output:\n{prompt}\n")

    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=50,
        verbose=True
    )
    print(f"Generation: {response}\n")

    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    return model, tokenizer, response

def compare_tokenization():
    """Compare tokenization with both configs."""
    test_texts = [
        "Hello, how are you?",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms."
    ]

    print("\n=== Tokenization Comparison ===\n")

    # Default config
    _, default_tokenizer, _ = test_default_config()

    # Explicit config
    _, explicit_tokenizer, _ = test_explicit_config()

    for text in test_texts:
        default_tokens = default_tokenizer.encode(text)
        explicit_tokens = explicit_tokenizer.encode(text)

        print(f"Text: {text}")
        print(f"Default tokens: {len(default_tokens)} - {default_tokens[:10]}...")
        print(f"Explicit tokens: {len(explicit_tokens)} - {explicit_tokens[:10]}...")
        print(f"Match: {default_tokens == explicit_tokens}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen Configuration Test")
    print("=" * 60)

    # Run all tests
    test_default_config()
    test_explicit_config()
    compare_tokenization()

    print("\n" + "=" * 60)
    print("Test complete. Review outputs above.")
    print("=" * 60)
```

### Test Cases

1. **Chat Template Application**
   - Does the template format correctly?
   - Are system/user roles handled properly?

2. **EOS Token Handling**
   - Is EOS token set correctly?
   - Does generation stop appropriately?

3. **Tokenization Accuracy**
   - Do token counts match expectations?
   - Are there encoding/decoding issues?

4. **Generation Quality**
   - Does output quality differ?
   - Are there truncation or repetition issues?

### Decision Criteria

**If explicit config improves behavior:**
```python
# Update all mlx_lm.load() calls to include:
mlx_model, mlx_tokenizer = mlx_lm.load(
    settings.MLX_MODEL_NAME,
    tokenizer_config={
        "eos_token": "<|endoftext|>",
        "trust_remote_code": True
    }
)
```

**If no differences observed:**
- Document that Qwen3 model includes correct defaults
- No code changes needed
- Update comments to note testing was done

### Files to Update if Config Needed

- `load_journals.py` (2 `mlx_lm.load()` calls)
- `test_embedding_size.py` (1 call)
- `test_mlx_extraction.py` (1 call)
- Any other files with `mlx_lm.load()`

---

## Phase 3: Testing & Validation

### After Batching Removal

**Verify:**
1. All generation still works correctly
2. MLX_LOCK prevents segfaults
3. Embeddings work without batch processing
4. No performance regression for single requests
5. Codebase is simpler and easier to understand

**Test commands:**
```bash
# Test entity extraction
python load_journals.py

# Test embeddings
python test_embedding_size.py

# Check for any batching artifacts
grep -r "batch" app/llm/
```

### After Qwen Config Testing

**Verify:**
1. Tokenization is correct
2. EOS handling works properly
3. Generation quality is maintained or improved
4. No warnings about trust_remote_code

---

## Implementation Checklist

### Phase 1: Remove Batching (1-2 hours)

- [ ] Remove `RequestBatcher` import from `app/llm/client.py`
- [ ] Remove `enable_batching` parameter from `GraphitiLM.__init__`
- [ ] Remove `self.batcher` initialization
- [ ] Remove `_generate_batch()` method
- [ ] Simplify `_direct_generate()` to direct serial processing
- [ ] Remove `_generate_single()` wrapper (merge with `_generate_single_unlocked`)
- [ ] Simplify `app/llm/embedder.py` to serial-only processing
- [ ] Remove batch embedding methods
- [ ] Delete `app/llm/batcher.py`
- [ ] Update all initialization code to remove `enable_batching` parameter
- [ ] Run tests to verify everything still works
- [ ] Clean up any unused imports

### Phase 2: Qwen Testing (1 hour)

- [ ] Create `test_qwen_config.py` script
- [ ] Run test with default configuration
- [ ] Run test with explicit Qwen configuration
- [ ] Compare results
- [ ] Make decision on whether explicit config is needed
- [ ] If needed, update all `mlx_lm.load()` calls
- [ ] Document findings in comments or README

### Phase 3: Final Validation (30 minutes)

- [ ] Test entity extraction with sample data
- [ ] Verify embeddings work correctly
- [ ] Check for any "batch" references in code
- [ ] Verify MLX_LOCK is still in place
- [ ] Update documentation if needed

---

## Expected Outcomes

### Code Simplification
- **Before:** ~400 lines of batching/generation code
- **After:** ~150 lines of simple serial generation code
- **Removed:** ~250 lines of batching complexity

### Maintainability
✅ Simpler codebase for DSPy migration
✅ Easier to understand and debug
✅ Less code to maintain
✅ No false promise of parallelism

### Performance
⚠️ No change for single requests (same as before)
⚠️ No batching overhead for concurrent requests
✅ MLX_LOCK still prevents segfaults
✅ Predictable serial behavior

### Correctness
✅ Qwen configuration tested and validated
✅ Generation quality maintained or improved
✅ Thread safety preserved with MLX_LOCK

---

## Timeline

| Phase | Duration | Priority |
|-------|----------|----------|
| 1. Remove Batching | 1-2 hours | High |
| 2. Qwen Testing | 1 hour | Medium |
| 3. Final Validation | 30 min | High |

**Total estimated effort:** 2.5-3.5 hours

---

## Notes

### Why Remove Batching?

The current batching system adds complexity without true parallelism:
- Outlines doesn't support batched structured generation
- Requests are collected but processed serially anyway
- MLX_LOCK serializes everything regardless
- Adds ~250 lines of code for minimal benefit
- DSPy integration coming soon will change prompt layer entirely

**Better to simplify now than carry complexity forward.**

### What We're Keeping

- **MLX_LOCK**: Essential for Metal thread safety, prevents segfaults
- **Serial processing**: Honest about what's actually happening
- **Outlines integration**: Works well for structured generation
- **Core generation logic**: No changes to actual model inference

### Future Optimization Path

After DSPy integration:
1. Evaluate if DSPy provides better batching support
2. Consider MLX-LM's native `batch_generate` if DSPy allows
3. Implement optimizations based on actual DSPy architecture
4. Re-evaluate batching only if it provides real parallelism
