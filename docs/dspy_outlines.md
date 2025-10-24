# DSPy + Outlines Integration: Architecture Fix Plan

**Status**: CRITICAL FLAW IDENTIFIED - FIX REQUIRED BEFORE PROCEEDING

## The Problem

**The implementation does not use Outlines' constrained generation.**

Current code in `hybrid_lm.py:76`:
```python
signature = kwargs.get('signature', None)  # Always returns None!
```

**Why**: DSPy's adapter architecture never passes signatures to LM instances. Adapters format prompts and parse responses themselves.

**Evidence**: Logging shows `Generating with schema: None` on every call.

**Impact**: Zero benefit from Outlines. Currently just unconstrained text → regex parsing (same as any LLM).

## DSPy Call Flow (How It Actually Works)

```
User → dspy.Predict → Adapter.format (uses signature)
                    → lm(messages=..., **lm_kwargs)  ← signature NOT here
                    → LM generates text
                    → Adapter.parse (extracts Pydantic)
                    → Return to user
```

Adapters own both prompt formatting AND response parsing. The LM only sees formatted text.

## The Fix: Custom Adapter (Option 1)

### Implementation Plan

**1. Create `dspy_outlines/adapter.py`:**

```python
"""Custom adapter that passes Pydantic schemas to OutlinesDSPyLM."""

from dspy.adapters import ChatAdapter
from .schema_extractor import extract_output_schema

class OutlinesAdapter(ChatAdapter):
    """
    Adapter that enables Outlines constrained generation by passing
    Pydantic schemas to the LM via lm_kwargs.
    """

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # Extract Pydantic schema from signature
        schema = extract_output_schema(signature)

        # Pass schema to LM via custom kwarg
        if schema:
            lm_kwargs['_outlines_schema'] = schema

        # Continue with normal adapter flow
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
```

**2. Update `OutlinesDSPyLM.forward()` in `hybrid_lm.py`:**

```python
def forward(self, prompt=None, messages=None, **kwargs):
    max_tokens = kwargs.get('max_tokens', 512)

    # Get schema from adapter (not signature - that's always None)
    schema = kwargs.pop('_outlines_schema', None)

    # Format prompt
    if messages:
        formatted_prompt = self._format_messages(messages)
    else:
        formatted_prompt = prompt

    logger.info(f"Generating with schema: {schema.__name__ if schema else 'None'}")

    # ACTUAL CONSTRAINED GENERATION
    if schema:
        result_json = self.outlines_model(
            formatted_prompt,
            output_type=schema,
            max_tokens=max_tokens
        )
        parsed = schema.model_validate_json(result_json)
        completion = parsed.model_dump_json()
    else:
        # Fallback for non-Pydantic outputs
        completion = self.outlines_model(formatted_prompt, max_tokens=max_tokens)

    # Store and return...
```

**3. Update `__init__.py` exports:**

```python
from .adapter import OutlinesAdapter
from .schema_extractor import extract_output_schema
from .hybrid_lm import OutlinesDSPyLM

__all__ = ["OutlinesDSPyLM", "OutlinesAdapter", "extract_output_schema"]
```

**4. Update usage pattern:**

```python
from dspy_outlines import OutlinesDSPyLM, OutlinesAdapter

lm = OutlinesDSPyLM()
adapter = OutlinesAdapter()

dspy.configure(lm=lm, adapter=adapter)  # Both required!
```

### Why This Works

- Adapter has access to signature (adapters always do)
- Adapter passes schema via `lm_kwargs` (LM always receives these)
- LM reads schema from kwargs and uses Outlines constraints
- Clean separation: adapter handles DSPy integration, LM handles generation

## Folder Structure Cleanup

**Current issues:**
- `base_lm.py` contains unused legacy code (PassthroughLM)
- `hybrid_lm.py` is unclear naming - should be `lm.py` (confirm this won't cause namespace clashes first!!)
- No adapter file yet

**Recommended structure:**

```
dspy_outlines/
  __init__.py          # Exports: OutlinesDSPyLM, OutlinesAdapter
  adapter.py           # OutlinesAdapter (NEW)
  lm.py                # OutlinesDSPyLM (rename from hybrid_lm.py)
  schema_extractor.py  # extract_output_schema (used by adapter)
  mlx_loader.py        # create_outlines_model (used by lm)
```

**Actions:**
1. DELETE `base_lm.py` and all references (unused legacy code)
2. RENAME `hybrid_lm.py` → `lm.py` after confirming safe namespacing
3. CREATE `adapter.py`
4. UPDATE `__init__.py` imports

## Implementation Checklist

- [ ] Create `dspy_outlines/adapter.py` with `OutlinesAdapter`
- [ ] Update `OutlinesDSPyLM.forward()` to read `_outlines_schema` from kwargs
- [ ] Rename `hybrid_lm.py` → `lm.py`
- [ ] Delete `base_lm.py`
- [ ] Update `__init__.py` exports
- [ ] Update test to use `OutlinesAdapter`
- [ ] Add validation test that verifies constrained generation works
- [ ] Update `CLAUDE.md` with new usage pattern
- [ ] Update `gradio_app.py` and `dspy-poc.py` to use adapter

## Validation Test (Critical!)

```python
def test_outlines_actually_constrains():
    """Verify Outlines enforces schema constraints."""

    class StrictCount(BaseModel):
        count: int  # MUST be integer, not string

    class CountSig(dspy.Signature):
        text: str = dspy.InputField()
        result: StrictCount = dspy.OutputField()

    lm = OutlinesDSPyLM()
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)

    predictor = dspy.Predict(CountSig)

    # Run multiple times - with constraints, ALWAYS valid
    for _ in range(10):
        result = predictor(text="How many people?")
        assert isinstance(result.result.count, int)
        # Should NEVER get "approximately 5", "around 3", etc.
```

## Additional Notes

### Multiple Output Fields
`schema_extractor.py` currently warns but uses first field when multiple Pydantic outputs exist. This is acceptable for now - handle edge case later if needed.

### MLX_LOCK
Currently defined but unused (`hybrid_lm.py:23`). **Keep it** - may be needed for actual Outlines generation, or when async support (`aforward()`) is implemented. Note for future:
```python
async def aforward(self, prompt=None, messages=None, **kwargs):
    async with MLX_LOCK:  # Prevent MLX segfaults
        # ... async generation
```

### Token Counting
Hardcoded to 0 because MLX doesn't expose token counts easily. Future improvement: implement proper tokenizer-based counting for cost tracking.

### Untested DSPy Features
Once fix is implemented, test compatibility with:
- Optimizers (MIPRO, BootstrapFewShot)
- Advanced modules (ReAct, ProgramOfThought)
- Async/streaming (after implementing)

## Success Criteria

✅ Logging shows `Generating with schema: KnowledgeGraph` (not None)
✅ Validation test confirms constraints are enforced
✅ Generated output matches schema 100% of the time
✅ Invalid structures are impossible (not just unlikely)

---

*Architecture review: 2025-10-24*
*Fix strategy: Custom Adapter (Option 1)*
*Ready for implementation*
