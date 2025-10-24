# DSPy + Outlines + MLX Implementation Checkpoint

**Date:** 2025-10-23
**Status:** Tasks 1-3 complete, ready for Tasks 4-6

---

## Completed Work

### ✓ Task 1: DSPy LM Interface Research
- **File:** `research/dspy-lm-interface.md`
- **Key Finding:** Inherit from `dspy.BaseLM`, override `forward()` method, signature passed in kwargs

### ✓ Task 2: PassthroughLM (Proof of Concept)
- **Files:** `dspy_outlines/base_lm.py`, `tests/test_base_lm.py`
- **Status:** Test passing ✓
- **Proves:** DSPy call interception works

### ✓ Task 3: MLX Model Loading
- **Files:** `dspy_outlines/mlx_loader.py`, `tests/test_mlx_loader.py`
- **Status:** 2 tests passing ✓
- **API:** `outlines.from_mlxlm(mlx_model, mlx_tokenizer)`

---

## Next Steps (Tasks 4-6)

**For the next Claude agent:**

1. **Task 4:** Create `dspy_outlines/schema_extractor.py`
   - Extract Pydantic models from `signature.output_fields`
   - See plan for TDD test cases

2. **Task 5:** Implement `OutlinesDSPyLM` hybrid LM
   - Combine PassthroughLM pattern + MLX loader + schema extractor
   - Route DSPy calls through Outlines constrained generation
   - Return OpenAI-format responses

3. **Task 6:** Update `dspy-poc.py`
   - Replace `PassthroughLM` with `OutlinesDSPyLM`
   - Verify guaranteed valid JSON output
   - Should work without LM Studio running

**Full plan:** `plans/dspy-outlines-mlx.md`

---

## Architecture Overview

```
DSPy Signature
    ↓
OutlinesDSPyLM.__call__
    ↓
extract_output_schema(signature)  ← Task 4
    ↓
outlines_model(prompt, output_type=PydanticModel)  ← Tasks 3+5
    ↓
Return OpenAI format response
```

---

## Test Commands

```bash
# Verify current state
pytest tests/test_base_lm.py -v           # PassthroughLM
pytest tests/test_mlx_loader.py -v        # MLX loading (2 tests)

# All tests should pass
pytest tests/ -v
```

---

## Important Notes

1. **MLX_LOCK Required:** Use `asyncio.Lock()` for async to prevent segfaults (user note)
2. **Outlines API:** Use `outlines.from_mlxlm()` not `outlines.models.mlxlm()`
3. **Model Path:** `.models/mlx-community--Qwen3-4B-Instruct-2507-8bit`
4. **OpenAI Format:** Response must match OpenAI API structure (see `research/dspy-lm-interface.md`)

---

## Git Status

**Branch:** `dspy-poc`
**Recent Commits:**
- `514a4e8` docs: update plan with Tasks 1-3 completion status
- `0be7b19` feat: add MLX model loading via Outlines
- `820da40` feat: create PassthroughLM to intercept DSPy calls
- `7869598` docs: research DSPy LM interface for custom implementation

All changes committed and ready for next phase.
