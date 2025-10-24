# Fallback Extraction Architecture

**Date:** 2025-10-24
**Status:** Design Phase
**Goal:** Enable fast-path JSON extraction with graceful fallback to Outlines constrained generation

## Problem Statement

Outlines constrained generation provides guaranteed valid JSON but incurs a 7x slowdown (measured end-to-end) compared to unconstrained generation. For small LLMs (Qwen3-4B) running locally on end-user hardware, we need:

1. **Fast path** when the model produces valid JSON naturally
2. **Guaranteed fallback** when JSON parsing fails
3. **Visibility** into failure rates to inform model scaling decisions

## Architecture Overview

### Three-Tier Fallback Chain

```
dspy.Predict(Signature)
    ↓
OutlinesAdapter.__call__()
    ↓
Tier 1: ChatAdapter format (field markers)
    ↓ [on exception]
Tier 2: JSON format (unconstrained, fast)
    ↓ [on AdapterParseError]
Tier 3: Outlines constrained generation (slow, guaranteed)
    ↓
OutlinesLM
```

### Key Components

**OutlinesAdapter** - Single adapter extending `ChatAdapter`:
- Implements all three fallback tiers internally
- Tier 1: Delegates to `super().__call__()` (ChatAdapter's field-marker format)
- Tier 2: `_json_fallback()` - JSON prompting without constraints
- Tier 3: `_constrained_fallback()` - Adds `_outlines_constraint` for guaranteed valid output
- Tracks metrics at each tier for experimentation

**OutlinesLM** - DSPy language model:
- Receives `_outlines_constraint` kwarg from adapter
- Passes constraint to `self.model(prompt, output_type=constraint)`
- Exposes `lm.model` (the Outlines MLXLM instance) for direct Outlines usage

## Design Decisions

### 1. Constraint Type Agnostic

**Decision:** Use `_outlines_constraint` (not `_outlines_schema`) to support all Outlines constraint types.

**Rationale:**
- Outlines supports: Pydantic models, Literal (multiple choice), Regex, CFG, basic types
- Outlines' `python_types_to_terms()` (.venv/.../outlines/types/dsl.py:707) handles type detection
- Pass raw annotation from DSPy signature → Outlines handles it

**Implementation reference:**
- `.venv/.../outlines/types/dsl.py:707-798` - `python_types_to_terms()` converts Python types to Terms
- Line 780-781: `is_literal(ptype)` → calls `_handle_literal()` (line 814) → `Alternatives`
- Line 763-765: Pydantic/dataclass/TypedDict → `JsonSchema`
- `.venv/.../outlines/generator.py:237` - `SteerableGenerator.__init__` shows how `output_type` is processed
- Line 245-249: JsonSchema → `get_json_schema_logits_processor`
- Line 252-256: Otherwise regex → `get_regex_logits_processor`

**Example:**
```python
# Pydantic model
sentiment: SentimentModel  # → JsonSchema

# Multiple choice
sentiment: Literal['positive', 'negative', 'neutral']  # → Alternatives

# Both work with the same interface
lm_kwargs['_outlines_constraint'] = output_field.annotation
```

### 2. Single Adapter (Not Multiple)

**Decision:** One `OutlinesAdapter` containing all fallback logic.

**Rationale:**
- Simpler mental model - one import, one configuration
- All metrics in one place
- Easier to maintain fallback coordination
- Follows DSPy's existing pattern (ChatAdapter already has fallback logic)

**DSPy precedent:**
- `.venv/.../dspy/adapters/chat_adapter.py:37-47` - ChatAdapter's `__call__` catches exceptions and falls back to JSONAdapter
- Pattern: `try/except` with `isinstance(e, ContextWindowExceededError)` check to prevent re-raising certain errors
- Line 43: `isinstance(self, JSONAdapter)` prevents infinite recursion
- Our adapter extends this pattern to add third tier

### 3. Expose Outlines Model Directly

**Decision:** `OutlinesLM.model` provides direct access to underlying Outlines MLXLM instance.

**Rationale:**
- Enables future non-Pydantic constraints without DSPy overhead (multiple choice, regex)
- Single model instance in memory (no double-loading)
- Follows Outlines conventions: `model(prompt, output_type)`

**Outlines interface reference:**
- `.venv/.../outlines/models/base.py:80-122` - `Model.__call__` signature
- Line 80-86: `def __call__(self, model_input, output_type=None, backend=None, **inference_kwargs)`
- Line 122: Creates Generator and calls it
- `.venv/.../outlines/models/mlxlm.py:118-150` - `MLXLM.generate()` is what actually runs
- Line 144-149: Calls `mlx_lm.generate()` with formatted input and logits processors

**Current OutlinesLM state:**
- `dspy_outlines/lm.py:67` - Creates `self.outlines_model` via `create_outlines_model()`
- `dspy_outlines/mlx_loader.py:48` - Returns `outlines.from_mlxlm(mlx_model, tokenizer)`
- This returns an `MLXLM` instance which is already callable

**Rename:**
- Change `self.outlines_model` → `self.model`
- Matches Outlines documentation conventions
- Users call `lm.model(prompt, Customer)` just like Outlines examples

**Usage:**
```python
lm = OutlinesLM()

# DSPy integration
dspy.configure(lm=lm, adapter=OutlinesAdapter())

# Direct Outlines (future: multiple choice, regex)
result = lm.model("Choose:", Literal["Paris", "London", "Rome"])
```

### 4. Metrics for Experimentation

**Decision:** Track success/failure at each tier with `adapter.metrics`.

**Rationale:**
- Primary goal is data-driven model selection
- High Tier 3 usage → need larger model
- High Tier 2 success → fast path working, consider simplifying
- Informs memory/latency trade-offs for end-user deployments

**Metrics structure:**
```python
self.metrics = {
    'tier1_success': 0,    # ChatAdapter format worked
    'tier2_success': 0,    # JSON format worked
    'tier3_success': 0,    # Outlines constrained used
    'tier1_failures': 0,   # ChatAdapter failed
    'tier2_failures': 0,   # JSON parsing failed
}
```

## DSPy Integration Points

### Adapter Call Flow

**Reference:** `.venv/.../dspy/predict/predict.py:182-194` - `Predict.forward()`
- Line 185: `adapter = settings.adapter or ChatAdapter()` - gets configured adapter
- Line 192: `completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)`
- Adapter receives LM instance, kwargs, signature, demos, inputs
- Must return `list[dict[str, Any]]` of completions

**Reference:** `.venv/.../dspy/adapters/base.py:116-128` - `Adapter.__call__`
- Line 124-125: Calls `format()` to create messages, then `lm(messages=..., **lm_kwargs)`
- Line 128: Returns output from `_call_postprocess()`
- Line 89: `parse()` method is called during postprocessing to extract fields from completion text

### Error Handling

**AdapterParseError:** `.venv/.../dspy/utils/exceptions.py` (imported in json_adapter.py:23)
- Raised when JSON parsing fails in JSONAdapter
- `.venv/.../dspy/adapters/json_adapter.py:154-162` - Raised when `json_repair.loads()` can't parse
- Line 157-162: Also raised when parsed fields don't match signature output fields

**ContextWindowExceededError:** From litellm (imported in chat_adapter.py:5)
- Should NOT trigger fallback (re-raise immediately)
- No point retrying if context is too large

### JSON Adapter Behavior

**Reference:** `.venv/.../dspy/adapters/json_adapter.py`
- Line 41-82: `JSONAdapter.__call__` implementation
- Line 48-52: Checks if model supports `response_format` parameter
- Line 54-59: Falls back to `{"type": "json_object"}` if ToolCalls present or open-ended dict
- Line 73-82: Try structured output, catch Exception, fall back to JSON mode
- Line 80: Logs warning "Failed to use structured output format, falling back to JSON mode"

**Our Tier 2 should replicate:**
- Line 132-135: `user_message_output_requirements()` - adds JSON instruction to user message
- Line 149-179: `parse()` - uses `json_repair.loads()` to parse, raises `AdapterParseError` on failure
- Uses `json_repair` (imported line 5) for robustness

### Signature Field Extraction

**Reference:** `dspy_outlines/schema_extractor.py:5-30` - `extract_output_schema()`
- Iterates through `signature.output_fields`
- Returns first Pydantic BaseModel found
- Logs warnings for multiple output fields

**For constraint extraction:**
- Same pattern but return raw annotation instead of validating it's BaseModel
- `field.annotation` contains the type (could be Pydantic, Literal, Regex, etc.)

## Module Compatibility

### How DSPy Modules Work

**Reference:** `.venv/.../dspy/predict/` - All modules eventually call `Predict.forward()`
- `.../chain_of_thought.py:14-24` - Extends signature to add `reasoning` field
- `.../react.py` - Uses ToolCalls type for tool integration
- All modules use same adapter call flow

**ChainOfThought**:
- ✅ Works - adds `reasoning` field to output signature
- Adapter sees modified signature with extra field
- All tiers handle this transparently

**ProgramOfThought**:
- ✅ Works if output is Pydantic
- Output must be structured (not raw string)

**ReAct (tool calling)**:
- ⚠️ Skip Tier 3 - Outlines doesn't support ToolCalls
- `.venv/.../dspy/adapters/json_adapter.py:54` - Checks for `field.annotation == ToolCalls`
- If detected in signature, skip constrained fallback

**Multiple completions (n > 1)**:
- ⚠️ Log warning, Tier 3 returns single completion
- `.venv/.../dspy/predict/predict.py:139` - `num_generations` from config
- Outlines constrained generation is deterministic (can't generate n different outputs)

### Tool Call Detection

**Reference:** `.venv/.../dspy/adapters/types/tool.py` - ToolCalls type definition
**Reference:** `.venv/.../dspy/adapters/base.py:387-391` - `_get_tool_call_output_field_name()`
- Iterates output fields looking for `annotation == ToolCalls`
- Returns field name if found

**Pattern for Tier 3:**
```python
def _has_tool_calls(self, signature):
    for field in signature.output_fields.values():
        if field.annotation == ToolCalls:
            return True
    return False
```

## Outlines Constraint Handling

### Type Detection Flow

**Reference:** `.venv/.../outlines/types/dsl.py:707-798` - `python_types_to_terms()`

**Literal handling:**
- Line 780-781: `is_literal(ptype)` check
- Line 814-815: `_handle_literal()` → `Alternatives([python_types_to_terms(arg) for arg in args])`
- Each literal value becomes a term in alternatives

**Pydantic/dataclass/TypedDict:**
- Line 758-765: Structured type checks
- Line 764: `TypeAdapter(ptype).json_schema()` gets schema
- Line 765: Returns `JsonSchema(schema)`

**Enum:**
- Line 771-777: Extracts enum members, converts to Alternatives

**Basic types:**
- Line 734-749: int/float/bool/str mapped to predefined types
- `.venv/.../outlines/types/__init__.py:95-140` - Type definitions (e.g., `integer = Regex(...)`)

### Model Call Interface

**Reference:** `.venv/.../outlines/models/base.py:80-122` - `Model.__call__`
- User calls: `model(prompt, output_type, **kwargs)`
- Line 120-122: Creates Generator, calls it
- Generator handles type detection and logits processor creation

**Reference:** `.venv/.../outlines/models/mlxlm.py:118-150` - `MLXLM.generate()`
- Line 142-150: Calls `mlx_lm.generate()` with logits_processors
- `output_type` is formatted by type adapter (line 148)
- Type adapter converts to logits processor

**We bypass Generator when calling from DSPy:**
- OutlinesLM already wraps MLXLM
- When `_outlines_constraint` present, call `self.model(prompt, output_type=constraint)`
- Outlines handles type detection internally

## Implementation Checklist

### 1. Refactor OutlinesLM

- [ ] Rename `self.outlines_model` → `self.model`
- [ ] Update `forward()` to use `self.model`
- [ ] Change kwarg: `_outlines_schema` → `_outlines_constraint`
- [ ] Update constraint handling to pass any type to `self.model()`
- [ ] Ensure unconstrained path still works (no constraint = no output_type)

**Files to modify:**
- `dspy_outlines/lm.py`

### 2. Refactor OutlinesAdapter

- [ ] Extend `ChatAdapter` (not just use it)
- [ ] Implement three-tier `__call__` method
- [ ] Add `_json_fallback()` method (replicates JSONAdapter behavior)
- [ ] Add `_constrained_fallback()` method (sets `_outlines_constraint`)
- [ ] Add metrics tracking (`__init__` sets up dict, incremented in each tier)
- [ ] Add `_has_tool_calls()` helper
- [ ] Skip Tier 3 if tool calls detected
- [ ] Log warnings for n > 1 at Tier 3

**Files to modify:**
- `dspy_outlines/adapter.py`

### 3. Update Schema Extractor

- [ ] Rename to `constraint_extractor.py` (or keep name, just change function)
- [ ] Change `extract_output_schema()` → `extract_constraint()`
- [ ] Return raw `field.annotation` instead of validating BaseModel
- [ ] Keep warning for multiple output fields

**Files to modify:**
- `dspy_outlines/schema_extractor.py`

### 4. Write Tests

- [ ] Benchmark test: measure Tier 1, 2, 3 speeds
- [ ] Test fallback chain with failing JSON
- [ ] Test Literal type constraint
- [ ] Test ChainOfThought compatibility
- [ ] Test metrics tracking
- [ ] Test direct `lm.model()` usage

**New files:**
- `tests/test_fallback_chain.py`
- `tests/test_benchmark_tiers.py`

### 5. Update Documentation

- [ ] Update `dspy_outlines/README.md` with new architecture
- [ ] Document metrics access: `adapter.metrics`
- [ ] Document direct Outlines usage: `lm.model(prompt, constraint)`
- [ ] Add examples for Literal, Regex, CFG constraints

**Files to modify:**
- `dspy_outlines/README.md`

## Future Extensions

### Multiple Choice Without DSPy

```python
from typing import Literal
lm.model(prompt, output_type=Literal["A", "B", "C"])
```

### Regex Patterns

```python
from outlines.types import Regex
lm.model(prompt, output_type=Regex(r"\d{3}-\d{4}"))
```

### Context-Free Grammars

```python
from outlines.types import CFG
lm.model(prompt, output_type=CFG(arithmetic_grammar))
```

These work today via direct Outlines access (`lm.model`). Future work could integrate into DSPy signatures with custom field annotations, but that's beyond current scope.

## References

**DSPy Source Code:**
- `.venv/lib/python3.13/site-packages/dspy/adapters/base.py` - Adapter base class
- `.venv/lib/python3.13/site-packages/dspy/adapters/chat_adapter.py` - Fallback pattern
- `.venv/lib/python3.13/site-packages/dspy/adapters/json_adapter.py` - JSON parsing logic
- `.venv/lib/python3.13/site-packages/dspy/predict/predict.py` - Module call flow
- `.venv/lib/python3.13/site-packages/dspy/utils/exceptions.py` - Error types

**Outlines Source Code:**
- `.venv/lib/python3.13/site-packages/outlines/types/dsl.py` - Type detection
- `.venv/lib/python3.13/site-packages/outlines/types/__init__.py` - Type definitions
- `.venv/lib/python3.13/site-packages/outlines/models/base.py` - Model interface
- `.venv/lib/python3.13/site-packages/outlines/models/mlxlm.py` - MLX integration
- `.venv/lib/python3.13/site-packages/outlines/generator.py` - Generator logic

**Outlines Documentation:**
- MLX-LM integration: Structured generation examples (Pydantic, Literal, Regex, CFG)
- Shows `model(prompt, output_type)` calling convention

**DSPy Documentation:**
- Modules page: Shows how ChainOfThought, ReAct, etc. work
- Adapters page: JSONAdapter behavior and when to use it
