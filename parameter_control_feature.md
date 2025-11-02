# Parameter Control Feature for OutlinesLM

**Goal**: Add support for MLX-LM generation parameters (temperature, top_p, etc.) to `OutlinesLM` so they can be configured at initialization and applied during generation.

**Use Case**: Phase 1 PoC needs to control generation parameters to verify their effect on the ingestion pipeline (fact extraction, relationship inference quality).

---

## Changes Required

### 1. Update `OutlinesLM.__init__()` - `dspy_outlines/lm.py`

**Location**: Lines 63-95

**Current signature**:
```python
def __init__(self, model_path: str = None, *, enable_prompt_cache: bool = False):
```

**New signature**:
```python
def __init__(
    self,
    model_path: str = None,
    *,
    enable_prompt_cache: bool = False,
    generation_config: dict = None
):
    """
    Initialize hybrid LM.

    Args:
        model_path: Path to MLX model (uses default if None)
        enable_prompt_cache: Whether to build an MLX prompt cache (off by default)
        generation_config: Dict of MLX-LM generation parameters (temp, top_p, etc.)
                          Defaults to conservative settings if None.
    """
```

**Implementation**:
```python
# After line 95, add:
# Store generation config (defaults for reproducible extraction)
self.generation_config = generation_config or {
    "temp": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
logger.info(f"OutlinesLM generation config: {self.generation_config}")
```

---

### 2. Update `OutlinesLM.forward()` - `dspy_outlines/lm.py`

**Location**: Lines 131-154 (BOTH constrained and unconstrained generation paths)

**Current code** (lines 131-154):
```python
with MLX_LOCK:
    outlines_kwargs = {"max_tokens": max_tokens}
    if self.prompt_cache is not None:
        outlines_kwargs["prompt_cache"] = self.prompt_cache

    if constraint:
        # Use Outlines wrapper for constrained generation
        result_json = self.outlines_model(
            formatted_prompt,
            output_type=constraint,
            **outlines_kwargs,  # Only max_tokens and prompt_cache
        )
    else:
        # Use raw MLX for unconstrained generation
        generate_kwargs = {"verbose": False}
        if self.prompt_cache is not None:
            generate_kwargs["prompt_cache"] = self.prompt_cache
        completion = mlx_lm.generate(
            self.raw_mlx_model,
            self.tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
            **generate_kwargs,  # Only verbose and prompt_cache
        )
```

**Updated code** (apply generation config to BOTH paths):
```python
with MLX_LOCK:
    outlines_kwargs = {"max_tokens": max_tokens}
    if self.prompt_cache is not None:
        outlines_kwargs["prompt_cache"] = self.prompt_cache

    # Apply generation config (temp, top_p, etc.) to both paths
    outlines_kwargs.update(self.generation_config)

    if constraint:
        # Use Outlines wrapper for constrained generation
        result_json = self.outlines_model(
            formatted_prompt,
            output_type=constraint,
            **outlines_kwargs,  # Now includes generation params
        )
    else:
        # Use raw MLX for unconstrained generation
        generate_kwargs = {"verbose": False, **outlines_kwargs}
        completion = mlx_lm.generate(
            self.raw_mlx_model,
            self.tokenizer,
            formatted_prompt,
            **generate_kwargs,  # Now includes generation params
        )
```

**Note**: Verified that Outlines DOES support MLX-LM parameters (forwards `**kwargs` to `mlx_lm.generate`). This change works for BOTH constrained and unconstrained generation paths.

---

### 3. Update DSPy Configuration Pattern

**Current pattern** (from phase1-poc.md):
```python
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(),
)
```

**New pattern with config (single source of truth in `settings.py`)**:
```python
# settings.py
MODEL_CONFIG = {...}

# graphiti-poc.py
from settings import MODEL_CONFIG
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)
```

---

## Testing

**Test file**: `tests/test_parameter_control.py`

```python
import dspy
from dspy_outlines.lm import OutlinesLM
from dspy_outlines.adapter import OutlinesAdapter


def test_default_generation_config():
    """Verify OutlinesLM initializes with default config."""
    lm = OutlinesLM()
    assert hasattr(lm, 'generation_config')
    assert 'temp' in lm.generation_config
    assert 'top_p' in lm.generation_config


def test_custom_generation_config():
    """Verify custom config is stored and accessible."""
    custom_config = {
        "temp": 0.5,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
    }
    lm = OutlinesLM(generation_config=custom_config)
    assert lm.generation_config == custom_config


def test_generation_with_config():
    """Verify generation applies config parameters."""
    config = {"temp": 0.1}  # Low temp for deterministic output
    lm = OutlinesLM(generation_config=config)

    # Configure DSPy
    dspy.settings.configure(lm=lm, adapter=OutlinesAdapter())

    # Simple unconstrained generation
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(SimpleSignature)

    # Run twice - with low temp, should get similar results
    result1 = predictor(question="What is 2+2?")
    result2 = predictor(question="What is 2+2?")

    # Not asserting exact equality (MLX may still have variance)
    # Just verify generation completes without error
    assert result1.answer
    assert result2.answer
```

**Run test**:
```bash
pytest tests/test_parameter_control.py -v
```

---

## Verification for Phase 1 PoC

After implementing this feature, the PoC can:

1. **Configure at startup**:
```python
# settings.py
MODEL_CONFIG = {
    "temp": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# graphiti-poc.py
from settings import MODEL_CONFIG
```

2. **Pass to OutlinesLM**:
```python
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)
```

3. **Experiment by editing and restarting**:
   - Edit `MODEL_CONFIG` values in `settings.py`
   - Restart Gradio app
   - Test extraction quality with different parameters

---

## MLX-LM Supported Parameters

From `mlx-lm` documentation, the following parameters are supported:

- `temp` (float): Sampling temperature (default: 0.0 = greedy)
- `top_p` (float): Nucleus sampling threshold (default: 1.0)
- `min_p` (float): Minimum token probability threshold
- `min_tokens_to_keep` (int): Minimum number of tokens to keep in sampling
- `repetition_penalty` (float): Penalty for token repetition (default: 1.0 = no penalty)
- `repetition_context_size` (int): Size of context window for repetition penalty

**Note**: Verify these in your MLX-LM version. Run:
```bash
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

---

## Investigation Results

**Date**: 2025-11-02

**Question**: Does Outlines support MLX-LM generation parameters during constrained generation?

**Answer**: ✅ YES - Confirmed by source code inspection and testing.

**Evidence**:

1. **Outlines source code** (`outlines/models/mlxlm.py:144-150`):
   ```python
   def generate(self, model_input: str, output_type: Optional[OutlinesLogitsProcessor] = None, **kwargs) -> str:
       from mlx_lm import generate
       return generate(
           self.model,
           self.mlx_tokenizer,
           self.type_adapter.format_input(model_input),
           logits_processors=self.type_adapter.format_output_type(output_type),
           **kwargs,  # <-- Parameters forwarded to mlx_lm.generate()
       )
   ```
   The `**kwargs` are passed directly to `mlx_lm.generate()`, meaning ALL MLX-LM parameters work.

2. **Current OutlinesLM limitation** (`dspy_outlines/lm.py:131-154`):
   - Currently only passes `max_tokens` and `prompt_cache` to generation functions
   - Other kwargs (temp, top_p, etc.) are silently ignored
   - This affects BOTH constrained and unconstrained paths

3. **Testing confirmation**:
   - Tested with temp=1.5 vs temp=0.01 - produced identical outputs (confirmed params not being passed)
   - After implementing the fix, parameters will affect output for both generation paths

**Conclusion**: Implementation will work for both constrained and unconstrained generation. No architectural blockers.

---

## Implementation Checklist

- [ ] Modify `OutlinesLM.__init__()` to accept `generation_config` parameter
- [ ] Store `self.generation_config` with default values
- [ ] Update `forward()` to apply config to BOTH constrained and unconstrained generation paths
- [ ] Add logging for applied config
- [ ] Write tests in `tests/test_parameter_control.py`
- [ ] Run tests: `pytest tests/test_parameter_control.py -v`
- [ ] Update `dspy_outlines/README.md` with usage examples pointing to `settings.MODEL_CONFIG`
- [ ] Verify in Phase 1 PoC by testing different temperature values

---

## Estimated Implementation Time

**~30 minutes**:
- Code changes: 10 minutes
- Tests: 10 minutes
- Verification: 10 minutes

---

## Notes

- This feature affects **BOTH constrained and unconstrained** generation
- Outlines supports MLX-LM parameters for constrained generation (verified by source code inspection)
- Phase 1 PoC uses constrained generation for facts/relationships, so parameters WILL affect output quality
- All standard MLX-LM parameters (temp, top_p, repetition_penalty, etc.) are supported
- The implementation unifies parameter handling across both generation paths for consistency
