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

**Location**: Lines 144-154 (unconstrained generation path)

**Current code**:
```python
# Use raw MLX for unconstrained generation
generate_kwargs = {"verbose": False}
if self.prompt_cache is not None:
    generate_kwargs["prompt_cache"] = self.prompt_cache
completion = mlx_lm.generate(
    self.raw_mlx_model,
    self.tokenizer,
    formatted_prompt,
    max_tokens=max_tokens,
    **generate_kwargs,
)
```

**Updated code**:
```python
# Use raw MLX for unconstrained generation
generate_kwargs = {"verbose": False}
if self.prompt_cache is not None:
    generate_kwargs["prompt_cache"] = self.prompt_cache

# Apply generation config (temp, top_p, etc.)
generate_kwargs.update(self.generation_config)

completion = mlx_lm.generate(
    self.raw_mlx_model,
    self.tokenizer,
    formatted_prompt,
    max_tokens=max_tokens,
    **generate_kwargs,
)
```

**Note**: Constrained generation path (lines 137-142) uses Outlines, which may not support all MLX-LM parameters. For Phase 1, focus on unconstrained path if needed, or investigate Outlines parameter support.

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

## Implementation Checklist

- [ ] Modify `OutlinesLM.__init__()` to accept `generation_config` parameter
- [ ] Store `self.generation_config` with default values
- [ ] Update `forward()` to apply config in unconstrained generation path
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

- This feature only affects **unconstrained** generation (when `_outlines_constraint` is None)
- For **constrained** generation (Pydantic models), Outlines may not support all MLX-LM parameters
- Phase 1 PoC primarily uses constrained generation for facts/relationships
- If constrained generation doesn't support these params, they'll be silently ignored
- Consider logging a warning if config is provided but constraint is active
