# dspy_outlines

Integration layer bridging DSPy's signature-based programming with Outlines' guaranteed constrained generation and MLX's local inference. Features a three-tier fallback system optimized for speed with guaranteed correctness.

## Architecture Overview

```
DSPy Signature → OutlinesAdapter (3-tier fallback) → OutlinesLM → Outlines/MLX
```

**Three-tier fallback strategy:**
- **Chat**: ChatAdapter field-marker format (fastest, tries first)
- **JSON**: JSON unconstrained generation (fast, json_repair parsing)
- **OutlinesJSON**: Outlines constrained generation (slow, guaranteed valid)

Each tier falls back to the next on failure. Metrics track success/failure rates for experimentation.

## Component Responsibilities

### OutlinesAdapter (`adapter.py`)
**Purpose**: Three-tier fallback with automatic constraint extraction.

- Chat: Delegates to `ChatAdapter` (field markers like `Result: ...`)
- JSON: Prompts for JSON, parses with `json_repair` (robustness)
- OutlinesJSON: Adds `_outlines_constraint` kwarg for constrained generation
- Extracts constraints from signature output fields (Pydantic, Literal, Regex, etc.)
- Tracks metrics: `adapter.metrics['chat_success']`, etc.
- Skips OutlinesJSON for ToolCalls (Outlines doesn't support them)

**Thread Safety**: ✅ Stateless - safe for concurrent use

### OutlinesLM (`lm.py`)
**Purpose**: Execute constrained/unconstrained generation using Outlines and MLX.

- Inherits from `dspy.BaseLM`
- Receives constraints via `_outlines_constraint` kwarg
- Supports any Outlines constraint type (Pydantic, Literal, Regex, CFG, basic types)
- Exposes `lm.model` for direct Outlines access
- Returns OpenAI-format response as `SimpleNamespace` objects

**Thread Safety**: ⚠️ **Requires locking** - MLX is NOT thread-safe (uses `MLX_LOCK`)

### MLX Loader (`mlx_loader.py`)
**Purpose**: Load quantized MLX models for Outlines.

- Loads from `.models/` directory (not default location)
- Default: `mlx-community--Qwen3-4B-Instruct-2507-8bit`
- Creates Outlines wrapper via `outlines.from_mlxlm()`

**Thread Safety**: ✅ Model loading is single-threaded during initialization

## Quick Start

```python
from pydantic import BaseModel
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

# Define schema
class Answer(BaseModel):
    response: str
    confidence: float

# Define signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()

# Configure with fallback adapter
lm = OutlinesLM()
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)

# Use - automatically tries Chat → JSON → OutlinesJSON
predictor = dspy.Predict(QA)
result = predictor(question="What is 2+2?")
print(result.answer.response)

# Check which adapter succeeded
print(adapter.metrics)  # {'chat_success': 0, 'json_success': 1, ...}
```

## Direct Outlines Access

For non-Pydantic constraints (multiple choice, regex), use direct model access:

```python
from typing import Literal
from outlines.types import Regex

lm = OutlinesLM()

# Multiple choice (Literal)
city = lm.model("Name a European capital:", Literal["Paris", "London", "Rome"])

# Regex pattern
phone = lm.model("Your phone number:", Regex(r"\d{3}-\d{4}"))

# Context-free grammar
from outlines.types import CFG
result = lm.model("Calculate:", CFG(arithmetic_grammar))
```

**Why direct access?** DSPy signatures add overhead for simple tasks. For one-off extractions with non-Pydantic constraints, `lm.model()` provides direct Outlines access.

## Metrics Tracking

OutlinesAdapter tracks fallback performance for experimentation:

```python
adapter = OutlinesAdapter()
dspy.configure(lm=OutlinesLM(), adapter=adapter)

# Run predictions...

print(adapter.metrics)
# {
#     'chat_success': 5,           # ChatAdapter worked
#     'json_success': 3,           # JSON fallback worked
#     'outlines_json_success': 2,  # Outlines constrained used
#     'chat_failures': 5,          # ChatAdapter failed
#     'json_failures': 2,          # JSON parsing failed
# }
```

**High OutlinesJSON usage?** Consider larger model (better JSON formatting).
**High JSON success?** Fast path working, constrained generation rarely needed.

## Thread Safety

MLX (via Metal framework) is **NOT thread-safe**. `OutlinesLM` uses module-level `MLX_LOCK` to serialize inference:

```python
# Single LM instance shared across threads
lm = OutlinesLM()

def worker():
    dspy.configure(lm=lm, adapter=OutlinesAdapter())  # Thread-local config
    predictor = dspy.Predict(Signature)
    result = predictor(input="...")  # Lock prevents race conditions

threads = [threading.Thread(target=worker) for _ in range(5)]
```

Lock scope: Only wraps `self.model()` call, not entire `forward()`. Prompt formatting and result processing run concurrently.

## Testing

```bash
pytest tests/test_outlines_adapter_parsing.py -v  # Adapter parsing and integration tests
pytest tests/test_mlx_loader.py -v                # MLX model loading
pytest tests/test_mlx_lock.py -v                  # Thread safety verification
```
