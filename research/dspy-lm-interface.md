# DSPy LM Interface Research

## Summary

DSPy provides a `BaseLM` base class for custom language model implementations. Custom LMs must implement the `forward()` method and return responses in OpenAI format.

## Base Class to Inherit

**Class:** `dspy.clients.base_lm.BaseLM`

**Location:** `.venv/lib/python3.13/site-packages/dspy/clients/base_lm.py`

## Required Methods

### `__init__`

```python
def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
    self.model = model
    self.model_type = model_type
    self.cache = cache
    self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
    self.history = []
```

### `forward()` (REQUIRED)

```python
def forward(self, prompt=None, messages=None, **kwargs):
    """Forward pass for the language model.

    Subclasses must implement this method, and the response should be identical to
    OpenAI response format: https://platform.openai.com/docs/api-reference/responses/object
    """
    raise NotImplementedError("Subclasses must implement this method.")
```

**Parameters:**
- `prompt`: String prompt (optional, for text completion)
- `messages`: List of message dicts `[{"role": "...", "content": "..."}]` (for chat)
- `**kwargs`: Generation parameters (`max_tokens`, `temperature`, etc.)

**Returns:** OpenAI-format response object with structure:
```python
{
    "choices": [
        {
            "message": {"content": "...", "role": "assistant"},
            "finish_reason": "stop"
        }
    ],
    "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M},
    "model": "model-name"
}
```

### `aforward()` (OPTIONAL - for async support)

```python
async def aforward(self, prompt=None, messages=None, **kwargs):
    """Async forward pass for the language model."""
    raise NotImplementedError("Subclasses must implement this method.")
```

## How DSPy Uses Custom LMs

1. **User calls `dspy.Predict`** with a signature
2. **DSPy's adapter** (ChatAdapter or JSONAdapter) formats the prompt
3. **Adapter calls `lm(prompt, messages, **kwargs)`**
4. **BaseLM's `__call__`** intercepts and calls `forward()`
5. **Custom `forward()`** generates response
6. **BaseLM processes response** via `_process_lm_response()`
7. **Result returned to DSPy**

## Adapter Layer

DSPy uses adapters to bridge signatures ↔ LM calls:

- **ChatAdapter**: Basic chat message formatting
- **JSONAdapter**: Structured output formatting (extends ChatAdapter)
- **XMLAdapter**: XML-based formatting
- **TwoStepAdapter**: Two-step reasoning

**Key insight:** Adapters call `lm(prompt, messages, **kwargs)` which invokes `__call__`, which calls `forward()`. We intercept at the `forward()` level.

## Extracting Pydantic Schemas from Signatures

DSPy's `JSONAdapter` already has logic to extract Pydantic models from signatures:

**Function:** `_get_structured_outputs_response_format(signature)`

**Location:** `dspy/adapters/json_adapter.py:210-288`

**What it does:**
1. Takes a `SignatureMeta` (signature class)
2. Iterates over `signature.output_fields.items()`
3. Builds Pydantic model from field annotations: `pydantic.create_model("DSPyProgramOutputs", **fields)`
4. Enforces strict schema with `extra="forbid"`

**Key code:**
```python
fields = {}
for name, field in signature.output_fields.items():
    annotation = field.annotation
    default = field.default if hasattr(field, "default") else ...
    fields[name] = (annotation, default)

pydantic_model = pydantic.create_model(
    "DSPyProgramOutputs",
    __config__=pydantic.ConfigDict(extra="forbid"),
    **fields,
)
```

**For our hybrid LM:**
- We can access `signature` parameter in `forward()` via `kwargs` (adapters pass it)
- Extract `signature.output_fields` to get Pydantic models
- If output field is a `BaseModel`, use it for Outlines constrained generation

## Custom LM Implementation Strategy

1. **Subclass `BaseLM`** (not `LM` - simpler)
2. **Implement `forward()`** to:
   - Extract signature from kwargs if present
   - Get Pydantic schema from signature's output fields
   - Generate using Outlines with schema constraint
   - Return OpenAI-format response
3. **No adapter changes needed** - adapters already pass signature to LM

## Example Custom LM Skeleton

```python
import dspy
from pydantic import BaseModel

class OutlinesDSPyLM(dspy.BaseLM):
    def __init__(self, model_path=None):
        super().__init__(model="outlines-mlx")
        # Load Outlines + MLX model

    def forward(self, prompt=None, messages=None, **kwargs):
        # 1. Get signature from kwargs
        signature = kwargs.get('signature', None)

        # 2. Extract Pydantic schema
        schema = self._extract_schema(signature) if signature else None

        # 3. Format prompt
        formatted_prompt = self._format_prompt(prompt, messages)

        # 4. Generate with Outlines
        if schema:
            result = self.outlines_model(
                formatted_prompt,
                output_type=schema,
                max_tokens=kwargs.get('max_tokens', 512)
            )
        else:
            result = self.outlines_model(formatted_prompt)

        # 5. Return OpenAI format
        return {
            "choices": [{"message": {"content": result, "role": "assistant"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": self.model
        }

    def _extract_schema(self, signature):
        for field in signature.output_fields.values():
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                return field.annotation
        return None
```

## Key Findings

1. **BaseLM is the right base class** - simpler than LM (which uses LiteLLM)
2. **forward() is the interception point** - all generation goes through here
3. **Signature is passed in kwargs** - adapters pass it automatically
4. **output_fields contains Pydantic models** - easy to extract for Outlines
5. **Must return OpenAI format** - specific structure required
6. **No adapter modifications needed** - existing adapters work as-is

## Next Steps

1. Create minimal passthrough LM to verify interception works
2. Implement schema extraction utility
3. Integrate Outlines constrained generation
4. Test with existing dspy-poc.py
