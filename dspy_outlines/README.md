# dspy_outlines

Integration layer bridging DSPy's signature-based programming with Outlines' guaranteed constrained generation and MLX's local inference.

## Architecture Overview

This module enables structured output generation by connecting three frameworks:

```
DSPy Signature → OutlinesAdapter → OutlinesLM → Outlines → MLX → Generated JSON
```

### Component Responsibilities

#### 1. OutlinesAdapter (`adapter.py`)
**Purpose**: Extract Pydantic schemas from DSPy signatures and pass them to the LM.

- Extends `dspy.adapters.ChatAdapter`
- Inspects DSPy Signature output fields at runtime
- Extracts Pydantic `BaseModel` schemas using `schema_extractor`
- Passes schema via `_outlines_schema` kwarg to LM
- Passes field name via `_outlines_field_name` kwarg for JSON wrapping

**Thread Safety**: ✅ Stateless - safe for concurrent use

#### 2. OutlinesLM (`lm.py`)
**Purpose**: Execute constrained generation using Outlines and MLX.

- Inherits from `dspy.BaseLM` (not `dspy.LM`)
- Receives Pydantic schemas from adapter via kwargs
- Formats prompts using DSPy's message formatting
- Calls Outlines with schema constraints
- Returns OpenAI-format response as `SimpleNamespace` objects
- Maintains generation history

**Thread Safety**: ⚠️ **Requires locking** - see Thread Safety section below

#### 3. Schema Extractor (`schema_extractor.py`)
**Purpose**: Runtime introspection of DSPy signatures.

- Extracts Pydantic `BaseModel` from signature output field annotations
- Validates that output field is a Pydantic model
- Logs warnings for multiple output fields (uses first one)

**Thread Safety**: ✅ Stateless - safe for concurrent use

#### 4. MLX Loader (`mlx_loader.py`)
**Purpose**: Load and wrap MLX models for Outlines.

- Loads quantized MLX models from `.models/` directory
- Creates Outlines wrapper via `outlines.from_mlxlm()`
- Default model: `mlx-community--Qwen3-4B-Instruct-2507-8bit`

**Thread Safety**: ✅ Model loading is single-threaded during initialization

## Thread Safety

### Critical Requirement: MLX Requires Locking

**MLX (via Apple's Metal framework) is NOT thread-safe.** Multiple threads cannot call the same MLX model simultaneously without causing Metal command buffer race conditions.

**Evidence**:
```
Metal Error: "A command encoder is already encoding to this command buffer"
Metal Error: "Completed handler provided after commit call"
```

These errors occur when multiple threads attempt concurrent inference on the same MLX model instance.

### Implementation

`OutlinesLM` uses a **threading.Lock** to serialize access to the MLX model:

```python
import threading

# Module-level lock shared across all OutlinesLM instances
MLX_LOCK = threading.Lock()

class OutlinesLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        with MLX_LOCK:
            # All MLX model calls happen inside lock
            result = self.outlines_model(...)
        return result
```

**Design Decisions**:

1. **Module-level lock**: Since all `OutlinesLM` instances share the same underlying MLX model (loaded once), a single module-level lock protects all instances.

2. **threading.Lock vs asyncio.Lock**: We use `threading.Lock` because:
   - MLX operations are synchronous (CPU/GPU compute)
   - DSPy predictor calls are synchronous
   - `threading.Lock` has lower overhead than async locks
   - Lock scope is limited to actual MLX inference (not entire forward())

3. **Lock granularity**: The lock wraps only the `outlines_model()` call, not the entire `forward()` method. This allows:
   - Concurrent prompt formatting
   - Concurrent schema validation
   - Concurrent history updates
   - Only model inference is serialized

### Thread Safety Summary

| Component | Thread Safe? | Notes |
|-----------|--------------|-------|
| `OutlinesAdapter` | ✅ Yes | Stateless |
| `schema_extractor` | ✅ Yes | Pure function |
| `mlx_loader` | ✅ Yes | Called during init only |
| `OutlinesLM` | ⚠️ With Lock | Requires `MLX_LOCK` |

### Usage Patterns

**Single-threaded (default)**: Works without modification
```python
lm = OutlinesLM()
dspy.configure(lm=lm, adapter=OutlinesAdapter())
predictor = dspy.Predict(Signature)
result = predictor(input="...")  # Lock handled automatically
```

**Multi-threaded**: Each thread configures DSPy, but shares LM instance
```python
lm = OutlinesLM()  # Single instance shared across threads

def worker():
    dspy.configure(lm=lm, adapter=OutlinesAdapter())  # Per-thread config
    predictor = dspy.Predict(Signature)
    result = predictor(input="...")  # Lock prevents race conditions

threads = [threading.Thread(target=worker) for _ in range(5)]
```

**Note on DSPy threading**: DSPy requires `dspy.configure()` to be called in each thread that uses predictors (thread-local settings), but the LM instance can be shared.

## Implementation Details

### DSPy Integration Points

**BaseLM vs LM**: This module uses `dspy.BaseLM` instead of `dspy.LM`:

- **Simpler interface**: No LiteLLM dependency required
- **Custom backends**: Ideal for Outlines+MLX integration
- **Return format**: Must return `SimpleNamespace` objects (not dicts)
- **Usage tracking**: `usage` field must be dict-convertible (we use `AttrDict`)

**Adapter System**: DSPy's adapter system allows intercepting calls between the signature and LM:

```python
# Without adapter
signature → lm.forward(prompt)

# With OutlinesAdapter
signature → adapter(extracts schema) → lm.forward(prompt, _outlines_schema=Schema)
```

### Outlines Integration

Outlines provides guaranteed structured output through:

1. **Schema compilation**: Converts Pydantic models to JSON Schema
2. **FSM generation**: Creates finite state machine for valid JSON paths
3. **Constrained sampling**: Masks logits to only allow valid next tokens
4. **Validation**: Returns valid JSON that parses to Pydantic model

**Integration flow**:
```python
# In OutlinesLM.forward()
if schema:
    result_json = self.outlines_model(
        prompt,
        output_type=schema,  # Pydantic model
        max_tokens=max_tokens
    )
    parsed = schema.model_validate_json(result_json)  # Always succeeds
```

### MLX Integration

MLX provides efficient local inference on Apple Silicon:

- **Quantized models**: 8-bit quantization for memory efficiency
- **Metal acceleration**: Direct GPU access via Metal framework
- **Model loading**: One-time load, reused across generations
- **Memory management**: Models stay resident in GPU memory

**Model path**: `.models/mlx-community--Qwen3-4B-Instruct-2507-8bit`

## Usage Examples

### Basic Extraction

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

# Configure
dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())

# Use
predictor = dspy.Predict(QA)
result = predictor(question="What is 2+2?")
print(result.answer.response)  # Guaranteed valid Answer object
```

### Knowledge Graph Extraction

```python
from pydantic import BaseModel
from typing import List

class Node(BaseModel):
    id: str
    label: str
    type: str

class Edge(BaseModel):
    source: str
    target: str
    relation: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class ExtractGraph(dspy.Signature):
    """Extract knowledge graph from text."""
    text: str = dspy.InputField()
    graph: KnowledgeGraph = dspy.OutputField()

dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())
predictor = dspy.Predict(ExtractGraph)

result = predictor(text="Paris is the capital of France.")
# result.graph is guaranteed to be a valid KnowledgeGraph
```

## Design Constraints

### Why Module-Level Lock?

**Question**: Why not instance-level locks?

**Answer**: MLX models are loaded once and shared. Creating multiple `OutlinesLM` instances doesn't create multiple MLX models - they all reference the same loaded model in memory. A module-level lock reflects this reality.

**Alternative considered**: Load separate MLX model per `OutlinesLM` instance
- **Rejected**: Would consume 4GB+ GPU memory per instance (model size)
- **Rejected**: Model loading takes 5-10 seconds per instance

### Why Not Async?

**Question**: Why `threading.Lock` instead of `asyncio.Lock`?

**Answer**:
1. MLX inference is synchronous CPU/GPU compute (cannot be awaited)
2. DSPy predictors are synchronous
3. Using async would require `run_in_executor()` which adds overhead
4. `threading.Lock` is simpler and has lower overhead for sync code

### Lock Scope Trade-offs

**Current design**: Lock only wraps `outlines_model()` call

**Considered**: Lock entire `forward()` method
- **Rejected**: Unnecessarily serializes prompt formatting, validation, history updates
- **Benefit of current**: Only inference is serialized, rest can run concurrently

**Considered**: No lock, document as single-threaded only
- **Rejected**: Users will inevitably use threading (e.g., Gradio, FastAPI)
- **Rejected**: Silent failures and segfaults are worse than serialization

## Dependencies

**External**:
- `dspy-ai`: Signature-based LLM programming
- `outlines`: Constrained generation engine
- `mlx`: Apple Silicon inference framework
- `mlx-lm`: LLM utilities for MLX
- `pydantic`: Schema validation

**Internal**:
- Uses `dspy.BaseLM` interface
- Uses `dspy.adapters.ChatAdapter` base class
- Integrates with DSPy's configuration system

## Future Improvements

### Potential Enhancements

1. **Async support**: Add async variants with proper lock coordination
   - `async def aforward()` using `asyncio.Lock`
   - Would require `outlines` async support (not yet available)

2. **Model pooling**: Multiple MLX model instances for parallel inference
   - Trade-off: Memory (4GB+ per instance) vs throughput
   - Requires coordination between pool and lock mechanism

3. **Multiple output fields**: Currently uses first output field only
   - Schema extraction could merge multiple Pydantic models
   - Outlines would need to support multi-schema generation

4. **Batching**: Batch multiple prompts into single inference
   - MLX supports batching
   - Requires coordinating multiple DSPy predictor calls

5. **Lock metrics**: Track lock contention and wait times
   - Help identify bottlenecks in threaded applications
   - Could inform decision to switch to model pooling

### Known Limitations

1. **Serial inference**: Lock serializes all inference, limiting throughput
   - Acceptable for interactive use (Gradio UI)
   - May bottleneck high-throughput scenarios

2. **Single model**: All instances share one model
   - Prevents using different models simultaneously
   - Could add model registry with per-model locks

3. **DSPy thread-local settings**: Each thread must call `dspy.configure()`
   - This is a DSPy limitation, not specific to this module
   - Documented in usage examples

## Testing

Run integration tests:
```bash
pytest tests/test_constrained_generation.py -v
```

Test schema extraction:
```bash
pytest tests/test_schema_extractor.py -v
```

Verify constraint enforcement:
```bash
pytest tests/test_constrained_generation.py::test_outlines_actually_constrains -v
```

## Troubleshooting

### Metal Framework Errors

**Symptom**: `failed assertion` errors from Metal framework

**Cause**: Multiple threads calling MLX without proper locking

**Solution**: Verify `MLX_LOCK` is being used in `OutlinesLM.forward()`

### Schema Not Extracted

**Symptom**: Generation produces text instead of JSON

**Cause**: Output field is not a Pydantic `BaseModel`

**Solution**: Ensure signature output field is typed as `BaseModel` subclass:
```python
# Wrong
class Sig(dspy.Signature):
    output: str = dspy.OutputField()  # Plain string

# Right
class Output(BaseModel):
    text: str

class Sig(dspy.Signature):
    output: Output = dspy.OutputField()  # Pydantic model
```

### Thread Hanging

**Symptom**: Thread blocks forever waiting for lock

**Cause**: Deadlock or lock not released (exception during inference)

**Solution**: Lock uses context manager (`with MLX_LOCK`) which guarantees release even on exception. Check for nested lock acquisition.

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Outlines Documentation](https://outlines-dev.github.io/outlines/)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Metal Framework](https://developer.apple.com/metal/)
