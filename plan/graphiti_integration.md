# Graphiti Integration: Unified Local Inference Architecture

## Executive Summary

**Goal:** Bridge Graphiti's knowledge graph pipeline with local inference backends (MLX, llama.cpp) using Outlines for structured generation.

**Current State:**
- ✅ Working prototype: `mlx_test.py` (2/10 features, ~600 lines)
- ✅ Outlines + MLX integration proven
- ❌ Missing 7 Graphiti features (deduplication, summarization, reflexion, etc.)

**Target State:**
- ✅ Full Graphiti pipeline (10/10 features)
- ✅ Backend-agnostic (MLX + llama.cpp via Outlines)
- ✅ DSPy-compatible architecture (Phase 2 ready)
- ✅ ~170 lines integration code

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────┐
│ GRAPHITI                                    │
│ • All 9 operations (extraction, dedup, etc)│
└────────────┬────────────────────────────────┘
             │ _generate_response(messages, schema)
┌────────────▼────────────────────────────────┐
│ BRIDGE LAYER                                │
│ ┌─────────────────────────────────────────┐ │
│ │ GraphitiLM(LLMClient)                   │ │
│ │ • Format messages → prompt              │ │
│ │ • Call Outlines with schema             │ │
│ │ • Return validated dict                 │ │
│ │ • DSPy-compatible design                │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ Embedder(EmbedderClient)                │ │
│ │ • Wrap backend embedding APIs           │ │
│ └─────────────────────────────────────────┘ │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│ OUTLINES (Unified Structured Generation)    │
│ • Backend abstraction (mlxlm, llamacpp)    │
│ • Pydantic → constrained generation        │
│ • Guarantees valid JSON                    │
└────────────┬────────────────────────────────┘
             │
      ┌──────┴───────┐
      │              │
┌─────▼────┐  ┌──────▼──────┐
│ MLX      │  │ llama.cpp   │
│ (Apple   │  │ (Universal) │
│ Silicon) │  │             │
└──────────┘  └─────────────┘
```

---

## Module Structure

```
app/
├── llm/
│   ├── client.py              # GraphitiLM (~100 lines)
│   ├── embedder.py            # Embedder (~50 lines)
│   ├── backend.py             # Backend selection (~20 lines)
│   └── prompts.py             # Message formatting (~30 lines)
└── prompts/
    └── (Phase 2: DSPy modules)
```

**Total MVP: ~200 lines**

---

## Core Components

### 1. GraphitiLM: Graphiti ↔ Outlines Bridge

**File:** `app/llm/client.py`

**Responsibilities:**
- Implement `LLMClient` interface for Graphiti
- Convert Graphiti `Message` list → formatted prompt string
- Call Outlines with Pydantic schema for structured generation
- Parse/validate JSON response → dict
- Support mode toggle for future DSPy integration

**Key Design Pattern:**
```python
class GraphitiLM(LLMClient):
    """
    Bridge: Graphiti ↔ Outlines

    Design:
    1. Receives: list[Message] + Pydantic schema (from Graphiti)
    2. Formats: messages → prompt string
    3. Generates: Outlines structured generation
    4. Returns: validated dict
    """

    def __init__(self, outlines_model, mode="direct"):
        # mode: "direct" (Phase 1) or "dspy" (Phase 2)
        self.model = outlines_model
        self.mode = mode

    async def _generate_response(self, messages, response_model):
        """
        Core bridge logic:
        - Format messages → prompt
        - Generate with Outlines + schema
        - Parse JSON → dict
        """
        if self.mode == "dspy":
            return await self._dspy_generate(messages, response_model)
        else:
            return await self._direct_generate(messages, response_model)
```

**DSPy Compatibility:**
- `mode` parameter enables future DSPy routing
- `_dspy_generate()` stub ready for Phase 2
- No refactoring needed when adding DSPy

---

### 2. Embedder: Embedding Bridge

**File:** `app/llm/embedder.py`

**Responsibilities:**
- Implement `EmbedderClient` interface for Graphiti
- Delegate to backend-specific embedding methods
- Support both MLX (mean pooling) and llama.cpp (built-in)

**Key Design Pattern:**
```python
class Embedder(EmbedderClient):
    """
    Wraps backend embedding APIs

    Backends:
    - MLX: Mean pooling over hidden states
    - llama.cpp: Built-in embed() method
    """

    def __init__(self, backend_type, model_ref):
        self.backend = backend_type  # "mlx" or "llamacpp"
        self.model = model_ref

    async def create(self, text):
        if self.backend == "mlx":
            # MLX: tokenize + forward + mean pool
            pass
        elif self.backend == "llamacpp":
            # llama.cpp: use built-in embed()
            pass
```

---

### 3. Backend Selection: Environment-Aware Initialization

**File:** `app/llm/backend.py`

**Responsibilities:**
- Detect environment (Apple Silicon vs other)
- Initialize appropriate Outlines model
- Return model + backend type for embedder

**Key Design Pattern:**
```python
def create_backend(config=None):
    """
    Select and initialize backend based on environment/config

    Returns: (outlines_model, backend_type)
    """
    if config and config.backend:
        backend = config.backend
    else:
        backend = auto_detect()  # "mlx" on Apple Silicon, else "llamacpp"

    if backend == "mlx":
        model = outlines.models.mlxlm(config.model_name)
    elif backend == "llamacpp":
        model = outlines.models.llamacpp(config.model_path, n_gpu_layers=-1)

    return model, backend

def auto_detect():
    """Detect: Apple Silicon → mlx, else → llamacpp"""
    import platform
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        return "mlx"
    return "llamacpp"
```

---

### 4. Message Formatting: Graphiti Messages → Prompt

**File:** `app/llm/prompts.py`

**Responsibilities:**
- Convert Graphiti `Message` objects to prompt string
- Apply chat template (try native tokenizer first, fallback to manual)
- Keep simple and maintainable

**Key Design Pattern:**
```python
def format_messages(messages, tokenizer=None):
    """
    Convert Graphiti messages to prompt string

    Strategy:
    1. Try tokenizer.apply_chat_template() if available
    2. Fallback to simple template: system + user + assistant starter
    """
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        # Use native chat template
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Simple fallback
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
        prompt += "Assistant:"
        return prompt
```

---

## Usage Patterns

### Pattern 1: MLX Backend (Apple Silicon)

```python
from app.llm.backend import create_backend
from app.llm.client import GraphitiLM
from app.llm.embedder import Embedder

# Initialize backend
model, backend = create_backend(backend="mlx", model_name="mlx-community/Qwen2.5-3B-Instruct-4bit")

# Create bridge components
llm = GraphitiLM(model, mode="direct")
embedder = Embedder(backend, model)

# Initialize Graphiti
graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

### Pattern 2: llama.cpp Backend (Universal)

```python
# Change backend - everything else identical
model, backend = create_backend(backend="llamacpp", model_path="models/qwen2.5-3b.gguf")

llm = GraphitiLM(model, mode="direct")
embedder = Embedder(backend, model)

graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

### Pattern 3: Auto-Detection

```python
# Auto-select based on environment
model, backend = create_backend()  # Auto-detects Apple Silicon vs other

llm = GraphitiLM(model)
embedder = Embedder(backend, model)

graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

---

## Implementation Phases

### Phase 1: MVP (170 lines, 1-2 days)

**Components:**
1. `backend.py`: Environment detection + Outlines initialization (~20 lines)
2. `prompts.py`: Message formatting with fallback (~30 lines)
3. `client.py`: GraphitiLM bridge with DSPy stub (~100 lines)
4. `embedder.py`: Backend-specific embedding (~50 lines)

**Testing:**
- Run all 9 Graphiti operations
- Verify structured outputs (entity extraction, relationships, deduplication, etc.)
- Test backend switching (MLX ↔ llama.cpp)
- Collect baseline quality metrics

**Deliverable:** Working system with full Graphiti pipeline

---

### Phase 2: DSPy Integration (330 lines, conditional)

*Only proceed if Phase 1 quality/performance insufficient*

**Components:**
1. `dspy_backend.py`: Outlines-backed DSPy LM (~80 lines)
2. `dspy_adapter.py`: Custom adapter for small models (~100 lines)
3. `prompts/modules.py`: DSPy signatures + modules (~150 lines)

**Changes to Phase 1:**
- `GraphitiLM._dspy_generate()`: Implement DSPy routing
- Add mode toggle: `llm.set_mode("dspy")`

**Testing:**
- A/B test: direct vs DSPy modes
- Measure quality improvement
- Benchmark latency overhead

**Deliverable:** DSPy integration with measurable quality gains

---

### Phase 3: DSPy Optimization (weeks, conditional)

*Only proceed if Phase 2 shows value*

**Activities:**
1. Collect training examples (100+ labeled extractions)
2. Run DSPy optimizers (BootstrapFewShot, MIPRO)
3. Measure improvement vs baseline
4. Deploy optimized modules

---

## Design Decisions

### 1. Outlines as Single Abstraction Layer

**Decision:** Use Outlines for both MLX and llama.cpp (no custom backend abstraction)

**Rationale:**
- Outlines already provides backend abstraction
- Native support for both MLX and llama.cpp
- Unified structured generation (no dual GBNF/Outlines systems)

**Benefit:** Eliminates ~250 lines of redundant abstraction code

---

### 2. Defer DSPy to Phase 2

**Decision:** Build direct mode first, add DSPy only if needed

**Rationale:**
- DSPy requires training data not yet collected
- Unproven value for this specific use case
- Can be added without refactoring (via mode toggle)
- 72% code reduction in MVP

**Benefit:** Faster to production, measure baseline before optimizing

---

### 3. Simple Message Formatting

**Decision:** Try native chat templates, fallback to simple format

**Rationale:**
- Both MLX and llama.cpp tokenizers support `apply_chat_template()`
- Simple fallback sufficient for most cases
- Avoid premature abstraction

**Benefit:** ~110 lines saved vs custom ModelFormatter layer

---

### 4. DSPy-Compatible Bridge Design

**Decision:** Include `mode` parameter and `_dspy_generate()` stub in Phase 1

**Rationale:**
- Zero refactoring when adding DSPy
- A/B testing capability built-in
- Clear separation of concerns

**Benefit:** Phase 2 integration is additive, not refactor

---

## Success Criteria

### Phase 1 MVP
- ✅ All 9 Graphiti features functional
- ✅ Both backends work (MLX + llama.cpp)
- ✅ Backend switching without code changes
- ✅ Structured output valid 100% of time
- ✅ Code < 200 lines
- ✅ Baseline quality metrics collected

### Phase 2 DSPy (If Needed)
- ✅ Measurable quality improvement (precision, recall, F1)
- ✅ A/B testing working
- ✅ Acceptable latency overhead (<2x direct mode)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Native chat templates inadequate | Use simple fallback format |
| Backend-specific quirks | Outlines handles differences |
| DSPy needed for quality | Phase 2 design supports seamless addition |
| Outlines limitations | Proven working in current codebase |

---

## Dependencies

```toml
[project]
dependencies = [
    "graphiti-core>=0.21.0",
    "kuzu>=0.11.2",
    "mlx-lm>=0.28.2",
    "outlines>=1.2.6",
    # "dspy>=2.6.0",  # Phase 2 only
]

[tool.uv]
override-dependencies = [
    "outlines-core @ git+https://github.com/dottxt-ai/outlines-core.git@0.2.13",
]
```

---

## Migration from mlx_test.py

**Current prototype** (~600 lines):
- `MLXEmbedder` → reuse as-is, wrap in new `Embedder`
- `build_*_prompt()` → replace with Graphiti's prompts
- Custom extraction logic → delete (use Graphiti)
- `outlines.from_mlxlm()` → wrap in `GraphitiLM`

**Net result:** 600 lines custom → 170 lines integration
