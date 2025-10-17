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
│ • All 9 operations (extraction, dedup, etc) │
└────────────┬────────────────────────────────┘
│ _generate_response(messages, schema)
┌────────────▼────────────────────────────────┐
│ BRIDGE LAYER                                │
│ ┌─────────────────────────────────────────┐ │
│ │ GraphitiLM(LLMClient)                   │ │
│ │ • Format messages → prompt              │ │
│ │ • Call Outlines with schema             │ │
│ │ • Return validated Pydantic object      │ │
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
│ • Backend abstraction (mlxlm, llamacpp)     │
│ • Pydantic → constrained generation         │
│ • Guarantees valid JSON + object            │
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
│   ├── embedder.py            # Embedder (~60 lines)
│   ├── backend.py             # Backend selection (~20 lines)
│   └── prompts.py             # Message formatting (~30 lines)
└── prompts/
└── (Phase 2: DSPy modules)

````

**Total MVP: ~200 lines**

---

## Core Components

### 1. GraphitiLM: Graphiti ↔ Outlines Bridge

**File:** `app/llm/client.py`

**Responsibilities:**
- Implement `LLMClient` interface for Graphiti
- Convert Graphiti `Message` list → formatted prompt string
- Call Outlines with a Pydantic schema for structured generation
- Return the generated **Pydantic object instance**, not just a dict
- Support mode toggle for future DSPy integration

**Key Design Pattern:**
```python
class GraphitiLM(LLMClient):
    """
    Bridge: Graphiti ↔ Outlines

    Design:
    1. Receives: list[Message] + Pydantic schema (response_model)
    2. Formats: messages → prompt string
    3. Generates: Outlines structured generation
    4. Returns: validated Pydantic object (instance of response_model)
    """

    def __init__(self, outlines_model, mode="direct"):
        self.model = outlines_model
        self.mode = mode

    async def _generate_response(self, messages, response_model):
        """
        Core bridge logic:
        - Format messages → prompt
        - Generate using Outlines + schema
        - Return Pydantic object instance (not dict)
        """
        if self.mode == "dspy":
            return await self._dspy_generate(messages, response_model)
        else:
            return await self._direct_generate(messages, response_model)

    async def _direct_generate(self, messages, response_model):
        prompt = format_messages(messages, getattr(self.model, "tokenizer", None))
        generator = outlines.generate.json(self.model, response_model)
        # Run blocking generation safely in async context
        result = await asyncio.to_thread(generator, prompt)
        return result
````

**Behavioral Clarification:**
`_generate_response()` mirrors Graphiti’s native OpenAI and Gemini clients:

* **Receives:** Graphiti `Message[]` and a `response_model` (Pydantic class)
* **Passes:** `response_model` directly to Outlines’ generator
* **Returns:** A validated **Pydantic object** of type `response_model`
* This ensures downstream Graphiti code can access attributes directly (e.g. `result.entities`), not just via a dict.

**Async Note:**
Outlines’ `generate.json()` is synchronous.
To maintain Graphiti’s async contract, `asyncio.to_thread()` is used to offload blocking generation without modifying Outlines internals.
This ensures compatibility with Graphiti’s async pipeline while preserving simplicity.

---

### 2. Embedder: Embedding Bridge

**File:** `app/llm/embedder.py`

**Responsibilities:**

* Implement `EmbedderClient` interface for Graphiti
* Use backend-specific embedding logic
* Support both MLX (manual mean pooling) and llama.cpp (native embed)
* Return consistent vector output for downstream use

**Key Design Pattern:**

```python
class Embedder(EmbedderClient):
    """
    Wraps backend embedding APIs

    Backends:
    - MLX: Mean pooling over last hidden states
    - llama.cpp: Built-in embed() method
    """

    def __init__(self, backend_type, model_ref):
        self.backend = backend_type  # "mlx" or "llamacpp"
        self.model = model_ref

    async def create(self, text):
        if self.backend == "mlx":
            # Tokenize and forward pass through model
            tokens = self.model.tokenizer(text, return_tensors="np")
            hidden = await asyncio.to_thread(self.model.model.forward, tokens["input_ids"])
            # Mean pool over sequence length
            embedding = hidden.mean(axis=1).squeeze()
        elif self.backend == "llamacpp":
            embedding = await asyncio.to_thread(self.model.embed, text)
        return embedding.tolist()
```

**Embedding Clarification:**

* **MLX:**

  * Tokenization via `mlx_lm` tokenizer
  * Forward pass via model’s `forward()`
  * Mean-pool across hidden states for sentence embedding
  * Normalize vectors if required by Graphiti’s downstream usage
* **llama.cpp:**

  * Uses the built-in `embed()` API from `llama_cpp.Llama`
  * Already returns float vectors, consistent dimensionality per model

This design guarantees uniform embeddings for both backends with minimal code.

---

### 3. Backend Selection: Environment-Aware Initialization

**File:** `app/llm/backend.py`

**Responsibilities:**

* Detect environment (Apple Silicon vs other)
* Initialize appropriate Outlines model
* Return model + backend type for embedder

**Key Design Pattern:**

```python
def create_backend(config=None):
    """
    Select and initialize backend based on environment/config

    Returns: (outlines_model, backend_type)
    """
    backend = config.backend if config and config.backend else auto_detect()

    if backend == "mlx":
        model = outlines.models.mlxlm(config.model_name)
    elif backend == "llamacpp":
        from llama_cpp import Llama
        llama = Llama(model_path=config.model_path, n_gpu_layers=-1)
        model = outlines.models.LlamaCpp(llama)

    return model, backend

def auto_detect():
    """Detect: Apple Silicon → mlx, else → llamacpp"""
    import platform
    if platform.system() == "Darwin" and "arm" in platform.machine():
        return "mlx"
    return "llamacpp"
```

---

### 4. Message Formatting: Graphiti Messages → Prompt

**File:** `app/llm/prompts.py`

**Responsibilities:**

* Convert Graphiti `Message` objects to a prompt string compatible with Outlines
* Prefer backend-native chat templates
* Fallback gracefully when unavailable

**Key Design Pattern:**

```python
def format_messages(messages, tokenizer=None):
    """
    Convert Graphiti messages to a chat prompt string.

    Strategy:
    1. If tokenizer supports apply_chat_template() → use it.
    2. Else → fallback to simple textual format.
    """
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
        prompt += "Assistant:"
        return prompt
```

**Templating Clarification:**

* MLX and other chat models (e.g. Qwen, Llama) often ship native chat templates.
  Using these via `apply_chat_template()` ensures consistent token boundaries and generation behavior.
* For models without templates, a minimal human-readable fallback is applied (`System:` / `User:` / `Assistant:`).
* This approach avoids tight coupling with backend-specific prompt rules, maximizing portability.

---

## Usage Patterns

### Pattern 1: MLX Backend (Apple Silicon)

```python
from app.llm.backend import create_backend
from app.llm.client import GraphitiLM
from app.llm.embedder import Embedder

model, backend = create_backend(backend="mlx", model_name="mlx-community/Qwen2.5-3B-Instruct-4bit")

llm = GraphitiLM(model, mode="direct")
embedder = Embedder(backend, model)

graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

### Pattern 2: llama.cpp Backend (Universal)

```python
model, backend = create_backend(backend="llamacpp", model_path="models/qwen2.5-3b.gguf")

llm = GraphitiLM(model)
embedder = Embedder(backend, model)

graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

### Pattern 3: Auto-Detection

```python
model, backend = create_backend()  # Auto-detects Apple Silicon vs other

llm = GraphitiLM(model)
embedder = Embedder(backend, model)

graphiti = Graphiti(llm_client=llm, embedder=embedder, driver=driver)
```

---

## Implementation Phases

### Phase 1: MVP

**Components:**

1. `backend.py`: Environment detection + Outlines initialization
2. `prompts.py`: Message formatting with fallback
3. `client.py`: GraphitiLM bridge with async handling + DSPy stub
4. `embedder.py`: Backend-specific embedding logic

**Testing:**

* Verify all 9 Graphiti operations work with structured outputs
* Validate returned objects are correct Pydantic instances
* Test both backends (MLX ↔ llama.cpp)
* Collect baseline quality and latency metrics

**Deliverable:**
Fully functional Graphiti pipeline using local inference via Outlines.

---

### Phase 2: DSPy Integration (conditional)

Only pursued if Phase 1 quality or latency is insufficient.

**Additions:**

* `dspy_backend.py` and `dspy_adapter.py` for DSPy model routing
* Implement `_dspy_generate()` in `GraphitiLM`
* Extend test suite for A/B comparisons (direct vs DSPy)

---

## Design Decisions

1. **Outlines as Unified Abstraction Layer**

   * Handles both MLX and llama.cpp via a consistent API
   * Enables structured generation using Pydantic
   * Eliminates redundant backend code

2. **Async Safety for Graphiti**

   * Outlines runs synchronously → wrapped in `asyncio.to_thread()`
   * Ensures non-blocking Graphiti pipeline integration

3. **Direct Return of Pydantic Objects**

   * Mirrors OpenAI/Gemini client semantics
   * Ensures Graphiti’s operations consume typed objects seamlessly

4. **Simple, Adaptive Message Formatting**

   * Uses backend-native templates when available
   * Provides consistent fallback for non-chat models

5. **Deferred DSPy Integration**

   * Optional, additive, and isolated
   * No refactoring required in Phase 1

---

## Success Criteria

**Phase 1 MVP**

* ✅ All Graphiti features functional
* ✅ Structured outputs are valid Pydantic objects
* ✅ Async-safe integration
* ✅ Works across MLX and llama.cpp backends
* ✅ Code under 200 lines
* ✅ Baseline quality metrics collected

**Phase 2 DSPy**

* ✅ A/B tested quality improvement
* ✅ Acceptable latency overhead (<2×)
* ✅ Reuses Phase 1 bridge

---

## Risk Assessment

| Risk                             | Mitigation                      |
| -------------------------------- | ------------------------------- |
| Backend-specific quirks          | Outlines abstraction            |
| Async blocking                   | Offload to thread executor      |
| Embedding dimension mismatch     | Normalize during Embedder stage |
| DSPy required for higher quality | Phase 2 ready architecture      |

---

## Dependencies

```toml
[project]
dependencies = [
    "graphiti-core>=0.22.0",
    "kuzu>=0.11.2",
    "mlx-lm>=0.28.2",
    "outlines>=1.2.6",
    "llama-cpp-python>=0.2.90",
    # "dspy>=2.6.0",  # Phase 2 only
]

[tool.uv]
override-dependencies = [
    "outlines-core @ git+https://github.com/dottxt-ai/outlines-core.git@0.2.13",
]
```

---

## Migration from mlx_test.py

**Current prototype (~600 lines):**

* `MLXEmbedder` → simplified into new `Embedder`
* `build_*_prompt()` → replaced by `format_messages()`
* Custom JSON validation → handled by Outlines
* `outlines.from_mlxlm()` → replaced by `create_backend() + GraphitiLM`

**Net result:**
600 lines custom → ~200 lines modular integration.
