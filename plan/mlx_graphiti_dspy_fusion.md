# MLX-Graphiti-DSPy Integration: Unified Architecture

## Executive Summary

**Integration Goal:** Implement a single `MLXLMClient` that bridges Graphiti's knowledge graph pipeline with DSPy's prompt optimization and Outlines' structured generation, all running locally on MLX.

**Current State:**
- ✅ `MLXEmbedder` working (from mlx_test.py)
- ✅ `KuzuDriver` supported by Graphiti
- ✅ Outlines MLX integration proven
- ❌ Homebrew extraction pipeline missing 7 Graphiti features
- ❌ No DSPy prompt optimization
- ❌ No integration between all three systems

**Target State:**
- ✅ Single `MLXLMClient(LLMClient)` bridges all three frameworks
- ✅ DSPy + Outlines enabled by default (toggleable for testing)
- ✅ All 9 Graphiti features work automatically
- ✅ Prompt optimization via DSPy compilers
- ✅ 100% local, ~400 lines of integration code

---

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 1: GRAPHITI ORCHESTRATION                                  │
│ ├─ Entity extraction + reflexion                                 │
│ ├─ Relationship extraction + reflexion                           │
│ ├─ Node/Edge deduplication                                       │
│ ├─ Summarization                                                 │
│ ├─ Fact paraphrasing                                             │
│ └─ Calls: llm_client._generate_response(messages, response_model)│
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│ Layer 2: MLXLMClient (BRIDGE)                                    │
│ ├─ Receives: list[Message], type[BaseModel]                     │
│ ├─ Mode Toggle: use_dspy (default: True)                        │
│ ├─ DSPy Mode: Route to DSPy modules                             │
│ ├─ Direct Mode: Bypass DSPy, use Outlines directly              │
│ └─ Returns: dict[str, Any]                                       │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
    ┌───────────────▼────────┐    ┌──────▼───────────────┐
    │ DSPy Mode (default)    │    │ Direct Mode          │
    └───────────┬────────────┘    └──────┬───────────────┘
                │                         │
┌───────────────▼───────────────┐         │
│ Layer 3a: DSPY LAYER          │         │
│ ├─ Custom DSPy Modules        │         │
│ ├─ DSPy Signatures            │         │
│ ├─ Custom GraphitiAdapter     │         │
│ └─ Uses: MLXOutlinesLM        │         │
└───────────────┬───────────────┘         │
                │                         │
                └────────┬────────────────┘
                         │
        ┌────────────────▼─────────────────┐
        │ Layer 3b: MLX + OUTLINES         │
        │ ├─ MLX model + tokenizer         │
        │ ├─ Outlines structured generation│
        │ ├─ ModelFormatter (Qwen/Llama)   │
        │ └─ Guarantees valid JSON         │
        └──────────────────────────────────┘
```

---

## Core Components

### 1. MLXLMClient - The Bridge

**File:** `app/llm/mlx_client.py`

**Purpose:** Implements Graphiti's `LLMClient` interface, dispatches to DSPy or direct mode

```python
from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from typing import Any

class MLXLMClient(LLMClient):
    """
    Graphiti LLM client adapter for MLX-LM + DSPy + Outlines.

    Routes Graphiti operations through DSPy modules (default) or direct generation.
    Uses Outlines for guaranteed structured outputs in both modes.
    """

    def __init__(
        self,
        model,                      # MLX model
        tokenizer,                  # MLX tokenizer
        formatter: ModelFormatter,  # Qwen/Llama chat template
        config: LLMConfig | None = None,
        use_dspy: bool = True,      # DSPy enabled by default
    ):
        super().__init__(config, cache=False)
        # Store models and create Outlines integration
        # Configure DSPy if enabled
        # Initialize DSPy modules registry
        pass

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Route to DSPy modules or direct generation based on mode"""
        if self.use_dspy:
            return await self._dspy_route(messages, response_model)
        else:
            return await self._direct_generate(messages, response_model)

    async def _dspy_route(self, messages, response_model):
        """
        DSPy Mode:
        1. Infer task type from response_model
        2. Parse Graphiti messages → structured inputs
        3. Invoke appropriate DSPy module
        4. Convert DSPy Prediction → dict
        """
        pass

    async def _direct_generate(self, messages, response_model):
        """
        Direct Mode:
        1. Format messages with ModelFormatter
        2. Call Outlines with Pydantic schema
        3. Return validated dict
        """
        pass

    def toggle_dspy(self, enabled: bool):
        """Toggle DSPy mode for A/B testing"""
        pass
```

**Key Responsibilities:**
- Implements Graphiti's `LLMClient` contract
- Dispatches to DSPy modules or direct generation
- Handles mode toggling for performance comparison
- Manages both code paths

---

### 2. MLXOutlinesLM - Custom DSPy Backend

**File:** `app/llm/mlx_dspy_backend.py`

**Purpose:** Makes MLX + Outlines available as a DSPy language model

```python
import dspy
import outlines
from app.llm.model_formatter import ModelFormatter

class MLXOutlinesLM(dspy.BaseLM):
    """
    Custom DSPy LM backend using MLX + Outlines.

    Enables DSPy modules to use local MLX models with
    Outlines-enforced structured generation.
    """

    def __init__(self, model, tokenizer, formatter: ModelFormatter):
        self.mlx_model = model
        self.mlx_tokenizer = tokenizer
        self.outlines_model = outlines.from_mlxlm(model, tokenizer)
        self.formatter = formatter

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Dual invocation mode:
        - DSPy path: prompt=str (from DSPy modules)
        - Direct path: messages=list[Message] (from direct mode)

        Both use Outlines for structured generation.
        """
        # Format input (prompt or messages)
        # Extract response_model from kwargs
        # Generate with Outlines structured output
        # Return JSON string
        pass
```

**Key Responsibilities:**
- Implements `dspy.BaseLM` interface
- Wraps MLX + Outlines for DSPy consumption
- Handles both DSPy-style and direct invocations
- Uses `ModelFormatter` for chat templates

---

### 3. GraphitiAdapter - Custom DSPy Adapter

**File:** `app/llm/graphiti_dspy_adapter.py`

**Purpose:** Formats DSPy signatures optimally for small local models

```python
import dspy
from dspy.adapters import Adapter

class GraphitiAdapter(Adapter):
    """
    Custom DSPy adapter optimized for small local models.

    Formats prompts more concisely than ChatAdapter's [[## field ##]] syntax.
    Designed to work with Outlines for structured output enforcement.
    """

    def format_field_structure(self, signature):
        """Define how output structure is presented"""
        pass

    def format_task_description(self, signature):
        """Format the task instructions"""
        pass

    def format_user_message_content(self, inputs: dict, signature):
        """Format input fields as natural language"""
        pass

    def parse(self, signature, completion: str):
        """
        Parse LM output to extract fields.
        Outlines already enforces structure, so mainly validates.
        """
        pass
```

**Key Responsibilities:**
- Implements DSPy's `Adapter` interface
- Optimizes prompt formatting for small models
- Works with Outlines-enforced structure
- Less verbose than default ChatAdapter

---

### 4. DSPy Modules - Custom Prompt Library

**File:** `app/prompts/dspy_modules.py`

**Purpose:** Define DSPy signatures and modules for each Graphiti operation

```python
import dspy
from typing import List

# ============================================================================
# DSPy Signatures - Define Input/Output Contracts
# ============================================================================

class EntityExtractionSignature(dspy.Signature):
    """Extract entities (people, organizations, locations) from text"""
    episode_content: str = dspy.InputField(desc="Text to extract entities from")
    entity_types: str = dspy.InputField(desc="Comma-separated entity types")
    entities: List[dict] = dspy.OutputField(desc="Extracted entities with id, name, type")

class RelationshipExtractionSignature(dspy.Signature):
    """Extract relationships between entities"""
    episode_content: str = dspy.InputField(desc="Original text")
    entities: str = dspy.InputField(desc="List of entities with IDs")
    reference_time: str = dspy.InputField(desc="Reference timestamp")
    relationships: List[dict] = dspy.OutputField(desc="Extracted relationships")

class NodeDeduplicationSignature(dspy.Signature):
    """Identify duplicate entities"""
    new_node: str = dspy.InputField(desc="New entity to check")
    existing_nodes: str = dspy.InputField(desc="Existing entities")
    duplicates: List[int] = dspy.OutputField(desc="Indices of duplicates")

class EdgeDeduplicationSignature(dspy.Signature):
    """Identify duplicate or contradictory relationships"""
    new_edge: str = dspy.InputField(desc="New relationship")
    existing_edges: str = dspy.InputField(desc="Existing relationships")
    duplicate_facts: List[int] = dspy.OutputField(desc="Duplicate indices")
    contradicted_facts: List[int] = dspy.OutputField(desc="Contradiction indices")

class SummarizationSignature(dspy.Signature):
    """Generate entity summary"""
    episode_content: str = dspy.InputField(desc="Messages mentioning entity")
    node: str = dspy.InputField(desc="Entity to summarize")
    summary: str = dspy.OutputField(desc="Concise summary (max 250 words)")

# ============================================================================
# DSPy Modules - Composable, Optimizable Units
# ============================================================================

class EntityExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(self, episode_content, entity_types):
        return self.extract(episode_content=episode_content, entity_types=entity_types)

class RelationshipExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(RelationshipExtractionSignature)

    def forward(self, episode_content, entities, reference_time):
        return self.extract(episode_content=episode_content, entities=entities, reference_time=reference_time)

# Additional modules for deduplication, summarization...

# ============================================================================
# Module Registry
# ============================================================================

DSPY_MODULES = {
    'extract_entities': EntityExtractorModule,
    'extract_relationships': RelationshipExtractorModule,
    'deduplicate_nodes': NodeDeduplicatorModule,
    'deduplicate_edges': EdgeDeduplicatorModule,
    'summarize': SummarizerModule,
}
```

**Key Responsibilities:**
- Define all Graphiti operations as DSPy signatures
- Wrap signatures in DSPy modules (Predict, ChainOfThought)
- Registry for module lookup by task type
- Can be optimized with DSPy compilers

---

### 5. ModelFormatter - Chat Template Abstraction

**File:** `app/llm/model_formatter.py`

**Purpose:** Abstract interface for model-specific chat templates

```python
from abc import ABC, abstractmethod
from graphiti_core.prompts.models import Message

class ModelFormatter(ABC):
    """Abstract interface for model-specific chat template formatting"""

    @abstractmethod
    def format_messages(self, messages: list[Message]) -> str:
        """Convert Graphiti messages to model-specific prompt format"""
        pass

    @abstractmethod
    def format_prompt(self, prompt: str) -> str:
        """Format single prompt string (for DSPy direct calls)"""
        pass

    @abstractmethod
    def get_stop_tokens(self) -> list[str]:
        """Return model-specific stop tokens"""
        pass
```

**File:** `app/llm/formatters/qwen.py`

```python
from app.llm.model_formatter import ModelFormatter
from graphiti_core.prompts.models import Message

class QwenFormatter(ModelFormatter):
    """Qwen chat template formatter"""

    def format_messages(self, messages: list[Message]) -> str:
        """
        Convert to Qwen chat format:
        <|im_start|>system\n{content}<|im_end|>\n
        <|im_start|>user\n{content}<|im_end|>\n
        <|im_start|>assistant\n
        """
        formatted = []
        for msg in messages:
            formatted.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
        formatted.append("<|im_start|>assistant\n")
        return "".join(formatted)

    def format_prompt(self, prompt: str) -> str:
        """Format single prompt for DSPy calls"""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def get_stop_tokens(self) -> list[str]:
        return ["<|im_end|>", "<|endoftext|>"]
```

**Key Responsibilities:**
- Abstract chat template formatting
- Support both multi-message (Graphiti) and single-prompt (DSPy) formats
- Model-agnostic interface (Qwen, Llama, etc.)
- Used by both MLXLMClient and MLXOutlinesLM

---

### 6. MLXEmbedder - Embedding Client

**File:** `app/llm/mlx_embedder.py`

**Purpose:** Local embedding generation (unchanged from mlx_test.py)

```python
import mlx.core as mx
from graphiti_core.embedder import EmbedderClient
from typing import List

class MLXEmbedder(EmbedderClient):
    """Local embedder using MLX for Apple Silicon"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def create(self, input_data: str | List[str]) -> List[float]:
        """Create embedding for a single text input"""
        # Tokenize, run through model, mean pool
        pass

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """Create embeddings for batch of text inputs"""
        pass
```

---

## Module Structure

```
app/
├── llm/
│   ├── __init__.py
│   ├── mlx_client.py              # MLXLMClient - Bridge layer
│   ├── mlx_embedder.py            # MLXEmbedder - Embedding client
│   ├── mlx_dspy_backend.py        # MLXOutlinesLM - Custom DSPy LM
│   ├── graphiti_dspy_adapter.py   # GraphitiAdapter - Custom DSPy adapter
│   ├── model_formatter.py         # ModelFormatter - Abstract interface
│   └── formatters/
│       ├── __init__.py
│       ├── qwen.py                # QwenFormatter - Qwen chat templates
│       └── llama.py               # LlamaFormatter - Llama templates (future)
├── prompts/
│   ├── __init__.py
│   ├── dspy_modules.py            # DSPy signatures + modules
│   └── dspy_optimizers.py         # DSPy optimization scripts (future)
└── services/
    └── graphiti_service.py        # High-level graph operations (future)
```

---

## Integration Flow

### DSPy Mode (Default)

```
Graphiti.add_episode("Alice works at Google")
  │
  ├─→ extract_nodes(episode_content, entity_types)
  │     │
  │     └─→ MLXLMClient._generate_response(messages, ExtractedEntities)
  │           │
  │           ├─→ Infer task: "extract_entities"
  │           ├─→ Parse messages → {episode_content, entity_types}
  │           ├─→ Get DSPy module: EntityExtractorModule
  │           ├─→ Invoke module(episode_content, entity_types)
  │           │     │
  │           │     └─→ DSPy.ChainOfThought
  │           │           │
  │           │           └─→ GraphitiAdapter.format()
  │           │                 │
  │           │                 └─→ MLXOutlinesLM(prompt, response_model=ExtractedEntities)
  │           │                       │
  │           │                       └─→ Outlines.from_mlxlm(prompt, output_type=ExtractedEntities)
  │           │                             │
  │           │                             └─→ Returns valid JSON
  │           │
  │           └─→ Convert Prediction → dict → Return to Graphiti
  │
  ├─→ resolve_extracted_nodes() → deduplicate
  ├─→ extract_edges() → relationships
  ├─→ resolve_extracted_edges() → deduplicate facts
  ├─→ extract_attributes_from_nodes() → summarize
  └─→ add_nodes_and_edges_bulk() → database write
```

### Direct Mode (DSPy Disabled)

```
Graphiti.add_episode("Alice works at Google")
  │
  ├─→ extract_nodes(episode_content, entity_types)
  │     │
  │     └─→ MLXLMClient._generate_response(messages, ExtractedEntities)
  │           │
  │           └─→ _direct_generate()
  │                 │
  │                 ├─→ ModelFormatter.format_messages(messages)
  │                 ├─→ Outlines.from_mlxlm(prompt, output_type=ExtractedEntities)
  │                 └─→ Return validated dict
  │
  └─→ [Rest of Graphiti pipeline unchanged]
```

---

## Implementation Phases

### Phase 1: Foundation (No DSPy Yet)
1. Implement `ModelFormatter` + `QwenFormatter`
2. Implement `MLXEmbedder` (migrate from mlx_test.py)
3. Implement `MLXLMClient` (direct mode only)
   - `_generate_response` → `_direct_generate` only
   - Verify Outlines integration works
4. Test with Graphiti - confirm all 9 features work

**Deliverable:** Working Graphiti + MLX + Outlines integration

---

### Phase 2: DSPy Integration
1. Implement `MLXOutlinesLM(dspy.BaseLM)`
   - Handle prompt/messages dual invocation
   - Integrate Outlines structured generation
2. Implement `GraphitiAdapter(dspy.Adapter)`
   - Optimize formatting for small models
3. Define DSPy signatures for all Graphiti operations
4. Implement DSPy modules (EntityExtractor, etc.)
5. Add DSPy routing to `MLXLMClient._dspy_route`
6. Add mode toggle: `MLXLMClient.toggle_dspy()`

**Deliverable:** Full DSPy integration with A/B testing capability

---

### Phase 3: Optimization (Future)
1. Collect training examples (labeled extractions)
2. Use DSPy optimizers:
   - `BootstrapFewShot` - Learn from examples
   - `MIPRO` - Optimize instructions
3. Compare optimized vs default prompts
4. Save compiled modules for production

**Deliverable:** Optimized prompts for small models

---

## Key Design Decisions

### 1. Why Two Modes (DSPy + Direct)?

**DSPy Mode Benefits:**
- Systematic prompt optimization via compilers
- Composable, modular extraction pipeline
- Can improve over time with more training data

**Direct Mode Benefits:**
- Simpler code path for debugging
- Lower latency (no DSPy overhead)
- Baseline for measuring DSPy improvements

**Toggle enables A/B testing:** Quantify DSPy's impact on extraction quality

---

### 2. Why Custom DSPy Adapter?

**Problem:** Default ChatAdapter uses `[[## field ##]]` markers - verbose for small models

**Solution:** `GraphitiAdapter` formats prompts more concisely:
- Natural language field descriptions
- Inline context (no separated schema)
- Optimized for 1B-7B parameter models

**Trade-off:** Custom adapter requires maintenance but yields better small-model performance

---

### 3. Why Separate ModelFormatter?

**Benefits:**
- Model-agnostic architecture
- Easy to add new models (Llama, Phi, etc.)
- Used by both DSPy and direct modes
- Single source of truth for chat templates

**Example:** Swap Qwen → Llama by changing one line:
```python
formatter = LlamaFormatter()  # Instead of QwenFormatter()
```

---

### 4. Why MLXOutlinesLM vs Direct Outlines?

**DSPy path needs LM backend:** DSPy modules expect `dspy.BaseLM` interface

**MLXOutlinesLM provides:**
- DSPy compatibility layer
- Outlines integration for structured generation
- Dual invocation support (prompt/messages)
- Reusable across all DSPy modules

**Result:** DSPy modules work with MLX + Outlines seamlessly

---

## Usage Examples

### Basic Usage (DSPy Enabled by Default)

```python
import mlx_lm
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.utils.datetime_utils import utc_now

from app.llm.mlx_client import MLXLMClient
from app.llm.mlx_embedder import MLXEmbedder
from app.llm.formatters.qwen import QwenFormatter

# Load model once
model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-3B-Instruct-4bit")

# Create components (DSPy enabled by default)
formatter = QwenFormatter()
llm_client = MLXLMClient(model, tokenizer, formatter)  # use_dspy=True default
embedder = MLXEmbedder(model, tokenizer)
driver = KuzuDriver(":memory:")

# Initialize Graphiti
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=llm_client,
    embedder=embedder,
)
await graphiti.build_indices_and_constraints()

# Use Graphiti normally - DSPy modules handle all operations
result = await graphiti.add_episode(
    name="test",
    episode_body="Alice works at Google and Bob founded TechCo",
    source_description="test input",
    reference_time=utc_now(),
)

print(f"Extracted {len(result.nodes)} entities, {len(result.edges)} relationships")
```

---

### A/B Testing: DSPy vs Direct Mode

```python
# Test with DSPy (optimized prompts)
llm_client.toggle_dspy(True)
result_dspy = await graphiti.add_episode(
    name="test_dspy",
    episode_body="Alice works at Google",
    source_description="test",
    reference_time=utc_now(),
)

# Test without DSPy (baseline)
llm_client.toggle_dspy(False)
result_direct = await graphiti.add_episode(
    name="test_direct",
    episode_body="Alice works at Google",
    source_description="test",
    reference_time=utc_now(),
)

# Compare results
print(f"DSPy Mode: {len(result_dspy.nodes)} entities")
print(f"Direct Mode: {len(result_direct.nodes)} entities")
```

---

### Future: DSPy Optimization

```python
import dspy
from app.prompts.dspy_modules import EntityExtractorModule

# Collect labeled examples
trainset = [
    dspy.Example(
        episode_content="Alice works at Google",
        entity_types="PERSON,ORGANIZATION",
        entities=[
            {"id": 0, "name": "Alice", "entity_type": "PERSON"},
            {"id": 1, "name": "Google", "entity_type": "ORGANIZATION"}
        ]
    ).with_inputs("episode_content", "entity_types"),
    # ... more examples
]

# Define metric
def entity_extraction_metric(example, prediction, trace=None):
    # Compare predicted entities with gold labels
    pass

# Optimize module with BootstrapFewShot
optimizer = dspy.BootstrapFewShot(metric=entity_extraction_metric)
optimized_extractor = optimizer.compile(
    EntityExtractorModule(),
    trainset=trainset
)

# Replace default module with optimized version
llm_client.dspy_modules['extract_entities'] = optimized_extractor
```

---

## Feature Comparison

### Before (mlx_test.py - Homebrew)
| Feature | Status | Implementation |
|---------|--------|----------------|
| Entity extraction | ✅ Basic | Custom prompts |
| Relationship extraction | ✅ Basic | Custom prompts |
| Node deduplication | ❌ | - |
| Edge deduplication | ❌ | - |
| Summarization | ❌ | - |
| Fact paraphrasing | ❌ | - |
| Reflexion loops | ❌ | - |
| Context window | ❌ | - |
| Contradiction detection | ❌ | - |
| Prompt optimization | ❌ | - |
| **TOTAL** | **2/10 features** | **~600 lines custom code** |

### After (MLXLMClient + DSPy + Graphiti)
| Feature | Status | Implementation |
|---------|--------|----------------|
| Entity extraction | ✅ Full | Graphiti + DSPy |
| Relationship extraction | ✅ Full | Graphiti + DSPy |
| Node deduplication | ✅ Semantic + LLM | Graphiti |
| Edge deduplication | ✅ Semantic + LLM | Graphiti |
| Summarization | ✅ Incremental | Graphiti |
| Fact paraphrasing | ✅ LLM-based | Graphiti |
| Reflexion loops | ✅ Auto | Graphiti |
| Context window | ✅ Configurable | Graphiti |
| Contradiction detection | ✅ Temporal | Graphiti |
| Prompt optimization | ✅ DSPy compilers | DSPy |
| **TOTAL** | **10/10 features** | **~400 lines integration** |

**Net Result:**
- 600 lines custom → 400 lines integration
- 2/10 features → 10/10 features
- No optimization → DSPy compilers
- Manual prompts → Optimizable modules

---

## Code Estimates

| Component | Lines | Purpose |
|-----------|-------|---------|
| `MLXLMClient` | ~150 | Bridge layer, routing logic |
| `MLXOutlinesLM` | ~80 | Custom DSPy backend |
| `GraphitiAdapter` | ~100 | Custom DSPy adapter |
| `dspy_modules.py` | ~150 | Signatures + modules (5 modules) |
| `ModelFormatter` | ~30 | Abstract interface |
| `QwenFormatter` | ~40 | Qwen implementation |
| `MLXEmbedder` | ~30 | Embedding client (existing) |
| Supporting code | ~20 | Imports, utils |
| **TOTAL NEW CODE** | **~600** | Integration layer |
| **GRAPHITI REUSED** | **~2,800** | All orchestration logic |
| **NET BENEFIT** | **600 → 2,800** | 4.6x code leverage |

---

## Dependencies

```toml
[project]
dependencies = [
    "graphiti-core>=0.21.0",
    "kuzu>=0.11.2",
    "mlx-lm>=0.28.2",
    "outlines>=1.2.6",
    "dspy>=2.6.0",           # NEW: DSPy framework
]

[tool.uv]
override-dependencies = [
    "outlines-core @ git+https://github.com/dottxt-ai/outlines-core.git@0.2.13",
]
```

---

## Success Criteria

### Technical Requirements
- ✅ All 9 Graphiti features working with MLX models
- ✅ DSPy mode produces valid extractions
- ✅ Direct mode produces valid extractions (baseline)
- ✅ Toggle between modes without restart
- ✅ Outlines guarantees valid JSON in both modes

### Performance Metrics
- ✅ Extract entities from test corpus
- ✅ Measure: precision, recall, F1
- ✅ Compare: DSPy vs Direct mode
- ✅ Quantify: DSPy optimization improvement

### Maintainability
- ✅ Model-agnostic (swap via formatter)
- ✅ No Graphiti fork required
- ✅ DSPy modules independently testable
- ✅ Clear separation of concerns

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DSPy overhead too high | Medium | Medium | Toggle to direct mode |
| Small model quality poor | High | High | DSPy optimization, larger models |
| Outlines incompatibility | Low | High | Proven working in mlx_test.py |
| Complex adapter logic | Medium | Medium | Extensive testing, fallbacks |
| DSPy module bugs | Medium | Medium | Can fall back to direct mode |

---

## Next Steps

1. **Phase 1:** Implement direct mode (MLX + Outlines + Graphiti)
2. **Verify:** All 9 Graphiti features work
3. **Phase 2:** Add DSPy layer (modules, adapter, backend)
4. **Test:** Compare DSPy vs direct mode performance
5. **Phase 3:** Optimize prompts with DSPy compilers
6. **Deploy:** Production system with optimized modules

---

## Conclusion

This architecture achieves full integration of three complex systems:

**Graphiti:** Provides battle-tested knowledge graph extraction pipeline
**DSPy:** Enables systematic prompt optimization and modular design
**Outlines + MLX:** Guarantees valid structured outputs on local Apple Silicon

**Key Insight:** By implementing a single bridge class (`MLXLMClient`) with dual-mode dispatch, we get:
- All Graphiti features (~2,800 lines) for free
- DSPy optimization when needed
- Outlines structured generation always
- A/B testing to measure improvements
- Model-agnostic, maintainable architecture

**Total integration cost:** ~400 lines of carefully designed glue code
**Total value:** 10/10 features + prompt optimization + 100% local inference
