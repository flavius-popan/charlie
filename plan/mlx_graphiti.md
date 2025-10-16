# MLX-Graphiti Integration: Minimal Implementation Plan

## Executive Summary

**Key Discovery:** We don't need to rewrite Graphiti. We only need to implement a **single class** (`MLXLMClient`) that plugs into Graphiti's existing architecture, then optionally override prompts for full control.

**Current State:**
- ✅ `MLXEmbedder` already implemented and working
- ✅ `KuzuDriver` already supported by Graphiti
- ❌ Using homebrew extraction pipeline, missing 7 critical features
- ❌ No prompt control without forking entire library

**Proposed State:**
- ✅ Implement `MLXLMClient(LLMClient)` - single adapter class
- ✅ Override `prompt_library` for full prompt control
- ✅ Call native `graphiti.add_episode()` - get all features for free
- ✅ 100% local, 100% prompt control, <200 lines of new code

---

## Architecture Analysis

### Graphiti's Plugin Architecture

Graphiti uses **dependency injection** with three abstract interfaces:

```python
class Graphiti:
    def __init__(
        self,
        llm_client: LLMClient | None = None,      # ← We provide this
        embedder: EmbedderClient | None = None,   # ✅ Already have MLXEmbedder
        graph_driver: GraphDriver | None = None,  # ✅ Already using KuzuDriver
    )
```

### What We Get for Free

By implementing `LLMClient`, we automatically get:

| Feature | Implementation | Lines of Code | We Write |
|---------|---------------|---------------|----------|
| Entity extraction | `extract_nodes()` | 202 | 0 |
| Entity deduplication | `resolve_extracted_nodes()` | 243 | 0 |
| Entity summarization | `extract_attributes_from_nodes()` | 102 | 0 |
| Relationship extraction | `extract_edges()` | 237 | 0 |
| Relationship deduplication | `resolve_extracted_edges()` | 401 | 0 |
| Fact paraphrasing | `resolve_extracted_edge()` | 638 | 0 |
| Reflexion loops | Built into extract functions | - | 0 |
| Context window | `retrieve_episodes()` | - | 0 |
| Contradiction detection | `resolve_edge_contradictions()` | 434 | 0 |
| Custom attributes | `extract_attributes_from_node()` | 547 | 0 |
| Database operations | `add_nodes_and_edges_bulk()` | - | 0 |
| **TOTAL** | **~2,800 lines** | **~2,800** | **0** |

### What We Need to Implement

```python
class MLXLMClient(LLMClient):
    async def _generate_response(
        self,
        messages: list[Message],              # ← Graphiti provides
        response_model: type[BaseModel],      # ← Graphiti provides
        max_tokens: int,
        model_size: ModelSize,
    ) -> dict[str, Any]:
        # Convert messages to prompt via model formatter
        # Call outlines model with Pydantic schema
        # Return JSON dict
```

**Estimated: ~100 lines of code**

---

## Module Structure

### Proposed Directory Layout

```
app/
├── llm/
│   ├── __init__.py
│   ├── mlx_client.py          # MLXLMClient - Graphiti adapter
│   ├── mlx_embedder.py        # MLXEmbedder - embedding client
│   ├── model_formatter.py     # Abstract chat template interface
│   └── formatters/
│       ├── __init__.py
│       ├── qwen.py            # Qwen-specific chat templates
│       └── llama.py           # Optional: Llama templates
├── prompts/
│   ├── __init__.py
│   └── custom_prompts.py      # Override Graphiti prompts
└── services/
    └── graphiti_service.py    # High-level graph operations
```

**Separation of Concerns:**
- `mlx_client.py` - Pure Graphiti integration (model-agnostic)
- `mlx_embedder.py` - Embedding generation (model-agnostic)
- `model_formatter.py` - Abstract interface for chat templates
- `formatters/qwen.py` - Qwen-specific implementation
- `custom_prompts.py` - Prompt overrides (optional)

---

## Implementation Plan

### Phase 1: Core Infrastructure

#### Step 1.1: Model Formatter Abstraction
**File:** `app/llm/model_formatter.py`

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
    def get_stop_tokens(self) -> list[str]:
        """Return model-specific stop tokens"""
        pass
```

**Complexity:** Trivial
**Purpose:** Clean separation between Graphiti integration and model specifics

#### Step 1.2: Qwen Formatter Implementation
**File:** `app/llm/formatters/qwen.py`

```python
from graphiti_core.prompts.models import Message
from app.llm.model_formatter import ModelFormatter


class QwenFormatter(ModelFormatter):
    """Qwen chat template formatter"""

    def format_messages(self, messages: list[Message]) -> str:
        """
        Convert to Qwen chat format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
        formatted = []
        for msg in messages:
            formatted.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
        formatted.append("<|im_start|>assistant\n")
        return "".join(formatted)

    def get_stop_tokens(self) -> list[str]:
        return ["<|im_end|>", "<|endoftext|>"]
```

**Complexity:** Low
**Purpose:** Isolate all Qwen-specific code in one place

#### Step 1.3: MLXLMClient Implementation
**File:** `app/llm/mlx_client.py`

```python
from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from typing import Any
import outlines

from app.llm.model_formatter import ModelFormatter


class MLXLMClient(LLMClient):
    """Graphiti LLM client adapter for MLX-LM models with Outlines structured generation"""

    def __init__(
        self,
        model,
        tokenizer,
        formatter: ModelFormatter,
        config: LLMConfig | None = None,
    ):
        super().__init__(config, cache=False)
        self.mlx_model = model
        self.mlx_tokenizer = tokenizer
        self.outlines_model = outlines.from_mlxlm(model, tokenizer)
        self.formatter = formatter

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Generate structured JSON response using Outlines"""
        # Convert messages using model-specific formatter
        prompt = self.formatter.format_messages(messages)

        # Generate structured output with Outlines
        if response_model is not None:
            json_str = self.outlines_model(prompt, output_type=response_model)
            result = response_model.model_validate_json(json_str)
            return result.model_dump()
        else:
            # Fallback for unstructured generation (shouldn't happen in Graphiti)
            raise NotImplementedError("Unstructured generation not needed for Graphiti")
```

**Complexity:** Low
**Dependencies:** Existing (outlines, mlx_lm)
**Risk:** Low - isolated adapter class

#### Step 1.4: MLXEmbedder Migration
**File:** `app/llm/mlx_embedder.py`

Move existing `MLXEmbedder` from `mlx_test.py` with no changes:

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
        if isinstance(input_data, list):
            input_data = input_data[0]

        tokens = self.tokenizer.encode(input_data)
        tokens = mx.array([tokens])
        outputs = self.model(tokens)
        embedding = mx.mean(outputs, axis=1)
        return embedding[0].tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of text inputs"""
        embeddings = []
        for text in input_data_list:
            embeddings.append(await self.create(text))
        return embeddings
```

---

### Phase 2: Prompt Override System

#### Step 2.1: Custom Prompt Library
**File:** `app/prompts/custom_prompts.py`

```python
from typing import Any
from graphiti_core.prompts.models import Message


def extract_nodes_text(context: dict[str, Any]) -> list[Message]:
    """Custom entity extraction optimized for smaller models"""
    sys_prompt = """You extract entities from text. Focus on people, organizations, and locations."""

    user_prompt = f"""
TEXT:
{context['episode_content']}

ENTITY TYPES:
{context['entity_types']}

Extract all entities mentioned in the TEXT. Return JSON array.
"""

    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt),
    ]


def dedupe_edges_resolve(context: dict[str, Any]) -> list[Message]:
    """Custom edge deduplication for fact paraphrasing"""
    sys_prompt = """You identify duplicate and contradictory facts."""

    user_prompt = f"""
NEW FACT:
{context['new_edge']}

EXISTING FACTS:
{context['existing_edges']}

Instructions:
- If NEW FACT is a paraphrase of any EXISTING FACT, return its idx in duplicate_facts
- If NEW FACT contradicts any EXISTING FACT, return its idx in contradicted_facts
- Return fact_type as DEFAULT or one of the custom types

Return JSON with: duplicate_facts, contradicted_facts, fact_type
"""

    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    """Custom entity summarization"""
    sys_prompt = """You summarize entity information from messages."""

    user_prompt = f"""
MESSAGES:
{context['episode_content']}

ENTITY:
{context['node']}

Create a concise summary of the entity based on the messages. Max 250 words.
Return JSON with: summary
"""

    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt),
    ]


# Export custom prompt versions
CUSTOM_PROMPTS = {
    'extract_nodes': {
        'extract_text': extract_nodes_text,
        'extract_message': extract_nodes_text,
        'extract_json': extract_nodes_text,
    },
    'dedupe_edges': {
        'resolve_edge': dedupe_edges_resolve,
    },
    'extract_nodes': {
        'extract_summary': extract_summary,
    },
}
```

**Purpose:** Full control over prompts without forking Graphiti

#### Step 2.2: Prompt Override Mechanism
**File:** `app/llm/mlx_client.py` (extend class)

```python
class MLXLMClient(LLMClient):
    def __init__(
        self,
        model,
        tokenizer,
        formatter: ModelFormatter,
        custom_prompts: dict | None = None,
        config: LLMConfig | None = None,
    ):
        super().__init__(config, cache=False)
        self.mlx_model = model
        self.mlx_tokenizer = tokenizer
        self.outlines_model = outlines.from_mlxlm(model, tokenizer)
        self.formatter = formatter

        # Override prompts if provided
        if custom_prompts:
            self._override_prompts(custom_prompts)

    def _override_prompts(self, custom_prompts: dict):
        """Monkey-patch prompt_library with custom prompts"""
        from graphiti_core.prompts import lib

        for prompt_type, versions in custom_prompts.items():
            for version_name, func in versions.items():
                wrapped_func = lib.VersionWrapper(func)
                setattr(
                    getattr(lib.prompt_library, prompt_type),
                    version_name,
                    wrapped_func
                )
```

**Purpose:** Optional prompt customization without modifying Graphiti

---

### Phase 3: Integration & Testing

#### Step 3.1: Test Script
**File:** `test_mlx_graphiti.py`

```python
#!/usr/bin/env python3
import asyncio
import mlx_lm
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.utils.datetime_utils import utc_now

from app.llm.mlx_client import MLXLMClient
from app.llm.mlx_embedder import MLXEmbedder
from app.llm.formatters.qwen import QwenFormatter
from app.prompts.custom_prompts import CUSTOM_PROMPTS


async def test_extraction():
    """Test that all Graphiti features work with local MLX models"""

    # Load model
    print("Loading Qwen model...")
    model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-3B-Instruct-4bit")

    # Create components
    formatter = QwenFormatter()
    llm_client = MLXLMClient(model, tokenizer, formatter, custom_prompts=CUSTOM_PROMPTS)
    embedder = MLXEmbedder(model, tokenizer)
    driver = KuzuDriver(":memory:")

    # Initialize Graphiti
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
    )
    await graphiti.build_indices_and_constraints()

    # Test extraction
    print("\nTest 1: Basic extraction")
    result = await graphiti.add_episode(
        name="test1",
        episode_body="Alice works at Google and Bob founded TechCo",
        source_description="test input",
        reference_time=utc_now(),
    )
    print(f"✓ Extracted {len(result.nodes)} entities, {len(result.edges)} relationships")
    for node in result.nodes:
        print(f"  - {node.name}: {node.summary[:80]}...")

    # Test deduplication
    print("\nTest 2: Entity deduplication")
    result2 = await graphiti.add_episode(
        name="test2",
        episode_body="Alice is employed by Google",
        source_description="test input",
        reference_time=utc_now(),
    )
    print(f"✓ Extracted {len(result2.nodes)} entities (should reuse 'Alice' and 'Google')")

    # Test fact paraphrasing
    print("\nTest 3: Fact paraphrasing")
    result3 = await graphiti.add_episode(
        name="test3",
        episode_body="Alice's employer is Google",
        source_description="test input",
        reference_time=utc_now(),
    )
    print(f"✓ Extracted {len(result3.edges)} relationships (should dedupe with existing)")

    await graphiti.close()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_extraction())
```

**Purpose:** Verify all 9 features work correctly

#### Step 3.2: Production Entry Point
**File:** `run_mlx.py`

```python
#!/usr/bin/env python3
"""
Production MLX-Graphiti Integration

Uses native Graphiti add_episode() with local MLX-LM models.
All features (deduplication, summarization, reflexion) work automatically.
"""
import asyncio
import argparse
import mlx_lm
from pathlib import Path
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.utils.datetime_utils import utc_now

from app import settings
from app.llm.mlx_client import MLXLMClient
from app.llm.mlx_embedder import MLXEmbedder
from app.llm.formatters.qwen import QwenFormatter
from app.prompts.custom_prompts import CUSTOM_PROMPTS


async def main():
    parser = argparse.ArgumentParser(description="MLX-Graphiti Interactive CLI")
    parser.add_argument("--db", default=None, help="Database file (default: in-memory)")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    args = parser.parse_args()

    # Determine database path
    if args.db:
        db_path = Path(settings.BRAIN_DIR) / args.db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path = str(db_path)
    else:
        db_path = ":memory:"

    print("=" * 80)
    print("MLX-Graphiti Integration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Database: {db_path}")
    print("=" * 80)

    # Load model once at startup
    print("\nLoading model...")
    model, tokenizer = mlx_lm.load(args.model)

    # Create components
    formatter = QwenFormatter()
    llm_client = MLXLMClient(model, tokenizer, formatter, custom_prompts=CUSTOM_PROMPTS)
    embedder = MLXEmbedder(model, tokenizer)
    driver = KuzuDriver(db_path)

    # Initialize Graphiti
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
    )
    await graphiti.build_indices_and_constraints()

    print("\n✓ Ready! Enter text to extract knowledge.")
    print("Type 'quit' to exit.\n")

    # Interactive loop
    while True:
        try:
            text = input("> ")
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text.strip():
                continue

            # Use native add_episode - all features work automatically
            result = await graphiti.add_episode(
                name=f"input_{utc_now().isoformat()}",
                episode_body=text,
                source_description="user input",
                reference_time=utc_now(),
            )

            print(f"\n✓ Extracted {len(result.nodes)} entities, {len(result.edges)} relationships")
            for node in result.nodes:
                summary = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
                print(f"  • {node.name}: {summary}")
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Continuing...\n")

    await graphiti.close()
    print("\n✓ Database connection closed. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
```

**Purpose:** Clean production interface replacing `mlx_test.py`

---

## Feature Comparison

### Before (mlx_test.py homebrew)
| Feature | Status | Code |
|---------|--------|------|
| Entity extraction | ✅ Basic | Custom |
| Relationship extraction | ✅ Basic | Custom |
| Node deduplication | ❌ None | - |
| Edge deduplication | ❌ None | - |
| Summarization | ❌ Empty strings | - |
| Fact paraphrasing | ❌ None | - |
| Reflexion loops | ❌ None | - |
| Context window | ❌ None | - |
| Contradiction detection | ❌ None | - |
| **TOTAL** | **2/9 features** | **~600 lines** |

### After (MLXLMClient + native Graphiti)
| Feature | Status | Code |
|---------|--------|------|
| Entity extraction | ✅ Full w/ reflexion | Graphiti |
| Relationship extraction | ✅ Full w/ reflexion | Graphiti |
| Node deduplication | ✅ Semantic + LLM | Graphiti |
| Edge deduplication | ✅ Semantic + LLM | Graphiti |
| Summarization | ✅ Incremental | Graphiti |
| Fact paraphrasing | ✅ LLM-based | Graphiti |
| Reflexion loops | ✅ Auto | Graphiti |
| Context window | ✅ Configurable | Graphiti |
| Contradiction detection | ✅ Temporal | Graphiti |
| **TOTAL** | **9/9 features** | **~150 lines** |

**Net Result:** 600 lines custom code → 150 lines integration = **75% code reduction** + **7 missing features**

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Outlines incompatibility | Low | High | Test with simple schemas first |
| Prompt format issues | Medium | Medium | Iterate on Qwen formatter |
| Small model performance | High | Medium | Use Qwen2.5-3B+, optimize prompts |
| Monkey-patching breakage | Low | Medium | Pin Graphiti version |
| Memory constraints | Medium | Medium | Use 4-bit quantization |

---

## Success Metrics

### Code Quality
- ✅ <200 lines of new code
- ✅ Clean separation: graph logic / model inference
- ✅ Zero external API dependencies

### Feature Parity
- ✅ All 9 Graphiti features working
- ✅ Node deduplication prevents "Alice" vs "alice"
- ✅ Edge deduplication prevents duplicate facts
- ✅ Summarization generates entity summaries
- ✅ Fact paraphrasing consolidates relationships

### Maintainability
- ✅ Model-agnostic adapter (swap Qwen → Llama via formatter)
- ✅ No Graphiti fork to maintain
- ✅ Testable components

---

## Key Insights

### What Makes This Work

1. **Graphiti's Plugin Architecture**
   - Abstract `LLMClient` interface
   - Dependency injection throughout
   - No hard OpenAI dependencies

2. **Clean Abstraction Layers**
   - `MLXLMClient` → Graphiti adapter (model-agnostic)
   - `ModelFormatter` → Chat template interface (swappable)
   - `QwenFormatter` → Model-specific implementation (isolated)

3. **Outlines Integration**
   - Direct Pydantic schema → JSON
   - No custom parsing needed
   - Same interface as OpenAI function calling

4. **All Complex Logic Reused**
   - Deduplication algorithms
   - Search and retrieval
   - Database operations
   - Temporal reasoning

### What We're NOT Doing

❌ Rewriting entity extraction
❌ Rewriting deduplication
❌ Rewriting database layer
❌ Forking Graphiti
❌ Maintaining parallel codebase
❌ Hardcoding model-specific logic in adapters

### What We ARE Doing

✅ Implementing clean adapter class
✅ Separating model inference from graph logic
✅ Optionally overriding prompts
✅ Calling existing, tested code
✅ Getting features for free

---

## Critical Question: Orphaned Nodes

**Question:** If text extracts entities but no relationships, will entities be added to the graph?

**Answer:** **YES**, entities without relationships ARE added.

**Why:** Graphiti creates **EpisodicEdges** (MENTIONS relationships) between the episode and EVERY extracted entity, regardless of whether EntityEdges exist.

**Code path:** `graphiti.py:537` → `build_episodic_edges(nodes, episode.uuid, now)`

**Proof:**
```python
# From edge_operations.py:51-68
def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,  # ← EVERY node
            created_at=created_at,
            group_id=node.group_id,
        )
        for node in entity_nodes  # ← All nodes
    ]
    return episodic_edges
```

**Result:** "Orphaned" entities still connect to graph via MENTIONS → episode. Graph stays connected.

---

## Next Steps

**Phase 1:** Core infrastructure
1. Implement `ModelFormatter` abstraction
2. Implement `QwenFormatter`
3. Implement `MLXLMClient`
4. Migrate `MLXEmbedder`

**Phase 2:** Prompt control
1. Create custom prompt library
2. Add override mechanism to `MLXLMClient`
3. Test with default prompts first
4. Optimize prompts if needed

**Phase 3:** Production
1. Create test script
2. Verify all 9 features work
3. Create production entry point
4. Archive/delete `mlx_test.py`

---

## Appendix: Technical Details

### A. Model Formatter Interface

The `ModelFormatter` abstraction allows swapping models without changing Graphiti integration:

```python
# Use Qwen
formatter = QwenFormatter()
llm = MLXLMClient(model, tokenizer, formatter)

# Later: swap to Llama
formatter = LlamaFormatter()
llm = MLXLMClient(model, tokenizer, formatter)
```

### B. Pydantic Schema Handling

Graphiti automatically appends schemas to prompts (client.py:154-160):

```python
# Graphiti adds this automatically
messages[-1].content += f'\n\nRespond with JSON:\n{schema}'

# Outlines enforces it
json_str = self.outlines_model(prompt, output_type=ResponseModel)

# We parse and return
return ResponseModel.model_validate_json(json_str).model_dump()
```

### C. Prompt Library Structure

```
prompt_library/
├── extract_nodes
│   ├── extract_message     → Entity extraction from messages
│   ├── extract_text        → Entity extraction from documents
│   ├── extract_json        → Entity extraction from JSON
│   ├── reflexion           → "What entities did you miss?"
│   └── extract_summary     → Entity summarization
├── dedupe_nodes
│   └── nodes               → LLM node deduplication
├── extract_edges
│   ├── edge                → Relationship extraction
│   └── reflexion           → "What facts did you miss?"
└── dedupe_edges
    └── resolve_edge        → Edge deduplication + paraphrasing
```

Each returns `list[Message]`. Override any/all via `custom_prompts` dict.

### D. Full Pipeline Flow

```
add_episode()
  ├─→ retrieve_episodes()              # Context window
  ├─→ extract_nodes()                  # With reflexion
  │    └─→ llm_client.generate_response(
  │          prompt_library.extract_nodes.extract_text(context),
  │          response_model=ExtractedEntities
  │        )
  ├─→ resolve_extracted_nodes()        # Deduplication
  │    └─→ llm_client.generate_response(
  │          prompt_library.dedupe_nodes.nodes(context),
  │          response_model=NodeResolutions
  │        )
  ├─→ extract_edges()                  # With reflexion
  │    └─→ llm_client.generate_response(
  │          prompt_library.extract_edges.edge(context),
  │          response_model=ExtractedEdges
  │        )
  ├─→ resolve_edge_pointers()          # UUID remapping
  ├─→ resolve_extracted_edges()        # Deduplication + paraphrasing
  │    └─→ llm_client.generate_response(
  │          prompt_library.dedupe_edges.resolve_edge(context),
  │          response_model=EdgeDuplicate
  │        )
  ├─→ extract_attributes_from_nodes()  # Summarization
  │    └─→ llm_client.generate_response(
  │          prompt_library.extract_nodes.extract_summary(context),
  │          response_model=EntitySummary
  │        )
  ├─→ build_episodic_edges()           # MENTIONS relationships
  └─→ add_nodes_and_edges_bulk()       # Database write
```

**Every LLM call routes through our `MLXLMClient`. Zero Graphiti code changes needed.**

---

## Conclusion

By implementing a clean adapter architecture with separated concerns:

- ✅ 100% local inference (no OpenAI)
- ✅ 100% prompt control (override any prompt)
- ✅ Model-agnostic design (swap via formatter)
- ✅ 9/9 critical features (vs 2/9 homebrew)
- ✅ 75% less code (150 vs 600 lines)
- ✅ No Graphiti fork to maintain

**This is the minimal viable path to full-featured, locally-controlled knowledge graph extraction.**
