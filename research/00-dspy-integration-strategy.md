# DSPy Integration Strategy for Graphiti Prompts

## Overview

Graphiti uses a versioned prompt library system that can be replaced with DSPy signatures. This document outlines the strategy for integrating DSPy modules to replace Graphiti's default prompting.

## Current Prompt Architecture

### Prompt Library Structure (`graphiti_core/prompts/lib.py`)

```python
# Current implementation
PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    'extract_nodes': extract_nodes_versions,
    'dedupe_nodes': dedupe_nodes_versions,
    'extract_edges': extract_edges_versions,
    'dedupe_edges': dedupe_edges_versions,
    'invalidate_edges': invalidate_edges_versions,
    'extract_edge_dates': extract_edge_dates_versions,
    'summarize_nodes': summarize_nodes_versions,
    'eval': eval_versions,
}

prompt_library: PromptLibrary = PromptLibraryWrapper(PROMPT_LIBRARY_IMPL)
```

### How Prompts Are Called

All prompts are invoked through `llm_client.generate_response()`:

```python
# Example from node_operations.py:134-139
llm_response = await llm_client.generate_response(
    prompt_library.extract_nodes.extract_message(context),
    response_model=ExtractedEntities,
    group_id=episode.group_id,
    prompt_name='extract_nodes.extract_message',
)
```

## Prompt Inventory

### 1. Node Extraction Prompts
**Location**: `graphiti_core/prompts/extract_nodes.py`

| Function | Purpose | Response Model | DSPy Signature Mapping |
|----------|---------|----------------|------------------------|
| `extract_message` | Extract entities from conversations | `ExtractedEntities` | Similar to our `kg_extraction.py` |
| `extract_text` | Extract entities from documents | `ExtractedEntities` | Text-based extraction |
| `extract_json` | Extract entities from structured data | `ExtractedEntities` | JSON parsing + extraction |
| `reflexion` | Second-pass entity checking | `MissedEntities` | Self-correction signature |
| `classify_nodes` | Classify entity types | `EntityClassification` | Classification signature |
| `extract_attributes` | Extract entity attributes | Custom per entity type | Attribute extraction signature |
| `extract_summary` | Generate entity summaries | `EntitySummary` | Summarization signature |

### 2. Edge Extraction Prompts
**Location**: `graphiti_core/prompts/extract_edges.py`

| Function | Purpose | Response Model | DSPy Signature Mapping |
|----------|---------|----------------|------------------------|
| `edge` | Extract relationships | `ExtractedEdges` | Relationship extraction |
| `reflexion` | Second-pass fact checking | `MissingFacts` | Self-correction for edges |
| `extract_attributes` | Extract edge attributes | Custom per edge type | Edge attribute extraction |

### 3. Node Deduplication Prompts
**Location**: `graphiti_core/prompts/dedupe_nodes.py`

| Function | Purpose | Response Model |
|----------|---------|----------------|
| `node` | Compare single entity | `NodeDuplicate` |
| `nodes` | Batch entity comparison | `NodeResolutions` |
| `node_list` | Dedupe existing entities | Custom |

### 4. Edge Deduplication Prompts
**Location**: `graphiti_core/prompts/dedupe_edges.py`

| Function | Purpose | Response Model |
|----------|---------|----------------|
| `edge` | Compare single fact | `EdgeDuplicate` |
| `edge_list` | Batch fact comparison | Custom |
| `resolve_edge` | Full edge resolution with contradiction detection | Custom |

### 5. Temporal Prompts
**Location**: `graphiti_core/prompts/extract_edge_dates.py`, `invalidate_edges.py`

| Function | Purpose | Response Model |
|----------|---------|----------------|
| `extract_edge_dates.v1` | Extract temporal bounds | `EdgeDates` |
| `invalidate_edges.v2` | Detect contradictions | `InvalidatedEdges` |

## DSPy Integration Approach

### Strategy 1: LLMClient Wrapper (RECOMMENDED)

**Advantage**: No modification to graphiti code needed

### NER-Orchestrated Context

We must accommodate multiple roles for the local DistilBERT ONNX model (`distilbert_ner.py`) when replacing Graphiti prompts. When `llm_client.generate_response()` is invoked for entity or edge extraction prompts, the wrapper should:

1. Support configurable modes: NER-only (DistilBERT produces entities directly), hybrid (NER detections plus DSPy validation/augmentation), and DSPy-only with optional hints.
2. Run `distilbert_ner.predict_entities()` when the selected mode requires it (respecting stride/max length limits) and normalize results to the expected response models.
3. Expose runtime flags/telemetry so we can benchmark each mode (latency, recall, disagreement rates) and switch strategies without code changes.

This flexibility lets us use DistilBERT as a steering signal or as the primary extractor depending on quality targets, aligning with the existing Gradio fusion experience (`gradio_app.py`).

Create a custom `LLMClient` that intercepts `generate_response()` and routes to DSPy:

```python
from graphiti_core.llm_client import LLMClient
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

class DSPyLLMClient(LLMClient):
    def __init__(self):
        # Configure DSPy with our technology
        dspy.configure(
            lm=OutlinesLM(),
            adapter=OutlinesAdapter()
        )

        # Map prompt names to DSPy modules
        self.prompt_modules = {
            'extract_nodes.extract_message': ExtractEntitiesModule(),
            'extract_edges.edge': ExtractRelationshipsModule(),
            'dedupe_nodes.nodes': DedupeNodesModule(),
            # ... etc
        }

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        prompt_name: str = None,
        **kwargs
    ):
        # Get DSPy module for this prompt
        module = self.prompt_modules.get(prompt_name)

        if module:
            # Extract context from messages
            context = self._parse_context_from_messages(messages)

            # Execute DSPy module
            result = module(**context)

            # Convert to expected format
            return result.model_dump()
        else:
            # Fallback to default behavior
            return await super().generate_response(
                messages, response_model, **kwargs
            )
```

### Strategy 2: Prompt Library Replacement

**Advantage**: More explicit control

Replace the prompt library implementation:

```python
# Create custom prompt library
from graphiti_core.prompts.models import Message

def dspy_extract_message(context: dict[str, Any]) -> list[Message]:
    """
    This function signature matches graphiti's expectations,
    but we'll intercept it in the LLMClient
    """
    # Return placeholder - actual DSPy execution happens in LLMClient
    return [
        Message(role='system', content='DSPy Module'),
        Message(role='user', content=json.dumps(context))
    ]

# Override default library
from graphiti_core.prompts import lib
lib.PROMPT_LIBRARY_IMPL['extract_nodes']['extract_message'] = dspy_extract_message
```

## DSPy Module Implementations

### Example: Entity Extraction Module

```python
import dspy
from pydantic import BaseModel, Field

class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(description='ID of entity type')

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField(
        desc="Current episode text to extract entities from"
    )
    previous_episodes: list[str] = dspy.InputField(
        desc="Previous episodes for context",
        default_factory=list
    )
    entity_types: list[dict] = dspy.InputField(
        desc="Available entity types with descriptions"
    )

    extracted_entities: ExtractedEntities = dspy.OutputField(
        desc="List of extracted entities with type classifications"
    )

class ExtractEntitiesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractEntitiesSignature)

    def forward(self, episode_content, previous_episodes, entity_types, **kwargs):
        return self._predict(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_types=entity_types
        )
```

### Example: Relationship Extraction Module

```python
class ExtractedEdge(BaseModel):
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    fact: str
    valid_at: str | None = None
    invalid_at: str | None = None

class ExtractedEdges(BaseModel):
    edges: list[ExtractedEdge]

class ExtractRelationshipsSignature(dspy.Signature):
    """Extract relationships between entities."""

    episode_content: str = dspy.InputField()
    nodes: list[dict] = dspy.InputField(
        desc="Extracted entities with IDs and names"
    )
    reference_time: str = dspy.InputField(
        desc="ISO timestamp for temporal resolution"
    )
    edge_types: list[dict] = dspy.InputField(
        desc="Available relationship types",
        default_factory=list
    )

    edges: ExtractedEdges = dspy.OutputField(
        desc="List of relationships with temporal bounds"
    )

class ExtractRelationshipsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractRelationshipsSignature)

    def forward(self, episode_content, nodes, reference_time, edge_types, **kwargs):
        return self._predict(
            episode_content=episode_content,
            nodes=nodes,
            reference_time=reference_time,
            edge_types=edge_types or []
        )
```

## Implementation Plan

### Phase 1: Core Extraction Modules
1. `ExtractEntitiesModule` (for all three episode types)
2. `ExtractRelationshipsModule`
3. `DedupeNodesModule`
4. `DedupeEdgesModule`

### Phase 2: Refinement Modules
5. `ReflexionNodesModule` (entity self-correction)
6. `ReflexionEdgesModule` (relationship self-correction)
7. `ClassifyNodesModule`
8. `ExtractAttributesModule`

### Phase 3: Advanced Modules
9. `ExtractSummaryModule`
10. `ExtractEdgeDatesModule`
11. `InvalidateEdgesModule`

## Integration with Our Technology Stack

### Using MLX + Outlines Constrained Generation

```python
# In custom pipeline initialization
from dspy_outlines import OutlinesLM, OutlinesAdapter

lm = OutlinesLM(
    model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
    max_tokens=4096
)

adapter = OutlinesAdapter()

dspy.configure(lm=lm, adapter=adapter)

# Create DSPy-enabled Graphiti client
graphiti = Graphiti(
    uri="falkordb://localhost:6379",
    llm_client=DSPyLLMClient(),
    embedder=embedder,
    graph_driver=FalkorDriver(...)
)
```

## Critical Integration Points

### 1. Context Extraction
Graphiti passes context via `dict[str, Any]`. DSPy signatures need typed inputs:

```python
def _parse_context_from_messages(self, messages: list[Message]) -> dict:
    """Extract structured context from graphiti's message format."""
    # Parse user message content (usually contains context as formatted text)
    user_msg = next(m for m in messages if m.role == 'user')

    # Context is often in XML-like tags, e.g.:
    # <PREVIOUS MESSAGES>...</PREVIOUS MESSAGES>
    # <CURRENT MESSAGE>...</CURRENT MESSAGE>

    return extract_structured_context(user_msg.content)
```

### 2. Response Model Compatibility
Graphiti expects Pydantic models that match their response schemas. DSPy signatures must output compatible models.

### 3. Model Size Selection
Graphiti uses `ModelSize.small` for certain operations (deduplication). Map this to appropriate MLX model:

```python
MODEL_SIZE_MAP = {
    ModelSize.small: "mlx-community/Qwen2.5-3B-Instruct-4bit",
    ModelSize.large: "mlx-community/Qwen2.5-7B-Instruct-4bit",
}
```

## Testing Strategy

1. **Unit Tests**: Test each DSPy module independently
2. **Integration Tests**: Test with graphiti's pipeline using test episodes
3. **Comparison Tests**: Compare DSPy outputs vs default prompts
4. **Performance Tests**: Measure speed improvements with MLX

## Advantages of This Approach

1. **No Graphiti Modifications**: Works with stock graphiti-core
2. **Gradual Migration**: Implement one prompt at a time
3. **Fallback Support**: Can fall back to default prompts
4. **Type Safety**: DSPy signatures provide better type checking
5. **Optimization**: Can use DSPy's optimization features (MIPRO, etc.)
6. **Small Model Friendly**: Constrained generation works better with smaller models
