# Graphiti-Core Integration Plan: Custom DSPy Pipeline with FalkorDB

## 1. Executive Summary

### Overview of Graphiti-Core Architecture

Graphiti-core is a knowledge graph extraction system that processes **episodes** (messages, text, or JSON) into a graph of **entities** (EntityNode) and **relationships** (EntityEdge). The system uses:

- **EpisodicNode**: Raw input data with temporal metadata
- **EntityNode**: Extracted entities with embeddings, summaries, and attributes
- **EntityEdge**: Relationships between entities with temporal bounds and facts
- **EpisodicEdge**: MENTIONS relationships linking episodes to entities
- **CommunityNode/Edge**: Optional hierarchical entity groupings

The default pipeline uses OpenAI-style LLM clients with prompt templates to orchestrate extraction, deduplication, attribute extraction, and temporal reasoning.

### Custom Pipeline Goals

Our goal is to create a **custom knowledge graph extraction pipeline** that:

1. **Bypasses graphiti's automated episode processing** to gain full control over extraction steps
2. **Replaces all LLM operations with DSPy modules** using MLX + Outlines for constrained generation
3. **Reuses graphiti's Pydantic models** (EntityNode, EntityEdge, etc.) for data representation
4. **Uses FalkorDB as the graph backend** instead of Neo4j/Neptune
5. **Maintains compatibility** with graphiti's graph query patterns and bulk operations

This approach gives us:
- **Type-safe extraction** through Outlines constrained generation
- **Local inference** via MLX on Apple Silicon
- **Optimization capability** through DSPy's MIPRO
- **Full pipeline control** while leveraging battle-tested data models

### Integration Approach

**Strategy: Custom Pipeline with Graphiti Models + DSPy LLMClient Wrapper**

We will create a **custom ingestion pipeline** that:
- Uses graphiti's data models (EntityNode, EntityEdge, EpisodicNode) directly
- Implements DSPy modules for all extraction operations
- Wraps DSPy modules in a custom LLMClient for seamless integration
- Uses graphiti's bulk save utilities for FalkorDB persistence
- Bypasses `add_episode()` automation for granular control

This hybrid approach allows us to:
- Reuse graphiti's robust data models and database operations
- Replace LLM operations with optimized DSPy modules
- Maintain flexibility to add custom processing steps
- Gradually migrate functionality without forking graphiti

---

## 2. Data Flow Architecture

### Complete Flow: Episode → FalkorDB

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: Raw Episode Data                                                  │
│   - episode_body: str (message, text, or JSON)                           │
│   - reference_time: datetime (temporal anchor)                           │
│   - source: EpisodeType (message|text|json)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Episode Context Retrieval                                        │
│   - Retrieve previous N episodes (default N=3, EPISODE_WINDOW_LEN)       │
│   - Provides context for entity extraction and deduplication             │
│   - Uses FalkorDB queries via GraphDriver                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EpisodicNode Creation                                            │
│   - Create EpisodicNode object with metadata                             │
│   - Fields: uuid, name, content, source, valid_at, group_id              │
│   - Not saved yet - used as input to extraction                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Entity Extraction (DSPy Module)                                  │
│   INPUT:                                                                  │
│     - Current episode content                                             │
│     - Previous episodes (for context)                                     │
│     - Entity types (optional classification schema)                       │
│   DSPy MODULE: ExtractEntitiesModule                                      │
│   OUTPUT:                                                                 │
│     - List[EntityNode] with name, labels, uuid, group_id                  │
│     - No embeddings yet, no summary                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Entity Deduplication (DSPy Module)                               │
│   INPUT:                                                                  │
│     - Extracted entities from Step 3                                      │
│     - Previous episodes for context                                       │
│     - Existing entities in graph (via embedding search)                   │
│   DSPy MODULE: DedupeNodesModule                                          │
│   OUTPUT:                                                                 │
│     - deduplicated_nodes: List[EntityNode]                                │
│     - uuid_map: Dict[str, str] (maps extracted → resolved UUIDs)          │
│   PROCESS:                                                                │
│     1. For each extracted entity, search existing graph                   │
│     2. LLM determines if match exists                                     │
│     3. If duplicate: merge and map UUID                                   │
│     4. If new: keep original UUID                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Relationship Extraction (DSPy Module)                            │
│   INPUT:                                                                  │
│     - Episode content                                                     │
│     - Deduplicated entities (with resolved UUIDs)                         │
│     - Reference time (for temporal reasoning)                             │
│     - Edge types (optional relationship schema)                           │
│   DSPy MODULE: ExtractRelationshipsModule                                 │
│   OUTPUT:                                                                 │
│     - List[EntityEdge] with:                                              │
│       - source_node_uuid, target_node_uuid                                │
│       - name (relationship type)                                          │
│       - fact (textual description)                                        │
│       - valid_at, invalid_at (temporal bounds)                            │
│       - No embeddings yet                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Relationship Deduplication (DSPy Module)                         │
│   INPUT:                                                                  │
│     - Extracted edges                                                     │
│     - Existing edges between same entity pairs                            │
│   DSPy MODULE: DedupeEdgesModule                                          │
│   OUTPUT:                                                                 │
│     - resolved_edges: List[EntityEdge] (new or updated)                   │
│     - invalidated_edges: List[EntityEdge] (contradicted facts)            │
│   PROCESS:                                                                │
│     1. Compare new facts to existing facts                                │
│     2. Detect contradictions                                              │
│     3. Set invalid_at for contradicted edges                              │
│     4. Merge duplicate facts                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Attribute Extraction (DSPy Module - Optional)                    │
│   INPUT:                                                                  │
│     - Deduplicated entities                                               │
│     - Episode content                                                     │
│     - Entity type schemas (with attribute definitions)                    │
│   DSPy MODULE: ExtractAttributesModule                                    │
│   OUTPUT:                                                                 │
│     - Entities with populated attributes dict                             │
│   EXAMPLE:                                                                │
│     EntityNode(name="John", labels=["Person"],                            │
│                attributes={"age": 30, "occupation": "Engineer"})          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 8: Embedding Generation                                             │
│   - Generate name_embedding for each EntityNode                           │
│   - Generate fact_embedding for each EntityEdge                           │
│   - Uses graphiti's EmbedderClient (BGE, OpenAI, etc.)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 9: Episodic Edge Creation                                           │
│   - Create EpisodicEdge (MENTIONS relationship)                           │
│   - Links EpisodicNode → EntityNode for each extracted entity             │
│   - Tracks which episode mentioned which entity                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 10: Bulk Save to FalkorDB                                           │
│   FUNCTION: add_nodes_and_edges_bulk(driver, ...)                         │
│   SAVES:                                                                  │
│     - EpisodicNode (with content, metadata)                               │
│     - EntityNode[] (with embeddings, attributes)                          │
│     - EpisodicEdge[] (MENTIONS relationships)                             │
│     - EntityEdge[] (RELATES_TO relationships)                             │
│   DATABASE OPERATIONS:                                                    │
│     1. Episode save query                                                 │
│     2. Entity merge queries (upsert by UUID)                              │
│     3. Edge merge queries (upsert by UUID)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: FalkorDB Graph                                                    │
│   - Episodic nodes with temporal metadata                                │
│   - Entity nodes with embeddings and attributes                          │
│   - MENTIONS edges (episode → entity)                                    │
│   - RELATES_TO edges (entity → entity)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Intermediate Processing Steps

**Context Retrieval** (before extraction):
- Query: `MATCH (e:Episodic {group_id: $group_id}) RETURN e ORDER BY e.valid_at DESC LIMIT 3`
- Provides previous episodes as context for extraction and deduplication

**UUID Mapping** (between extraction and edge creation):
- Maps temporary extraction UUIDs to resolved entity UUIDs
- Critical for referential integrity when entities are merged

**Embedding Search** (during deduplication):
- Query existing entities by embedding similarity
- Find potential duplicates for LLM comparison
- Uses vector search on name_embedding field

### Where DSPy Modules Replace Graphiti Prompts

| Graphiti Prompt | DSPy Module | Purpose |
|-----------------|-------------|---------|
| `extract_nodes.extract_message` | `ExtractEntitiesModule` | Extract entities from conversations |
| `extract_nodes.extract_text` | `ExtractEntitiesModule` | Extract entities from documents |
| `extract_nodes.extract_json` | `ExtractEntitiesModule` | Extract entities from JSON |
| `dedupe_nodes.nodes` | `DedupeNodesModule` | Batch entity deduplication |
| `extract_edges.edge` | `ExtractRelationshipsModule` | Extract relationships |
| `dedupe_edges.edge` | `DedupeEdgesModule` | Edge deduplication |
| `invalidate_edges.v2` | `InvalidateEdgesModule` | Detect contradictions |
| `extract_attributes` | `ExtractAttributesModule` | Extract typed attributes |
| `extract_edge_dates` | `ExtractTemporalModule` | Extract temporal bounds |

---

## 3. Pydantic Models to Reuse

### Graphiti Models We Use Directly

#### EntityNode (graphiti_core/nodes.py:435-441)
```python
class EntityNode(Node):
    name_embedding: list[float] | None = None
    summary: str = ""
    attributes: dict[str, Any] = {}

    # Inherited from Node:
    uuid: str
    name: str
    group_id: str
    labels: list[str]
    created_at: datetime
```

**Usage**: Main entity representation. Our DSPy modules output these directly.

#### EntityEdge (graphiti_core/edges.py:221-240)
```python
class EntityEdge(Edge):
    name: str  # Relationship type
    fact: str  # Textual description
    fact_embedding: list[float] | None = None
    episodes: list[str] = []  # Episode UUIDs
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    attributes: dict[str, Any] = {}

    # Inherited from Edge:
    uuid: str
    group_id: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime
```

**Usage**: Relationship representation. Our DSPy modules output these.

#### EpisodicNode (graphiti_core/nodes.py:295-299)
```python
class EpisodicNode(Node):
    source: EpisodeType  # message, text, or json
    source_description: str
    content: str  # Raw episode data
    valid_at: datetime  # When original document was created
    entity_edges: list[str] = []  # UUIDs of entity edges
```

**Usage**: We create these to represent input episodes.

#### EpisodicEdge (graphiti_core/edges.py:131-144)
```python
class EpisodicEdge(Edge):
    # MENTIONS relationship from episode to entity
    # source_node_uuid: episode UUID
    # target_node_uuid: entity UUID
```

**Usage**: Generated automatically from extracted entities.

### Mappings Between kg_extraction.py and Graphiti Models

Our current `/Users/flavius/repos/charlie/dspy_outlines/kg_extraction.py`:

| Our Model | Graphiti Model | Mapping Strategy |
|-----------|----------------|------------------|
| `Node` | `EntityNode` | Replace with EntityNode |
| `Edge` | `EntityEdge` | Replace with EntityEdge |
| `KnowledgeGraph` | N/A (wrapper) | Use List[EntityNode], List[EntityEdge] instead |

**Current kg_extraction.py Node**:
```python
class Node(BaseModel):
    id: int
    label: str  # Entity name
    properties: dict
```

**Mapped to EntityNode**:
```python
EntityNode(
    uuid=str(uuid4()),  # Generate proper UUID
    name=node.label,    # Map label → name
    labels=["Entity"],  # Or extract entity type
    group_id=group_id,
    attributes=node.properties,  # Map properties → attributes
    created_at=utc_now()
)
```

**Current kg_extraction.py Edge**:
```python
class Edge(BaseModel):
    source: int  # Node ID
    target: int  # Node ID
    label: str   # Relationship type
    properties: dict
```

**Mapped to EntityEdge**:
```python
EntityEdge(
    uuid=str(uuid4()),
    source_node_uuid=uuid_map[edge.source],  # Resolve ID → UUID
    target_node_uuid=uuid_map[edge.target],
    name=edge.label,  # Map label → name
    fact=f"{source_name} {edge.label} {target_name}",  # Generate fact
    group_id=group_id,
    created_at=utc_now(),
    attributes=edge.properties
)
```

### Required Model Adaptations

**1. ID Resolution**
- Our models use integer IDs, graphiti uses UUID strings
- Need UUID mapping: `Dict[int, str]` during conversion

**2. Fact Generation**
- Graphiti EntityEdge requires a `fact` field (textual description)
- Generate from: `"{source.name} {edge.name} {target.name}"`
- Or extract from text via DSPy

**3. Temporal Information**
- Add `valid_at`, `invalid_at` extraction
- Use episode reference_time as default

**4. Entity Type Classification**
- Our `Node.label` is entity name
- Graphiti `EntityNode.labels` is type classification (e.g., ["Person"], ["Organization"])
- Add entity type classification step

---

## 4. Custom Pipeline Design

### Step-by-Step Custom Ingestion Flow

```python
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver
from datetime import datetime
from uuid import uuid4

async def custom_ingestion_pipeline(
    episode_body: str,
    reference_time: datetime,
    group_id: str,
    driver: FalkorDriver,
    embedder: EmbedderClient,
    dspy_modules: dict  # Our DSPy modules
):
    """
    Custom knowledge graph ingestion pipeline using graphiti models + DSPy.
    """

    # ============================================================
    # STEP 1: Retrieve Previous Episodes for Context
    # ============================================================
    previous_episodes = await retrieve_previous_episodes(
        driver=driver,
        group_id=group_id,
        reference_time=reference_time,
        limit=3  # EPISODE_WINDOW_LEN
    )

    # ============================================================
    # STEP 2: Create EpisodicNode
    # ============================================================
    episode = EpisodicNode(
        uuid=str(uuid4()),
        name=f"Episode {reference_time.isoformat()}",
        group_id=group_id,
        source=EpisodeType.message,
        source_description="User conversation",
        content=episode_body,
        valid_at=reference_time,
        created_at=utc_now(),
        labels=[]
    )

    # ============================================================
    # STEP 3: Extract Entities (DSPy)
    # ============================================================
    extract_module = dspy_modules['extract_entities']
    extraction_result = extract_module(
        episode_content=episode_body,
        previous_episodes=[ep.content for ep in previous_episodes],
        entity_types=[]  # Optional schema
    )

    # Convert DSPy output to EntityNode objects
    extracted_nodes = [
        EntityNode(
            uuid=str(uuid4()),
            name=entity.name,
            group_id=group_id,
            labels=entity.labels or ["Entity"],
            created_at=utc_now(),
            summary="",
            attributes=entity.attributes or {}
        )
        for entity in extraction_result.entities
    ]

    # ============================================================
    # STEP 4: Deduplicate Entities (DSPy)
    # ============================================================
    dedupe_module = dspy_modules['dedupe_nodes']

    # For each extracted entity, search for existing matches
    resolved_nodes = []
    uuid_map = {}  # Maps extracted UUID → resolved UUID

    for node in extracted_nodes:
        # Search existing graph by embedding similarity
        candidates = await search_similar_entities(
            driver=driver,
            embedder=embedder,
            entity_name=node.name,
            group_id=group_id,
            top_k=5
        )

        if candidates:
            # Use DSPy to determine if match exists
            dedupe_result = dedupe_module(
                extracted_entity=node.name,
                candidate_entities=[c.name for c in candidates],
                context=episode_body
            )

            if dedupe_result.is_duplicate:
                # Merge with existing entity
                matched = candidates[dedupe_result.match_index]
                uuid_map[node.uuid] = matched.uuid

                # Update existing entity (merge attributes, etc.)
                merged = merge_entity_nodes(matched, node)
                resolved_nodes.append(merged)
            else:
                # New entity
                uuid_map[node.uuid] = node.uuid
                resolved_nodes.append(node)
        else:
            # No candidates, definitely new
            uuid_map[node.uuid] = node.uuid
            resolved_nodes.append(node)

    # ============================================================
    # STEP 5: Extract Relationships (DSPy)
    # ============================================================
    edges_module = dspy_modules['extract_relationships']
    edges_result = edges_module(
        episode_content=episode_body,
        nodes=[{"uuid": n.uuid, "name": n.name} for n in resolved_nodes],
        reference_time=reference_time.isoformat(),
        edge_types=[]  # Optional schema
    )

    # Convert to EntityEdge objects
    extracted_edges = [
        EntityEdge(
            uuid=str(uuid4()),
            source_node_uuid=uuid_map[edge.source_uuid],
            target_node_uuid=uuid_map[edge.target_uuid],
            name=edge.relation_type,
            fact=edge.fact,
            group_id=group_id,
            created_at=utc_now(),
            valid_at=parse_datetime(edge.valid_at) if edge.valid_at else reference_time,
            invalid_at=parse_datetime(edge.invalid_at) if edge.invalid_at else None,
            episodes=[episode.uuid]
        )
        for edge in edges_result.edges
    ]

    # ============================================================
    # STEP 6: Deduplicate Edges (DSPy)
    # ============================================================
    dedupe_edges_module = dspy_modules['dedupe_edges']

    resolved_edges = []
    invalidated_edges = []

    for edge in extracted_edges:
        # Find existing edges between same node pair
        existing = await find_edges_between_nodes(
            driver=driver,
            source_uuid=edge.source_node_uuid,
            target_uuid=edge.target_node_uuid,
            group_id=group_id
        )

        if existing:
            dedupe_result = dedupe_edges_module(
                new_fact=edge.fact,
                existing_facts=[e.fact for e in existing],
                context=episode_body
            )

            if dedupe_result.is_contradiction:
                # Mark contradicted edge as invalid
                contradicted = existing[dedupe_result.contradicts_index]
                contradicted.invalid_at = reference_time
                invalidated_edges.append(contradicted)
                resolved_edges.append(edge)
            elif dedupe_result.is_duplicate:
                # Update existing edge
                matched = existing[dedupe_result.match_index]
                matched.episodes.append(episode.uuid)
                resolved_edges.append(matched)
            else:
                # New edge
                resolved_edges.append(edge)
        else:
            # No existing edges
            resolved_edges.append(edge)

    # ============================================================
    # STEP 7: Generate Embeddings
    # ============================================================
    for node in resolved_nodes:
        if not node.name_embedding:
            await node.generate_name_embedding(embedder)

    for edge in resolved_edges + invalidated_edges:
        if not edge.fact_embedding:
            await edge.generate_embedding(embedder)

    # ============================================================
    # STEP 8: Create Episodic Edges (MENTIONS)
    # ============================================================
    episodic_edges = [
        EpisodicEdge(
            uuid=str(uuid4()),
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            group_id=group_id,
            created_at=utc_now()
        )
        for node in resolved_nodes
    ]

    # Update episode with entity edge references
    episode.entity_edges = [e.uuid for e in resolved_edges + invalidated_edges]

    # ============================================================
    # STEP 9: Bulk Save to FalkorDB
    # ============================================================
    await add_nodes_and_edges_bulk(
        driver=driver,
        episodic_nodes=[episode],
        episodic_edges=episodic_edges,
        entity_nodes=resolved_nodes,
        entity_edges=resolved_edges + invalidated_edges,
        embedder=embedder
    )

    return {
        "episode": episode,
        "nodes": resolved_nodes,
        "edges": resolved_edges,
        "invalidated_edges": invalidated_edges
    }
```

### Control Points for Each Step

**1. Entity Extraction Control**
- Override entity type schema
- Filter extracted entities before deduplication
- Add custom entity classification logic

**2. Embedding Creation Control**
- Choose embedder (BGE, OpenAI, custom)
- Batch embedding generation
- Skip embeddings for testing

**3. Deduplication Control**
- Adjust similarity thresholds
- Override LLM deduplication decisions
- Implement custom merge strategies

**4. Custom Processing Hooks**
```python
# Add custom processing at any step
async def custom_ingestion_with_hooks(
    episode_body: str,
    hooks: dict = None
):
    hooks = hooks or {}

    # ... extraction ...

    if 'post_extraction' in hooks:
        extracted_nodes = await hooks['post_extraction'](extracted_nodes)

    # ... deduplication ...

    if 'post_deduplication' in hooks:
        resolved_nodes = await hooks['post_deduplication'](resolved_nodes)

    # ... etc
```

### How to Bypass Automated Episode Processing

**Don't call `graphiti.add_episode()`**. Instead:

1. **Import graphiti utilities** (not the Graphiti class):
```python
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver
```

2. **Create your own pipeline function** (as shown above)

3. **Call DSPy modules directly** instead of using LLMClient

4. **Use graphiti's save utilities** for database operations

This approach gives you:
- Full control over extraction order
- Ability to skip steps (e.g., no communities)
- Custom deduplication logic
- Direct access to intermediate results

### How to Use Graphiti Models with DSPy Modules

**Pattern 1: Output Graphiti Models Directly**
```python
from graphiti_core.nodes import EntityNode

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField()

    # Output graphiti model directly
    entities: list[EntityNode] = dspy.OutputField()
```

**Pattern 2: Use Intermediate Models + Conversion**
```python
class ExtractedEntity(BaseModel):
    """Intermediate model for extraction."""
    name: str
    entity_type: str
    attributes: dict = {}

class ExtractEntitiesSignature(dspy.Signature):
    episode_content: str = dspy.InputField()
    entities: list[ExtractedEntity] = dspy.OutputField()

# Convert after extraction
def convert_to_graphiti(entities: list[ExtractedEntity], group_id: str) -> list[EntityNode]:
    return [
        EntityNode(
            uuid=str(uuid4()),
            name=e.name,
            labels=[e.entity_type],
            group_id=group_id,
            attributes=e.attributes,
            created_at=utc_now()
        )
        for e in entities
    ]
```

**Recommendation**: Use Pattern 2 for cleaner DSPy module design, then convert to graphiti models in the pipeline.

---

## 5. DSPy Integration Plan

### Detailed LLMClient Wrapper Implementation

The LLMClient wrapper allows graphiti code to call DSPy modules transparently.

```python
from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter
import json
from typing import Any

class DSPyLLMClient(LLMClient):
    """
    Custom LLMClient that routes graphiti prompts to DSPy modules.

    This allows graphiti code to call DSPy modules without modification.
    """

    def __init__(
        self,
        dspy_modules: dict[str, dspy.Module],
        config: LLMConfig | None = None,
        cache: bool = False
    ):
        super().__init__(config=config, cache=cache)

        # Configure DSPy
        lm = OutlinesLM(
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            max_tokens=4096
        )
        adapter = OutlinesAdapter()
        dspy.configure(lm=lm, adapter=adapter)

        # Map prompt names to DSPy modules
        self.dspy_modules = dspy_modules

        # Context parsers for each prompt type
        self.context_parsers = {
            'extract_nodes.extract_message': self._parse_node_extraction_context,
            'extract_edges.edge': self._parse_edge_extraction_context,
            'dedupe_nodes.nodes': self._parse_dedupe_nodes_context,
            'dedupe_edges.edge': self._parse_dedupe_edges_context,
        }

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        group_id: str | None = None,
        prompt_name: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Intercept graphiti LLM calls and route to DSPy modules.

        Args:
            messages: List of Message objects (system, user)
            response_model: Expected Pydantic response type
            prompt_name: Graphiti prompt identifier (e.g., 'extract_nodes.extract_message')

        Returns:
            Dict representation of response_model
        """

        # Check if we have a DSPy module for this prompt
        if prompt_name and prompt_name in self.dspy_modules:
            # Parse context from messages
            context = self._parse_context(messages, prompt_name)

            # Execute DSPy module
            module = self.dspy_modules[prompt_name]
            result = module(**context)

            # Convert DSPy output to expected format
            return self._format_response(result, response_model)

        else:
            # Fallback to default LLM behavior (if needed)
            raise NotImplementedError(
                f"No DSPy module registered for prompt: {prompt_name}"
            )

    def _parse_context(self, messages: list[Message], prompt_name: str) -> dict:
        """
        Extract structured context from graphiti's message format.

        Graphiti passes context in XML-like tags within messages.
        """
        parser = self.context_parsers.get(prompt_name)
        if parser:
            return parser(messages)

        # Default: extract from user message
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if user_msg:
            return {"text": user_msg.content}

        return {}

    def _parse_node_extraction_context(self, messages: list[Message]) -> dict:
        """
        Parse context for entity extraction prompts.

        Expected format in messages:
        <PREVIOUS MESSAGES>
        episode 1 content...
        episode 2 content...
        </PREVIOUS MESSAGES>

        <CURRENT MESSAGE>
        current episode content
        </CURRENT MESSAGE>
        """
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract current episode
        current = self._extract_xml_content(content, "CURRENT MESSAGE")

        # Extract previous episodes
        previous_text = self._extract_xml_content(content, "PREVIOUS MESSAGES")
        previous_episodes = previous_text.split('\n\n') if previous_text else []

        return {
            "episode_content": current,
            "previous_episodes": previous_episodes,
            "entity_types": []  # TODO: Parse from message if provided
        }

    def _parse_edge_extraction_context(self, messages: list[Message]) -> dict:
        """Parse context for relationship extraction."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract episode content
        episode_content = self._extract_xml_content(content, "EPISODE")

        # Extract nodes (usually provided as JSON)
        nodes_text = self._extract_xml_content(content, "NODES")
        nodes = json.loads(nodes_text) if nodes_text else []

        # Extract reference time
        ref_time = self._extract_xml_content(content, "REFERENCE TIME")

        return {
            "episode_content": episode_content,
            "nodes": nodes,
            "reference_time": ref_time,
            "edge_types": []
        }

    def _parse_dedupe_nodes_context(self, messages: list[Message]) -> dict:
        """Parse context for entity deduplication."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract entities to compare
        new_entity = self._extract_xml_content(content, "NEW ENTITY")
        existing_entities = self._extract_xml_content(content, "EXISTING ENTITIES")

        return {
            "new_entity": new_entity,
            "existing_entities": json.loads(existing_entities) if existing_entities else [],
            "context": self._extract_xml_content(content, "CONTEXT")
        }

    def _parse_dedupe_edges_context(self, messages: list[Message]) -> dict:
        """Parse context for edge deduplication."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        return {
            "new_fact": self._extract_xml_content(content, "NEW FACT"),
            "existing_facts": json.loads(
                self._extract_xml_content(content, "EXISTING FACTS") or "[]"
            ),
            "context": self._extract_xml_content(content, "CONTEXT")
        }

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content between XML-like tags."""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            return ""

        return text[start_idx + len(start_tag):end_idx].strip()

    def _format_response(
        self,
        dspy_result: Any,
        expected_model: type[BaseModel]
    ) -> dict[str, Any]:
        """
        Convert DSPy output to graphiti's expected format.

        Args:
            dspy_result: Result from DSPy module
            expected_model: Graphiti's expected Pydantic model

        Returns:
            Dict matching expected_model schema
        """
        # If DSPy returned the expected model directly
        if isinstance(dspy_result, expected_model):
            return dspy_result.model_dump()

        # If DSPy returned a wrapper with the result
        if hasattr(dspy_result, expected_model.__name__.lower()):
            result_obj = getattr(dspy_result, expected_model.__name__.lower())
            if isinstance(result_obj, expected_model):
                return result_obj.model_dump()

        # Try to construct expected model from DSPy output
        try:
            model_instance = expected_model(**dspy_result.__dict__)
            return model_instance.model_dump()
        except Exception as e:
            raise ValueError(
                f"Could not convert DSPy result to {expected_model.__name__}: {e}"
            )
```

### DSPy Module Specifications

#### 1. ExtractEntitiesModule

```python
from pydantic import BaseModel, Field
import dspy

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity classification (Person, Organization, etc.)")
    attributes: dict[str, Any] = Field(default_factory=dict)

class ExtractedEntities(BaseModel):
    entities: list[ExtractedEntity]

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField(
        desc="Current episode text to extract entities from"
    )
    previous_episodes: list[str] = dspy.InputField(
        desc="Previous episodes for context",
        default_factory=list
    )
    entity_types: list[str] = dspy.InputField(
        desc="Valid entity types to extract",
        default_factory=list
    )

    entities: ExtractedEntities = dspy.OutputField(
        desc="List of extracted entities with type classifications"
    )

class ExtractEntitiesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractEntitiesSignature)

    def forward(self, episode_content, previous_episodes=None, entity_types=None):
        return self._predict(
            episode_content=episode_content,
            previous_episodes=previous_episodes or [],
            entity_types=entity_types or []
        )
```

#### 2. DedupeNodesModule

```python
class NodeDuplicateResult(BaseModel):
    is_duplicate: bool = Field(description="Whether entities are duplicates")
    confidence: float = Field(description="Confidence score 0-1")
    match_index: int | None = Field(description="Index of matching entity")
    reasoning: str = Field(description="Explanation of decision")

class DedupeNodesSignature(dspy.Signature):
    """Determine if an entity is a duplicate of existing entities."""

    new_entity: str = dspy.InputField(
        desc="Name of newly extracted entity"
    )
    existing_entities: list[str] = dspy.InputField(
        desc="Names of existing entities to compare against"
    )
    context: str = dspy.InputField(
        desc="Episode context for disambiguation"
    )

    result: NodeDuplicateResult = dspy.OutputField(
        desc="Deduplication decision with reasoning"
    )

class DedupeNodesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(DedupeNodesSignature)

    def forward(self, new_entity, existing_entities, context):
        return self._predict(
            new_entity=new_entity,
            existing_entities=existing_entities,
            context=context
        )
```

#### 3. ExtractRelationshipsModule

```python
class ExtractedEdge(BaseModel):
    source_uuid: str
    target_uuid: str
    relation_type: str = Field(description="Relationship name")
    fact: str = Field(description="Textual description of relationship")
    valid_at: str | None = Field(description="ISO timestamp when fact became true")
    invalid_at: str | None = Field(description="ISO timestamp when fact stopped being true")

class ExtractedEdges(BaseModel):
    edges: list[ExtractedEdge]

class ExtractRelationshipsSignature(dspy.Signature):
    """Extract relationships between entities."""

    episode_content: str = dspy.InputField()
    nodes: list[dict] = dspy.InputField(
        desc="Extracted entities with UUIDs and names: [{'uuid': '...', 'name': '...'}, ...]"
    )
    reference_time: str = dspy.InputField(
        desc="ISO timestamp for temporal resolution"
    )
    edge_types: list[str] = dspy.InputField(
        desc="Valid relationship types",
        default_factory=list
    )

    edges: ExtractedEdges = dspy.OutputField(
        desc="List of relationships with temporal bounds"
    )

class ExtractRelationshipsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractRelationshipsSignature)

    def forward(self, episode_content, nodes, reference_time, edge_types=None):
        return self._predict(
            episode_content=episode_content,
            nodes=nodes,
            reference_time=reference_time,
            edge_types=edge_types or []
        )
```

#### 4. DedupeEdgesModule

```python
class EdgeDeduplicateResult(BaseModel):
    is_duplicate: bool
    is_contradiction: bool
    match_index: int | None
    contradicts_index: int | None
    reasoning: str

class DedupeEdgesSignature(dspy.Signature):
    """Determine if a fact duplicates or contradicts existing facts."""

    new_fact: str = dspy.InputField()
    existing_facts: list[str] = dspy.InputField()
    context: str = dspy.InputField()

    result: EdgeDeduplicateResult = dspy.OutputField()

class DedupeEdgesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(DedupeEdgesSignature)

    def forward(self, new_fact, existing_facts, context):
        return self._predict(
            new_fact=new_fact,
            existing_facts=existing_facts,
            context=context
        )
```

### Context Extraction and Response Formatting

**Context Extraction** happens in `DSPyLLMClient._parse_context()`:
- Graphiti passes context as formatted text in Message objects
- Usually uses XML-like tags: `<CURRENT MESSAGE>...</CURRENT MESSAGE>`
- Parser extracts structured data for DSPy signatures

**Response Formatting** happens in `DSPyLLMClient._format_response()`:
- Converts DSPy output to graphiti's expected Pydantic model
- Handles model_dump() conversion
- Validates schema compatibility

### Optimization Strategy

**When to Run MIPRO**:

1. **After initial implementation** - Optimize all modules together
2. **Per-module optimization** - Fine-tune specific operations
3. **Domain adaptation** - When switching to new data domains

**MIPRO Setup**:
```python
import dspy
from dspy.teleprompt import MIPRO

# Prepare training data
train_examples = [
    dspy.Example(
        episode_content="John met with Sarah at the office.",
        entities=ExtractedEntities(entities=[
            ExtractedEntity(name="John", entity_type="Person"),
            ExtractedEntity(name="Sarah", entity_type="Person"),
            ExtractedEntity(name="office", entity_type="Location")
        ])
    ).with_inputs("episode_content")
    # ... more examples
]

# Optimize module
teleprompter = MIPRO(
    metric=entity_extraction_metric,
    num_candidates=10,
    init_temperature=1.0
)

optimized_module = teleprompter.compile(
    ExtractEntitiesModule(),
    trainset=train_examples,
    num_trials=100
)

# Save optimized prompts
optimized_module.save("optimized_prompts/extract_entities.json")
```

**Evaluation Metrics**:
```python
def entity_extraction_metric(example, prediction, trace=None):
    """
    Metric for entity extraction quality.

    Evaluates:
    - Recall: Did we extract all entities?
    - Precision: Are extracted entities valid?
    - Type accuracy: Correct entity classifications?
    """
    gold_entities = {e.name for e in example.entities.entities}
    pred_entities = {e.name for e in prediction.entities.entities}

    if len(gold_entities) == 0:
        return 0.0

    # Calculate recall
    recall = len(gold_entities & pred_entities) / len(gold_entities)

    # Calculate precision
    precision = len(gold_entities & pred_entities) / len(pred_entities) if pred_entities else 0.0

    # F1 score
    if recall + precision == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

---

## 6. FalkorDB Integration

### Database Connection Setup

```python
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver import GraphProvider

# Initialize FalkorDB driver
driver = FalkorDriver(
    uri="falkordb://localhost:6379",
    database="knowledge_graph",
    provider=GraphProvider.FALKORDB
)

# Driver handles:
# - Connection pooling
# - Query execution
# - Transaction management
# - Provider-specific query syntax
```

**Connection String Formats**:
- Local: `falkordb://localhost:6379`
- Remote: `falkordb://host:port?password=xxx`
- With database: `falkordb://localhost:6379/dbname`

### Query Patterns for FalkorDB

Graphiti uses provider-aware query execution:

```python
# Execute query with FalkorDB-specific syntax
records, summary, keys = await driver.execute_query(
    """
    MATCH (n:Entity {group_id: $group_id})
    WHERE n.name CONTAINS $search_term
    RETURN n.uuid, n.name, n.labels
    LIMIT $limit
    """,
    group_id=group_id,
    search_term="John",
    limit=10
)
```

**Common Query Patterns**:

1. **Node Creation/Merge**:
```python
await driver.execute_query(
    """
    MERGE (n:Entity {uuid: $uuid})
    ON CREATE SET
        n.name = $name,
        n.group_id = $group_id,
        n.labels = $labels,
        n.created_at = $created_at,
        n.name_embedding = $embedding
    ON MATCH SET
        n.summary = $summary,
        n.attributes = $attributes
    RETURN n
    """,
    uuid=node.uuid,
    name=node.name,
    # ... etc
)
```

2. **Edge Creation**:
```python
await driver.execute_query(
    """
    MATCH (source:Entity {uuid: $source_uuid})
    MATCH (target:Entity {uuid: $target_uuid})
    MERGE (source)-[r:RELATES_TO {uuid: $edge_uuid}]->(target)
    SET
        r.name = $name,
        r.fact = $fact,
        r.valid_at = $valid_at,
        r.created_at = $created_at
    RETURN r
    """,
    source_uuid=edge.source_node_uuid,
    target_uuid=edge.target_node_uuid,
    # ... etc
)
```

3. **Embedding Search** (FalkorDB vector indexing):
```python
# Create vector index (once)
await driver.execute_query(
    """
    CALL db.idx.vector.createNodeIndex(
        'Entity',
        'name_embedding_idx',
        'name_embedding',
        1024,
        'COSINE'
    )
    """
)

# Search by embedding
await driver.execute_query(
    """
    CALL db.idx.vector.queryNodes(
        'Entity',
        'name_embedding_idx',
        $top_k,
        $query_embedding
    ) YIELD node, score
    WHERE node.group_id = $group_id
    RETURN node.uuid, node.name, score
    """,
    query_embedding=embedding,
    top_k=5,
    group_id=group_id
)
```

### Index Management

**Create Indexes on Startup**:
```python
async def setup_indexes(driver: FalkorDriver):
    """Create indexes for efficient querying."""

    # UUID indexes (unique constraint)
    await driver.execute_query(
        "CREATE INDEX ON :Entity(uuid)"
    )
    await driver.execute_query(
        "CREATE INDEX ON :Episodic(uuid)"
    )

    # Group ID indexes (for partitioning)
    await driver.execute_query(
        "CREATE INDEX ON :Entity(group_id)"
    )
    await driver.execute_query(
        "CREATE INDEX ON :Episodic(group_id)"
    )

    # Vector indexes for similarity search
    await driver.execute_query(
        """
        CALL db.idx.vector.createNodeIndex(
            'Entity',
            'name_embedding_idx',
            'name_embedding',
            1024,
            'COSINE'
        )
        """
    )

    await driver.execute_query(
        """
        CALL db.idx.vector.createRelationshipIndex(
            'RELATES_TO',
            'fact_embedding_idx',
            'fact_embedding',
            1024,
            'COSINE'
        )
        """
    )
```

### Transaction Handling

Graphiti uses driver sessions for transaction management:

```python
# Automatic transaction (single query)
await driver.execute_query(query, **params)

# Manual transaction (multiple queries)
async with driver.session() as session:
    async with session.begin_transaction() as tx:
        await tx.run(query1, **params1)
        await tx.run(query2, **params2)
        await tx.commit()
```

**Bulk Operations** (handled by graphiti's `add_nodes_and_edges_bulk`):
- Batches all inserts into single transaction
- Handles embedding generation before save
- Uses MERGE queries for upsert behavior

---

## 7. Implementation Roadmap

### Phase 1: Core Extraction (Entities, Relationships)

**Goal**: Basic entity and relationship extraction with FalkorDB persistence

**Tasks**:
1. ✅ Set up FalkorDB driver and connection
2. ✅ Create database indexes
3. ✅ Implement `ExtractEntitiesModule` DSPy signature
4. ✅ Implement `ExtractRelationshipsModule` DSPy signature
5. ✅ Create basic custom pipeline function
6. ✅ Test extraction with simple episodes
7. ✅ Verify FalkorDB persistence

**Deliverables**:
- Working entity extraction (no deduplication)
- Working relationship extraction
- Successful FalkorDB saves
- Basic integration test

**Success Metrics**:
- Extract 90%+ of entities from test episodes
- Extract 80%+ of relationships
- No database errors

### Phase 2: Deduplication and Resolution

**Goal**: Entity and edge deduplication with UUID mapping

**Tasks**:
1. ✅ Implement `DedupeNodesModule` DSPy signature
2. ✅ Implement `DedupeEdgesModule` DSPy signature
3. ✅ Add embedding search for entity matching
4. ✅ Implement UUID mapping logic
5. ✅ Test deduplication across multiple episodes
6. ✅ Verify merge behavior in FalkorDB

**Deliverables**:
- Entity deduplication working
- Edge deduplication with contradiction detection
- Proper UUID mapping
- Multi-episode integration test

**Success Metrics**:
- <5% duplicate entities in graph
- Correctly identify 90%+ of contradictions
- No broken UUID references

### Phase 3: Advanced Features (Temporal, Communities)

**Goal**: Temporal reasoning and community detection

**Tasks**:
1. ✅ Implement `ExtractTemporalModule` for valid_at/invalid_at
2. ✅ Implement temporal invalidation logic
3. ✅ Add community detection (optional)
4. ✅ Implement `ExtractAttributesModule` for typed attributes
5. ✅ Test temporal queries (facts at specific time)

**Deliverables**:
- Temporal edge extraction
- Time-aware queries
- Optional community grouping
- Typed attribute extraction

**Success Metrics**:
- Correct temporal bounds on 85%+ of edges
- Time-travel queries return correct facts
- Communities detected for dense subgraphs

### Phase 4: Optimization and Fine-Tuning

**Goal**: Optimize DSPy modules with MIPRO

**Tasks**:
1. ✅ Create evaluation dataset (gold-standard extractions)
2. ✅ Define metrics for each module
3. ✅ Run MIPRO optimization
4. ✅ Compare optimized vs baseline performance
5. ✅ Save optimized prompts
6. ✅ Benchmark end-to-end pipeline

**Deliverables**:
- Optimized DSPy modules
- Performance benchmarks
- Saved prompt artifacts
- Documentation of improvements

**Success Metrics**:
- 10-20% improvement in extraction metrics
- Faster inference time (MLX optimization)
- Reproducible optimization process

---

## 8. Code Examples

### Complete Example: Custom Pipeline Using Graphiti Models + DSPy

```python
"""
Complete custom knowledge graph ingestion pipeline.

This example demonstrates:
- Using graphiti Pydantic models (EntityNode, EntityEdge, etc.)
- Custom DSPy modules for extraction
- FalkorDB persistence
- Full control over pipeline steps
"""

import asyncio
from datetime import datetime
from uuid import uuid4
from typing import Any

import dspy
from pydantic import BaseModel, Field

# Graphiti imports
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.utils.datetime_utils import utc_now

# Our DSPy integration
from dspy_outlines import OutlinesLM, OutlinesAdapter


# ============================================================
# DSPy Module Definitions
# ============================================================

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = "Entity"
    attributes: dict[str, Any] = Field(default_factory=dict)

class ExtractedEntities(BaseModel):
    entities: list[ExtractedEntity]

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField()
    previous_episodes: list[str] = dspy.InputField(default_factory=list)

    result: ExtractedEntities = dspy.OutputField()

class ExtractEntitiesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractEntitiesSignature)

    def forward(self, episode_content, previous_episodes=None):
        return self._predict(
            episode_content=episode_content,
            previous_episodes=previous_episodes or []
        )


class ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    relation_type: str
    fact: str

class ExtractedRelationships(BaseModel):
    relationships: list[ExtractedRelationship]

class ExtractRelationshipsSignature(dspy.Signature):
    """Extract relationships between entities."""

    episode_content: str = dspy.InputField()
    entity_names: list[str] = dspy.InputField()

    result: ExtractedRelationships = dspy.OutputField()

class ExtractRelationshipsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractRelationshipsSignature)

    def forward(self, episode_content, entity_names):
        return self._predict(
            episode_content=episode_content,
            entity_names=entity_names
        )


# ============================================================
# Custom Pipeline Implementation
# ============================================================

class KnowledgeGraphPipeline:
    """
    Custom knowledge graph ingestion pipeline.

    Uses graphiti models + DSPy for extraction + FalkorDB for storage.
    """

    def __init__(
        self,
        driver: FalkorDriver,
        embedder: EmbedderClient,
        group_id: str = "default"
    ):
        self.driver = driver
        self.embedder = embedder
        self.group_id = group_id

        # Configure DSPy
        lm = OutlinesLM(
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            max_tokens=2048
        )
        dspy.configure(lm=lm, adapter=OutlinesAdapter())

        # Initialize DSPy modules
        self.extract_entities = ExtractEntitiesModule()
        self.extract_relationships = ExtractRelationshipsModule()

    async def ingest_episode(
        self,
        episode_body: str,
        reference_time: datetime,
        episode_name: str | None = None
    ) -> dict[str, Any]:
        """
        Ingest a single episode into the knowledge graph.

        Args:
            episode_body: Raw episode text
            reference_time: When the episode occurred
            episode_name: Optional episode identifier

        Returns:
            Dict with extracted nodes, edges, and episode
        """

        # STEP 1: Create EpisodicNode
        episode = EpisodicNode(
            uuid=str(uuid4()),
            name=episode_name or f"Episode {reference_time.isoformat()}",
            group_id=self.group_id,
            source=EpisodeType.message,
            source_description="User conversation",
            content=episode_body,
            valid_at=reference_time,
            created_at=utc_now(),
            labels=[]
        )

        # STEP 2: Extract Entities
        extraction_result = self.extract_entities(
            episode_content=episode_body,
            previous_episodes=[]  # TODO: Retrieve from DB
        )

        # Convert to EntityNode objects
        entity_nodes = []
        name_to_uuid = {}  # Map entity names to UUIDs

        for entity in extraction_result.result.entities:
            node = EntityNode(
                uuid=str(uuid4()),
                name=entity.name,
                group_id=self.group_id,
                labels=[entity.entity_type],
                created_at=utc_now(),
                summary="",
                attributes=entity.attributes
            )
            entity_nodes.append(node)
            name_to_uuid[entity.name] = node.uuid

        # STEP 3: Extract Relationships
        relationship_result = self.extract_relationships(
            episode_content=episode_body,
            entity_names=[e.name for e in entity_nodes]
        )

        # Convert to EntityEdge objects
        entity_edges = []

        for rel in relationship_result.result.relationships:
            # Look up UUIDs for source/target
            source_uuid = name_to_uuid.get(rel.source_name)
            target_uuid = name_to_uuid.get(rel.target_name)

            if source_uuid and target_uuid:
                edge = EntityEdge(
                    uuid=str(uuid4()),
                    source_node_uuid=source_uuid,
                    target_node_uuid=target_uuid,
                    name=rel.relation_type,
                    fact=rel.fact,
                    group_id=self.group_id,
                    created_at=utc_now(),
                    valid_at=reference_time,
                    episodes=[episode.uuid]
                )
                entity_edges.append(edge)

        # STEP 4: Generate Embeddings
        for node in entity_nodes:
            await node.generate_name_embedding(self.embedder)

        for edge in entity_edges:
            await edge.generate_embedding(self.embedder)

        # STEP 5: Create Episodic Edges (MENTIONS)
        episodic_edges = [
            EpisodicEdge(
                uuid=str(uuid4()),
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                group_id=self.group_id,
                created_at=utc_now()
            )
            for node in entity_nodes
        ]

        # Update episode with entity edge references
        episode.entity_edges = [e.uuid for e in entity_edges]

        # STEP 6: Save to FalkorDB
        await add_nodes_and_edges_bulk(
            driver=self.driver,
            episodic_nodes=[episode],
            episodic_edges=episodic_edges,
            entity_nodes=entity_nodes,
            entity_edges=entity_edges,
            embedder=self.embedder
        )

        return {
            "episode": episode,
            "nodes": entity_nodes,
            "edges": entity_edges
        }


# ============================================================
# Initialization Code
# ============================================================

async def main():
    """Initialize pipeline and process an episode."""

    # 1. Create FalkorDB driver
    driver = FalkorDriver(
        uri="falkordb://localhost:6379",
        database="knowledge_graph"
    )

    # 2. Create embedder (using sentence transformers)
    from graphiti_core.embedder import BGEEmbedderClient

    embedder = BGEEmbedderClient(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 3. Create pipeline
    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="user_123"
    )

    # 4. Process an episode
    result = await pipeline.ingest_episode(
        episode_body="John met with Sarah at the office to discuss the project.",
        reference_time=datetime.now(),
        episode_name="Meeting discussion"
    )

    print(f"Extracted {len(result['nodes'])} entities")
    print(f"Extracted {len(result['edges'])} relationships")

    for node in result['nodes']:
        print(f"  - {node.name} ({node.labels})")

    for edge in result['edges']:
        print(f"  - {edge.fact}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Processing One Episode End-to-End

```python
"""
Minimal example: Process one episode from start to finish.
"""

import asyncio
from datetime import datetime

async def process_episode_example():
    # Setup
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder import BGEEmbedderClient

    driver = FalkorDriver(uri="falkordb://localhost:6379")
    embedder = BGEEmbedderClient()

    # Create pipeline
    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="demo"
    )

    # Process episode
    result = await pipeline.ingest_episode(
        episode_body="""
        Alice and Bob are working on the machine learning project.
        Alice is the team lead. Bob joined the team last month.
        They are using Python and TensorFlow.
        """,
        reference_time=datetime.now(),
        episode_name="Team update"
    )

    # Results
    print("✓ Episode processed successfully")
    print(f"✓ {len(result['nodes'])} entities extracted:")
    for node in result['nodes']:
        print(f"    {node.name} ({', '.join(node.labels)})")

    print(f"✓ {len(result['edges'])} relationships extracted:")
    for edge in result['edges']:
        print(f"    {edge.fact}")

    print(f"✓ Saved to FalkorDB (episode UUID: {result['episode'].uuid})")

asyncio.run(process_episode_example())
```

**Expected Output**:
```
✓ Episode processed successfully
✓ 5 entities extracted:
    Alice (Person)
    Bob (Person)
    machine learning project (Project)
    Python (Technology)
    TensorFlow (Technology)
✓ 4 relationships extracted:
    Alice works_on machine learning project
    Bob works_on machine learning project
    Alice leads Bob
    machine learning project uses Python
✓ Saved to FalkorDB (episode UUID: 123e4567-e89b-12d3-a456-426614174000)
```

---

## 9. Testing Strategy

### Unit Tests for DSPy Modules

```python
"""
Unit tests for DSPy extraction modules.
"""

import pytest
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

@pytest.fixture(scope="module")
def configure_dspy():
    """Configure DSPy once for all tests."""
    lm = OutlinesLM(model_name="mlx-community/Qwen2.5-7B-Instruct-4bit")
    dspy.configure(lm=lm, adapter=OutlinesAdapter())

def test_extract_entities_basic(configure_dspy):
    """Test basic entity extraction."""
    module = ExtractEntitiesModule()

    result = module(
        episode_content="John works at Google in San Francisco."
    )

    entities = result.result.entities
    entity_names = {e.name for e in entities}

    assert "John" in entity_names
    assert "Google" in entity_names
    assert "San Francisco" in entity_names

def test_extract_entities_with_types(configure_dspy):
    """Test entity type classification."""
    module = ExtractEntitiesModule()

    result = module(
        episode_content="Alice is a software engineer at Microsoft."
    )

    entities = {e.name: e.entity_type for e in result.result.entities}

    assert entities.get("Alice") == "Person"
    assert entities.get("Microsoft") == "Organization"

def test_extract_relationships(configure_dspy):
    """Test relationship extraction."""
    module = ExtractRelationshipsModule()

    result = module(
        episode_content="Bob manages the engineering team.",
        entity_names=["Bob", "engineering team"]
    )

    relationships = result.result.relationships

    assert len(relationships) > 0
    assert any(
        r.source_name == "Bob" and r.target_name == "engineering team"
        for r in relationships
    )

def test_dedupe_nodes_duplicate(configure_dspy):
    """Test entity deduplication detects duplicates."""
    module = DedupeNodesModule()

    result = module(
        new_entity="John Smith",
        existing_entities=["John Smith", "Jane Doe"],
        context="John Smith sent an email."
    )

    assert result.result.is_duplicate is True
    assert result.result.match_index == 0

def test_dedupe_nodes_not_duplicate(configure_dspy):
    """Test entity deduplication detects new entities."""
    module = DedupeNodesModule()

    result = module(
        new_entity="Alice Johnson",
        existing_entities=["John Smith", "Jane Doe"],
        context="Alice Johnson joined the team."
    )

    assert result.result.is_duplicate is False
```

### Integration Tests with FalkorDB

```python
"""
Integration tests for full pipeline with FalkorDB.
"""

import pytest
import asyncio
from datetime import datetime

@pytest.fixture(scope="module")
async def pipeline():
    """Create test pipeline with FalkorDB."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder import BGEEmbedderClient

    driver = FalkorDriver(uri="falkordb://localhost:6379", database="test_db")
    embedder = BGEEmbedderClient()

    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="test_group"
    )

    yield pipeline

    # Cleanup: Delete test data
    from graphiti_core.nodes import Node
    await Node.delete_by_group_id(driver, "test_group")

@pytest.mark.asyncio
async def test_ingest_single_episode(pipeline):
    """Test ingesting a single episode."""
    result = await pipeline.ingest_episode(
        episode_body="Alice works at Google.",
        reference_time=datetime.now(),
        episode_name="Test episode"
    )

    assert len(result['nodes']) >= 2  # Alice, Google
    assert len(result['edges']) >= 1  # works_at
    assert result['episode'].uuid is not None

@pytest.mark.asyncio
async def test_ingest_multiple_episodes_deduplication(pipeline):
    """Test entity deduplication across episodes."""
    # First episode
    result1 = await pipeline.ingest_episode(
        episode_body="Alice works at Google.",
        reference_time=datetime.now()
    )

    # Second episode (mentions Alice again)
    result2 = await pipeline.ingest_episode(
        episode_body="Alice is a senior engineer.",
        reference_time=datetime.now()
    )

    # Query database to check Alice appears only once
    from graphiti_core.nodes import EntityNode

    alice_nodes = await EntityNode.get_by_name(
        pipeline.driver,
        name="Alice",
        group_id="test_group"
    )

    # Should be deduplicated to single entity
    assert len(alice_nodes) == 1

@pytest.mark.asyncio
async def test_temporal_edges(pipeline):
    """Test temporal edge extraction."""
    result = await pipeline.ingest_episode(
        episode_body="Bob joined Google in 2020. He left in 2023.",
        reference_time=datetime(2023, 6, 1)
    )

    # Find "Bob works_at Google" edge
    work_edge = next(
        (e for e in result['edges'] if e.name == "works_at"),
        None
    )

    assert work_edge is not None
    assert work_edge.valid_at is not None
    assert work_edge.invalid_at is not None
```

### Performance Benchmarks

```python
"""
Performance benchmarks for pipeline.
"""

import time
import asyncio
from statistics import mean, stdev

async def benchmark_extraction_speed(pipeline, num_episodes=100):
    """Benchmark episode extraction speed."""

    episode_template = "Person{i} works at Company{i} on Project{i}."

    times = []

    for i in range(num_episodes):
        episode_body = episode_template.format(i=i)

        start = time.time()
        await pipeline.ingest_episode(
            episode_body=episode_body,
            reference_time=datetime.now()
        )
        elapsed = time.time() - start

        times.append(elapsed)

    print(f"Episodes processed: {num_episodes}")
    print(f"Average time: {mean(times):.2f}s")
    print(f"Std dev: {stdev(times):.2f}s")
    print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")
    print(f"Throughput: {num_episodes / sum(times):.2f} episodes/sec")

async def benchmark_query_performance(pipeline, num_queries=1000):
    """Benchmark graph query performance."""

    # First, populate graph
    for i in range(100):
        await pipeline.ingest_episode(
            episode_body=f"Entity{i} relates_to Entity{i+1}.",
            reference_time=datetime.now()
        )

    # Benchmark queries
    from graphiti_core.nodes import EntityNode

    times = []

    for i in range(num_queries):
        start = time.time()

        await EntityNode.get_by_name(
            pipeline.driver,
            name=f"Entity{i % 100}",
            group_id=pipeline.group_id
        )

        elapsed = time.time() - start
        times.append(elapsed)

    print(f"Queries executed: {num_queries}")
    print(f"Average time: {mean(times)*1000:.2f}ms")
    print(f"Throughput: {num_queries / sum(times):.0f} queries/sec")
```

**Run Benchmarks**:
```bash
pytest tests/test_performance.py --benchmark
```

**Expected Performance** (Qwen2.5-7B on M1 Max):
- Entity extraction: ~2-5 seconds/episode
- Relationship extraction: ~3-6 seconds/episode
- Database save: ~100-200ms/episode
- Total throughput: ~0.1-0.2 episodes/sec
- Query performance: ~5-10ms/query (with indexes)

---

## Summary

This integration plan provides a comprehensive roadmap for building a custom knowledge graph extraction pipeline that:

1. **Reuses graphiti-core's battle-tested Pydantic models** (EntityNode, EntityEdge, etc.)
2. **Replaces all LLM operations with DSPy modules** for type-safe, optimizable extraction
3. **Uses FalkorDB as the graph backend** with full compatibility
4. **Maintains granular control** over each pipeline step
5. **Supports optimization** through DSPy's MIPRO framework

The custom pipeline bypasses graphiti's automated `add_episode()` workflow, giving us complete control while leveraging the robust data models and database utilities that graphiti provides.

Key advantages:
- **Type safety**: Outlines constrained generation ensures valid outputs
- **Local inference**: MLX on Apple Silicon for fast, private inference
- **Optimization**: DSPy's MIPRO for continuous improvement
- **Flexibility**: Add custom processing steps anywhere in the pipeline
- **Compatibility**: Drop-in replacement for graphiti's default LLM operations

Next steps:
1. Implement Phase 1 (core extraction)
2. Test with FalkorDB
3. Add deduplication (Phase 2)
4. Optimize with MIPRO (Phase 4)
