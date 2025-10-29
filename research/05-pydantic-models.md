# Pydantic Models in Graphiti-core

**Source**: `.venv/lib/python3.13/site-packages/graphiti_core/`

## Overview

Graphiti-core uses Pydantic models for:
1. **Graph nodes** (Entity, Episodic, Community)
2. **Graph edges** (Entity, Episodic, Community)
3. **LLM response structures** (extraction, deduplication, classification)

---

## Node Models

**Source**: `.venv/lib/python3.13/site-packages/graphiti_core/nodes.py`

### Inheritance Hierarchy

```
Node (Abstract)
├─> EpisodicNode
├─> EntityNode
└─> CommunityNode
```

### Base Node (Abstract)

```python
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from datetime import datetime

class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver): ...

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False
```

**Common Fields**:
- `uuid`: Unique identifier (auto-generated)
- `name`: Human-readable name
- `group_id`: Partition key for multi-tenant graphs
- `labels`: Additional type labels (e.g., `["Entity", "Person"]`)
- `created_at`: Timestamp of creation (UTC)

**Methods**:
- `save()`: Persist to graph database (abstract, must implement)
- `delete()`: Remove from graph database
- Class methods: `get_by_uuid()`, `get_by_uuids()`, `delete_by_group_id()`, `delete_by_uuids()`

---

### EpisodicNode

**Purpose**: Represents a single observation/event (message, document, etc.)

```python
class EpisodeType(Enum):
    message = 'message'  # Format: "actor: content" (e.g., "user: Hello")
    json = 'json'        # Structured JSON data
    text = 'text'        # Plain text

class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created'
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )

    async def save(self, driver: GraphDriver): ...
```

**Fields**:
- `source`: Type of episode (message/json/text)
- `source_description`: Human-readable description (e.g., "Slack message from #engineering")
- `content`: Raw input data
- `valid_at`: When the episode occurred (not when it was processed)
- `entity_edges`: UUIDs of EntityEdge instances mentioned in this episode

**Relationships**:
- `MENTIONS` edges to EntityNode instances

**Custom Methods**:
- `get_by_entity_node_uuid()`: Find episodes that mention a specific entity

---

### EntityNode

**Purpose**: Represents a real-world entity (person, place, concept, etc.)

```python
class EntityNode(Node):
    name_embedding: list[float] | None = Field(
        default=None,
        description='embedding of the name'
    )
    summary: str = Field(
        description='regional summary of surrounding edges',
        default_factory=str
    )
    attributes: dict[str, Any] = Field(
        default={},
        description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient): ...
    async def load_name_embedding(self, driver: GraphDriver): ...
    async def save(self, driver: GraphDriver): ...
```

**Fields**:
- `name_embedding`: Vector embedding of entity name (for similarity search)
- `summary`: Concise description synthesized from connected edges (max 250 chars)
- `attributes`: Flexible dict for type-specific properties (e.g., `{"age": 30, "occupation": "engineer"}`)

**Embedding Strategy**:
- Embeddings generated from `name` field
- Used for deduplication and semantic search
- Loaded separately with `load_name_embedding()`

**Attributes Pattern**:
- Stored as dict (no fixed schema)
- Keys depend on entity type/labels
- Examples: `{"date_of_birth": "1990-01-01"}`, `{"headquarters": "San Francisco"}`

---

### CommunityNode

**Purpose**: Represents a cluster of related entities (hierarchical grouping)

```python
class CommunityNode(Node):
    name_embedding: list[float] | None = Field(
        default=None,
        description='embedding of the name'
    )
    summary: str = Field(
        description='region summary of member nodes',
        default_factory=str
    )

    async def generate_name_embedding(self, embedder: EmbedderClient): ...
    async def load_name_embedding(self, driver: GraphDriver): ...
    async def save(self, driver: GraphDriver): ...
```

**Fields**:
- `name_embedding`: Vector embedding of community name
- `summary`: Description of what connects the member entities

**Relationships**:
- `HAS_MEMBER` edges to EntityNode instances

**Use Case**: Hierarchical knowledge organization (e.g., "Engineering Team" → individual engineers)

---

## Edge Models

**Source**: `.venv/lib/python3.13/site-packages/graphiti_core/edges.py`

### Inheritance Hierarchy

```
Edge (Abstract)
├─> EpisodicEdge
├─> EntityEdge
└─> CommunityEdge
```

### Base Edge (Abstract)

```python
class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver): ...
```

**Common Fields**:
- `uuid`: Unique identifier
- `group_id`: Partition key (matches nodes)
- `source_node_uuid`: UUID of source node
- `target_node_uuid`: UUID of target node
- `created_at`: Timestamp of creation

---

### EpisodicEdge

**Purpose**: Links an episode to entities mentioned within it

```python
class EpisodicEdge(Edge):
    async def save(self, driver: GraphDriver): ...
```

**Relationship**: `(EpisodicNode)-[MENTIONS]->(EntityNode)`

**Fields**: Only base Edge fields (no additional properties)

**Usage**: Tracks which episodes mention which entities (provenance)

---

### EntityEdge

**Purpose**: Represents a factual relationship between two entities

```python
class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(
        description='fact representing the edge and nodes that it connects'
    )
    fact_embedding: list[float] | None = Field(
        default=None,
        description='embedding of the fact'
    )
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None,
        description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None,
        description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None,
        description='datetime of when the fact stopped being true'
    )
    attributes: dict[str, Any] = Field(
        default={},
        description='Additional attributes of the edge. Dependent on edge name'
    )

    async def generate_embedding(self, embedder: EmbedderClient): ...
    async def load_fact_embedding(self, driver: GraphDriver): ...
    async def save(self, driver: GraphDriver): ...
```

**Fields**:
- `name`: Relation type in SCREAMING_SNAKE_CASE (e.g., `WORKS_AT`, `FOUNDED`)
- `fact`: Natural language description (e.g., "Alice works at Acme Corp as a software engineer")
- `fact_embedding`: Vector embedding of the fact (for similarity search)
- `episodes`: UUIDs of EpisodicNode instances that support this fact
- `valid_at`: When the relationship started
- `invalid_at`: When the relationship ended (if applicable)
- `expired_at`: When the edge was superseded/contradicted (soft delete)
- `attributes`: Flexible dict for fact-specific properties

**Relationship**: `(EntityNode)-[RELATES_TO]->(EntityNode)`

**Temporal Semantics**:
- `valid_at`: Fact became true (e.g., hire date)
- `invalid_at`: Fact stopped being true (e.g., termination date)
- `expired_at`: Fact was determined to be incorrect/outdated (deduplication)

**Embedding Strategy**:
- Embeddings generated from `fact` field
- Used for deduplication (finding similar facts)
- Loaded separately with `load_fact_embedding()`

---

### CommunityEdge

**Purpose**: Links a community to its member entities

```python
class CommunityEdge(Edge):
    async def save(self, driver: GraphDriver): ...
```

**Relationship**: `(CommunityNode)-[HAS_MEMBER]->(EntityNode)`

**Fields**: Only base Edge fields

**Usage**: Represents membership in entity clusters

---

## Response Models (LLM Outputs)

### Extract Nodes (`prompts/extract_nodes.py`)

#### Entity Extraction

```python
class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(
        ...,
        description='List of extracted entities'
    )
```

**Usage**: LLM returns entity names + type IDs (IDs map to predefined entity types)

#### Reflexion (Missed Entity Detection)

```python
class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(
        ...,
        description="Names of entities that weren't extracted"
    )
```

**Usage**: Second-pass to catch missed entities

#### Entity Classification

```python
class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None,
        description='Type of the entity. Must be one of the provided types or None',
    )

class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ...,
        description='List of entities classification triples.'
    )
```

**Usage**: Classify existing entities with predefined types

#### Entity Summary

```python
class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 250 characters.',
    )
```

**Usage**: Generate concise entity descriptions

---

### Extract Edges (`prompts/extract_edges.py`)

#### Fact Extraction

```python
class Edge(BaseModel):  # Note: Different from edges.Edge (graph model)
    relation_type: str = Field(
        ...,
        description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE'
    )
    source_entity_id: int = Field(
        ...,
        description='The id of the source entity from the ENTITIES list'
    )
    target_entity_id: int = Field(
        ...,
        description='The id of the target entity from the ENTITIES list'
    )
    fact: str = Field(
        ...,
        description='A natural language description of the relationship between the entities, paraphrased from the source text',
    )
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )

class ExtractedEdges(BaseModel):
    edges: list[Edge]
```

**Usage**: Extract facts with entity IDs (indices from provided entity list)

**Naming Convention**: `relation_type` uses SCREAMING_SNAKE_CASE (e.g., `WORKS_AT`, `FOUNDED`)

#### Reflexion (Missed Fact Detection)

```python
class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(
        ...,
        description="facts that weren't extracted"
    )
```

---

### Dedupe Nodes (`prompts/dedupe_nodes.py`)

#### Node Deduplication

```python
class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. If no duplicate entities are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are a duplicate of the entity with the above id.',
    )

class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(
        ...,
        description='List of resolved nodes'
    )
```

**Usage**:
- `id`: ID of entity being checked
- `duplicate_idx`: Index of best match in existing entities (-1 if unique)
- `duplicates`: All matching indices (for merging multiple duplicates)
- `name`: Canonical name to use (most complete version)

**Pattern**: LLM receives numbered lists of entities and returns index references

---

### Dedupe Edges (`prompts/dedupe_edges.py`)

#### Edge Deduplication

```python
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of any duplicate facts. If no duplicate facts are found, default to empty list.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )
    fact_type: str = Field(
        ...,
        description='One of the provided fact types or DEFAULT'
    )
```

**Usage**:
- `duplicate_facts`: Indices of semantically identical facts
- `contradicted_facts`: Indices of facts that conflict with new fact
- `fact_type`: Classification of the new fact

**Pattern**: Detects both duplicates (merge) and contradictions (invalidate)

#### Unique Facts (Edge List Deduplication)

```python
class UniqueFact(BaseModel):
    uuid: str = Field(..., description='unique identifier of the fact')
    fact: str = Field(..., description='fact of a unique edge')

class UniqueFacts(BaseModel):
    unique_facts: list[UniqueFact]
```

**Usage**: Deduplicate a list of edges (batch operation)

---

## Model Reuse Strategy for Custom Pipeline

### Reuse As-Is

**Graph Models** (nodes.py, edges.py):
- `EntityNode`: Core entity representation
- `EntityEdge`: Fact representation with temporal info
- `EpisodicNode`: Episode/observation tracking

**Why**: These define the knowledge graph structure. Custom pipeline should target the same schema for graphiti-core compatibility.

**How**: Import directly:
```python
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
```

### Adapt for DSPy

**Response Models** (prompts/*.py):
- `ExtractedEntities`, `ExtractedEdges`, etc.

**Why**: These work perfectly as DSPy Signature output fields (already Pydantic models).

**How**: Use as OutputField types:
```python
class ExtractNodesSignature(dspy.Signature):
    """Extract entities from text"""
    # ... input fields ...
    extracted_entities: ExtractedEntities = dspy.OutputField()
```

**Mapping**:
1. **Extract Nodes** → Use `ExtractedEntities` (no changes needed)
2. **Extract Edges** → Use `ExtractedEdges` (no changes needed)
3. **Dedupe Nodes** → Use `NodeResolutions` (no changes needed)
4. **Dedupe Edges** → Use `EdgeDuplicate` (no changes needed)

### Replace/Simplify

**CommunityNode/CommunityEdge**:
- **Decision**: Skip in initial implementation (hierarchical clustering is complex)
- **Rationale**: Focus on core extraction pipeline first

**EpisodicNode.entity_edges**:
- **Decision**: May simplify to just track episode UUIDs on EntityEdge
- **Rationale**: Bidirectional tracking adds complexity; graph queries can derive this

---

## Example: End-to-End Model Usage

### Extraction Flow

```python
# 1. Extract entities
class ExtractNodesSignature(dspy.Signature):
    entity_types: str = dspy.InputField()
    episode_content: str = dspy.InputField()
    extracted_entities: ExtractedEntities = dspy.OutputField()

dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())
extractor = dspy.Predict(ExtractNodesSignature)

result = extractor(
    entity_types=json.dumps([{"id": 0, "type": "Person"}, {"id": 1, "type": "Organization"}]),
    episode_content="Alice works at Acme Corp"
)

# result.extracted_entities.extracted_entities = [
#     ExtractedEntity(name="Alice", entity_type_id=0),
#     ExtractedEntity(name="Acme Corp", entity_type_id=1)
# ]

# 2. Convert to EntityNode instances
entity_nodes = [
    EntityNode(
        name=e.name,
        group_id="user123",
        labels=["Entity", entity_types[e.entity_type_id]["type"]]
    )
    for e in result.extracted_entities.extracted_entities
]

# 3. Extract edges
class ExtractEdgesSignature(dspy.Signature):
    episode_content: str = dspy.InputField()
    nodes: str = dspy.InputField()  # JSON list with IDs
    reference_time: str = dspy.InputField()
    extracted_edges: ExtractedEdges = dspy.OutputField()

edge_extractor = dspy.Predict(ExtractEdgesSignature)

edge_result = edge_extractor(
    episode_content="Alice works at Acme Corp",
    nodes=json.dumps([{"id": 0, "name": "Alice"}, {"id": 1, "name": "Acme Corp"}]),
    reference_time="2025-10-29T00:00:00Z"
)

# edge_result.extracted_edges.edges = [
#     Edge(
#         relation_type="WORKS_AT",
#         source_entity_id=0,
#         target_entity_id=1,
#         fact="Alice works at Acme Corp",
#         valid_at="2025-10-29T00:00:00Z"
#     )
# ]

# 4. Convert to EntityEdge instances
entity_edges = [
    EntityEdge(
        source_node_uuid=entity_nodes[e.source_entity_id].uuid,
        target_node_uuid=entity_nodes[e.target_entity_id].uuid,
        name=e.relation_type,
        fact=e.fact,
        group_id="user123",
        created_at=datetime.utcnow(),
        valid_at=datetime.fromisoformat(e.valid_at.replace('Z', '+00:00')) if e.valid_at else None,
        invalid_at=datetime.fromisoformat(e.invalid_at.replace('Z', '+00:00')) if e.invalid_at else None
    )
    for e in edge_result.extracted_edges.edges
]
```

---

## Field Type Summary

| Model | UUID/ID | Timestamps | Embeddings | Flexible Attrs |
|-------|---------|------------|------------|----------------|
| `Node` (base) | `uuid: str` | `created_at: datetime` | ❌ | ❌ |
| `EpisodicNode` | Inherits | `+valid_at` | ❌ | ❌ |
| `EntityNode` | Inherits | Inherits | `name_embedding` | `attributes: dict` |
| `CommunityNode` | Inherits | Inherits | `name_embedding` | ❌ |
| `Edge` (base) | `uuid: str` | `created_at: datetime` | ❌ | ❌ |
| `EpisodicEdge` | Inherits | Inherits | ❌ | ❌ |
| `EntityEdge` | Inherits | `+valid_at, +invalid_at, +expired_at` | `fact_embedding` | `attributes: dict` |
| `CommunityEdge` | Inherits | Inherits | ❌ | ❌ |

---

## Key Decisions for Charlie

### Must Implement

1. **EntityNode + EntityEdge**: Core knowledge graph structure
2. **ExtractedEntities + ExtractedEdges**: Perfect fit for DSPy OutputField
3. **Embedding generation**: Required for deduplication
4. **Temporal fields**: `valid_at`, `invalid_at` for fact evolution

### Can Defer

1. **CommunityNode/CommunityEdge**: Hierarchical clustering (complex)
2. **Attributes dicts**: Start with fixed schemas, add flexibility later
3. **NodeResolutions/EdgeDuplicate**: Implement basic deduplication first

### Architecture Decision

**Use graphiti-core models directly** for graph persistence:
```python
# Your pipeline outputs graphiti-core models
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

# This makes your extracted graphs compatible with graphiti-core tools
# Can visualize/query using existing graphiti-core utilities
```

**Benefits**:
- Interoperability with graphiti-core ecosystem
- Reuse graph database schemas
- Standard format for knowledge graphs

---

## File Locations

- **Node models**: `.venv/lib/python3.13/site-packages/graphiti_core/nodes.py`
- **Edge models**: `.venv/lib/python3.13/site-packages/graphiti_core/edges.py`
- **Extract nodes responses**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/extract_nodes.py`
- **Extract edges responses**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/extract_edges.py`
- **Dedupe nodes responses**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/dedupe_nodes.py`
- **Dedupe edges responses**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/dedupe_edges.py`
