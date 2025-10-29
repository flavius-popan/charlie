# Edge Extraction and Resolution Pipeline

**Source files**: `.venv/lib/python3.13/site-packages/graphiti_core/`

This document details the edge extraction and resolution pipeline in graphiti-core, covering how relationships are extracted from episodes, deduplicated, checked for contradictions, and integrated into the knowledge graph.

---

## Table of Contents

1. [Edge Extraction Flow](#edge-extraction-flow)
2. [Response Models](#response-models)
3. [Edge Resolution Flow](#edge-resolution-flow)
4. [Episodic Edges](#episodic-edges)
5. [Community Edges](#community-edges)
6. [Key Integration Points](#key-integration-points)
7. [Pydantic Models and Data Structures](#pydantic-models-and-data-structures)
8. [Search-Based Edge Finding](#search-based-edge-finding)

---

## Edge Extraction Flow

### `extract_edges()` - Main Extraction Function

**Location**: `utils/maintenance/edge_operations.py:89-238`

**Signature**:
```python
async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityEdge]:
```

**Process**:

1. **Context Preparation** (lines 103-133):
   - Builds edge type signature map from custom edge types
   - Prepares context with episode content, entity nodes, previous episodes, and reference time
   - Includes custom edge type definitions with descriptions

2. **Reflexion Loop** (lines 135-168):
   - Iterates up to `MAX_REFLEXION_ITERATIONS` to ensure completeness
   - First pass: Extract edges with `prompt_library.extract_edges.edge()`
   - Reflexion pass: Check for missing facts with `prompt_library.extract_edges.reflexion()`
   - If missing facts found, adds custom prompt to guide next iteration
   - Continues until no missing facts or max iterations reached

3. **Edge Construction** (lines 175-235):
   - Converts extracted data into `EntityEdge` objects
   - Validates entity IDs are within valid range
   - Parses temporal bounds (`valid_at`, `invalid_at`) from ISO 8601 strings
   - Associates edge with current episode UUID
   - Filters out empty facts

**Key Features**:
- **Reflexion**: Self-correction loop ensures completeness
- **Temporal bounds**: Extracts when relationship became true/ended
- **Custom edge types**: Supports user-defined relationship types with specific source/target signatures
- **Validation**: Checks entity ID ranges and date parsing

### Temporal Bounds Extraction

**Related Functions**:
- `extract_edge_dates()` in `utils/maintenance/temporal_operations.py:33-71`
- Prompt in `prompts/extract_edge_dates.py`

**Purpose**: Standalone function to extract/refine temporal bounds for existing edges.

```python
async def extract_edge_dates(
    llm_client: LLMClient,
    edge: EntityEdge,
    current_episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
) -> tuple[datetime | None, datetime | None]:
```

**Key Rules** (from `prompts/extract_edge_dates.py:66-86`):
- Use reference timestamp to resolve relative time ("10 years ago", "2 mins ago")
- If relationship is present tense, use reference timestamp for `valid_at`
- Only extract dates directly related to relationship formation/change
- Do not infer dates from unrelated events
- If only date mentioned (no time), assume midnight
- If only year mentioned, use January 1st at 00:00:00

---

## Response Models

### ExtractedEdges

**Location**: `prompts/extract_edges.py:47-48`

```python
class ExtractedEdges(BaseModel):
    edges: list[Edge]
```

### Edge

**Location**: `prompts/extract_edges.py:25-44`

```python
class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_id: int = Field(
        ..., description='The id of the source entity from the ENTITIES list'
    )
    target_entity_id: int = Field(
        ..., description='The id of the target entity from the ENTITIES list'
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
```

**Key Fields**:
- `relation_type`: SCREAMING_SNAKE_CASE predicate (e.g., "WORKS_AT", "FOUNDED")
- `source_entity_id` / `target_entity_id`: Integer indices into the provided entities list
- `fact`: Natural language description (paraphrased, not verbatim quote)
- `valid_at` / `invalid_at`: ISO 8601 timestamps for temporal validity

### MissingFacts

**Location**: `prompts/extract_edges.py:51-52`

```python
class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="facts that weren't extracted")
```

Used in reflexion loop to identify gaps in extraction.

---

## Edge Resolution Flow

### `resolve_extracted_edges()` - Main Resolution Function

**Location**: `utils/maintenance/edge_operations.py:241-403`

**Signature**:
```python
async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
```

**Returns**: `(resolved_edges, invalidated_edges)`

**Process**:

1. **Fast Deduplication** (lines 249-263):
   - Exact match deduplication within extracted edges
   - Uses normalized fact text + source/target UUIDs as key
   - Eliminates duplicates before expensive LLM calls

2. **Embedding Generation** (line 268):
   - Creates embeddings for all extracted edge facts
   - Used for semantic similarity search

3. **Parallel Search Operations** (lines 270-312):
   - **Related edges search**: Finds existing edges between same source/target nodes
   - **Invalidation candidates search**: Broader search for potentially contradictory edges
   - Uses `EDGE_HYBRID_SEARCH_RRF` (BM25 + cosine similarity with RRF reranking)

4. **Edge Type Filtering** (lines 320-362):
   - Determines which custom edge types are valid for each edge based on source/target node labels
   - Fallback to `DEFAULT_EDGE_NAME` ("RELATES_TO") if custom type is disallowed

5. **Individual Edge Resolution** (lines 364-385):
   - Parallel calls to `resolve_extracted_edge()` for each edge
   - Each edge resolved against related edges and invalidation candidates
   - Returns resolved edge and any edges to invalidate

6. **Embedding Finalization** (lines 398-401):
   - Regenerates embeddings for resolved and invalidated edges

### `resolve_extracted_edge()` - Individual Edge Resolution

**Location**: `utils/maintenance/edge_operations.py:444-647`

**Signature**:
```python
async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    custom_edge_type_names: set[str] | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
```

**Returns**: `(resolved_edge, invalidated_edges, duplicate_edges)`

**Process**:

1. **Fast Path: Exact Match** (lines 482-493):
   - If normalized fact text and endpoints match existing edge exactly, reuse it
   - Append episode UUID to existing edge's episodes list

2. **LLM-Based Resolution** (lines 495-538):
   - Prepares context with existing edges, new edge, invalidation candidates
   - Calls `prompt_library.dedupe_edges.resolve_edge()` with `EdgeDuplicate` response model
   - LLM identifies duplicate facts and contradicted facts

3. **Duplicate Handling** (lines 550-558):
   - If duplicates found, reuses first duplicate edge
   - Appends current episode to duplicate edge's episodes list

4. **Edge Type Classification** (lines 575-613):
   - LLM determines appropriate edge type from custom types
   - If custom type allowed: extract structured attributes using `prompt_library.extract_edges.extract_attributes()`
   - If custom type disallowed: fallback to `DEFAULT_EDGE_NAME`
   - Non-custom LLM-generated labels allowed to pass through

5. **Temporal Invalidation** (lines 620-639):
   - If resolved edge already expired, mark it
   - Check if any invalidation candidates have more recent valid_at
   - Expire resolved edge if newer contradictory information exists

6. **Contradiction Resolution** (lines 642-644):
   - Call `resolve_edge_contradictions()` to expire contradictory edges
   - Uses temporal bounds to determine which edges to invalidate

### `resolve_edge_contradictions()` - Temporal Contradiction Logic

**Location**: `utils/maintenance/edge_operations.py:406-441`

**Purpose**: Determines which contradictory edges should be expired based on temporal bounds.

**Logic**:
```python
# Skip if edge already invalid before new edge becomes valid
if edge.invalid_at <= resolved_edge.valid_at:
    continue

# Skip if new edge invalid before existing edge becomes valid
if resolved_edge.invalid_at <= edge.valid_at:
    continue

# Invalidate edge if it became valid before resolved edge
if edge.valid_at < resolved_edge.valid_at:
    edge.invalid_at = resolved_edge.valid_at
    edge.expired_at = utc_now()
    invalidated_edges.append(edge)
```

### Deduplication Models

**EdgeDuplicate** - `prompts/dedupe_edges.py:25-34`

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
    fact_type: str = Field(..., description='One of the provided fact types or DEFAULT')
```

**Key Distinctions** (from prompt, lines 146-148):
- `duplicate_facts`: Use idx values from EXISTING FACTS (edges between same nodes)
- `contradicted_facts`: Use idx values from FACT INVALIDATION CANDIDATES (broader set)
- Two separate lists with independent idx ranges

### Contradiction Detection Models

**InvalidatedEdges** - `prompts/invalidate_edges.py:24-28`

```python
class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='List of ids of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )
```

Used by `get_edge_contradictions()` in `temporal_operations.py:74-107` for standalone contradiction detection.

---

## Episodic Edges

### `build_episodic_edges()` - MENTIONS Relationships

**Location**: `utils/maintenance/edge_operations.py:51-68`

**Purpose**: Creates MENTIONS edges from episodes to entities.

**Signature**:
```python
def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
```

**Process**:
- Creates one `EpisodicEdge` per entity mentioned in episode
- Edge direction: `Episode -> Entity`
- Edge type: `MENTIONS`
- Inherits group_id from entity node

**Graph Pattern**:
```
(Episode:Episodic)-[:MENTIONS]->(Entity:Entity)
```

**Use Case**: Tracks which entities were mentioned in which episodes. Used for:
- Episode-based filtering in search
- Provenance tracking
- Entity activity timeline

---

## Community Edges

### `build_community_edges()` - HAS_MEMBER Relationships

**Location**: `utils/maintenance/edge_operations.py:71-86`

**Purpose**: Creates HAS_MEMBER edges from communities to member entities.

**Signature**:
```python
def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
```

**Process**:
- Creates one `CommunityEdge` per entity in community
- Edge direction: `Community -> Entity`
- Edge type: `HAS_MEMBER`
- Inherits group_id from community node

**Graph Pattern**:
```
(Community:Community)-[:HAS_MEMBER]->(Entity:Entity)
```

**Use Case**: Represents hierarchical clustering of entities. Used for:
- Community detection algorithms
- Hierarchical search
- Entity grouping

---

## Key Integration Points

### 1. Custom Edge Types

**Edge Type Map**: `dict[tuple[str, str], list[str]]`
- Key: `(source_label, target_label)` - node label signature
- Value: List of edge type names valid for that signature

**Example**:
```python
edge_type_map = {
    ("Person", "Company"): ["WORKS_AT", "FOUNDED"],
    ("Person", "Person"): ["KNOWS", "MARRIED_TO"],
}
```

**Edge Types Dictionary**: `dict[str, type[BaseModel]]`
- Key: Edge type name
- Value: Pydantic model defining structured attributes

**Example**:
```python
class WorksAt(BaseModel):
    """Employment relationship between person and company"""
    position: str
    start_date: str | None

edge_types = {
    "WORKS_AT": WorksAt,
}
```

**Integration Flow**:
1. LLM extracts edges with relation_type
2. Edge type filtering ensures type matches node signature
3. If custom type assigned, extract structured attributes
4. Attributes stored in `EntityEdge.attributes` dict

### 2. Embedding Pipeline Integration

**Location**: `edges.py:623-631`

```python
async def create_entity_edge_embeddings(embedder: EmbedderClient, edges: list[EntityEdge]):
    filtered_edges = [edge for edge in edges if edge.fact]
    if len(filtered_edges) == 0:
        return
    fact_embeddings = await embedder.create_batch([edge.fact for edge in filtered_edges])
    for edge, fact_embedding in zip(filtered_edges, fact_embeddings, strict=True):
        edge.fact_embedding = fact_embedding
```

**Called**:
- After edge extraction (before resolution)
- After edge resolution (for resolved + invalidated edges)

**Purpose**: Enables semantic search over edge facts.

### 3. Episode Provenance

**Tracking**: `EntityEdge.episodes: list[str]`
- List of episode UUIDs where this edge was mentioned
- Appended to when duplicate edge found in new episode
- Enables provenance queries: "Which episodes support this fact?"

### 4. Group ID Partitioning

**Field**: `Edge.group_id: str`
- Enables multi-tenant partitioning
- All edges inherit group_id from parent nodes/episodes
- Search operations filter by group_id for isolation

---

## Pydantic Models and Data Structures

### EntityEdge Structure

**Location**: `edges.py:221-240`

```python
class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the edge. Dependent on edge name'
    )
```

**Inherited from Edge** (lines 45-50):
```python
class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime
```

**Key Methods**:
- `async def save(driver)` (line 285): Persists to graph database
- `async def generate_embedding(embedder)` (line 242): Creates fact embedding
- `async def load_fact_embedding(driver)` (line 253): Loads embedding from DB
- `@classmethod async def get_between_nodes(driver, source_uuid, target_uuid)` (line 345): Queries edges between specific nodes

### EpisodicEdge Structure

**Location**: `edges.py:131-182`

```python
class EpisodicEdge(Edge):
    # Inherits: uuid, group_id, source_node_uuid, target_node_uuid, created_at
    # No additional fields
```

**Pattern**: `(Episode:Episodic {uuid: source_node_uuid})-[:MENTIONS]->(Entity:Entity {uuid: target_node_uuid})`

### CommunityEdge Structure

**Location**: `edges.py:480-561`

```python
class CommunityEdge(Edge):
    # Inherits: uuid, group_id, source_node_uuid, target_node_uuid, created_at
    # No additional fields
```

**Pattern**: `(Community:Community {uuid: source_node_uuid})-[:HAS_MEMBER]->(Entity:Entity {uuid: target_node_uuid})`

### EdgeDates Model

**Location**: `prompts/extract_edge_dates.py:24-32`

```python
class EdgeDates(BaseModel):
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )
```

---

## Search-Based Edge Finding

### Search Configuration

**EDGE_HYBRID_SEARCH_RRF** - `search/search_config_recipes.py:110-116`

```python
EDGE_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    )
)
```

**Purpose**: Combines keyword (BM25) and semantic (cosine similarity) search with Reciprocal Rank Fusion reranking.

**Used in**:
- Finding related edges between same nodes (line 279-288 in `edge_operations.py`)
- Finding invalidation candidates (line 292-307)

### Search Workflow in Edge Resolution

**Related Edges Search** (lines 277-290):
```python
related_edges_results: list[SearchResults] = await semaphore_gather(
    *[
        search(
            clients,
            extracted_edge.fact,  # Query text
            group_ids=[extracted_edge.group_id],
            config=EDGE_HYBRID_SEARCH_RRF,
            search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
        )
        for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
    ]
)
```

**Key Points**:
- Query: Extracted edge fact text
- Filter: Only edges between same source/target nodes (`valid_edges`)
- Purpose: Find duplicate/similar edges for deduplication

**Invalidation Candidates Search** (lines 292-307):
```python
edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
    *[
        search(
            clients,
            extracted_edge.fact,  # Query text
            group_ids=[extracted_edge.group_id],
            config=EDGE_HYBRID_SEARCH_RRF,
            search_filter=SearchFilters(),  # No UUID filter - broader search
        )
        for extracted_edge in extracted_edges
    ]
)
```

**Key Points**:
- Query: Extracted edge fact text
- Filter: None (searches all edges in group)
- Purpose: Find potentially contradictory edges across entire graph

### Alternative Search Configurations

**Other edge-specific configs** (`search_config_recipes.py:119-153`):
- `EDGE_HYBRID_SEARCH_MMR`: Maximal Marginal Relevance reranking (diversity)
- `EDGE_HYBRID_SEARCH_NODE_DISTANCE`: Node distance-based reranking
- `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS`: Episode mention frequency reranking
- `EDGE_HYBRID_SEARCH_CROSS_ENCODER`: Deep learning reranking with BFS

---

## Summary

The edge extraction and resolution pipeline provides:

1. **Robust Extraction**: Reflexion loop ensures completeness, custom edge types enable domain-specific relationships
2. **Smart Deduplication**: Fast path for exact matches, LLM-based semantic deduplication for paraphrases
3. **Temporal Reasoning**: Tracks when relationships become true/end, automatically expires contradictory facts
4. **Search Integration**: Hybrid search (BM25 + semantic) finds related and contradictory edges
5. **Provenance Tracking**: Episode references enable fact verification and sourcing
6. **Type Safety**: Pydantic models with custom attributes for structured edge data

**Key Customization Points**:
- Define custom edge types with source/target signatures
- Add structured attributes via Pydantic models
- Tune search configurations (RRF, MMR, cross-encoder)
- Adjust reflexion iterations for completeness vs cost trade-offs
- Implement custom contradiction logic via temporal bounds
