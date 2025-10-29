# Episode Processing Pipeline

## Overview

Episodes are the primary entry point for all content ingestion in Graphiti. The `add_episode()` method orchestrates the complete pipeline from raw input to graph integration.

**Location**: `graphiti_core/graphiti.py:611-814`

## Episode Data Model

### EpisodicNode (graphiti_core/nodes.py:295-433)

```python
class EpisodicNode(Node):
    source: EpisodeType              # message, json, or text
    source_description: str          # Description of data source
    content: str                     # Raw episode data
    valid_at: datetime              # When original document was created
    entity_edges: list[str]         # UUIDs of entity edges referenced

    # Inherited from Node:
    uuid: str
    name: str
    group_id: str                   # Partition identifier
    labels: list[str]
    created_at: datetime
```

### Episode Types

| EpisodeType | Prompt Method | Use Case |
|-------------|---------------|----------|
| `message` | `extract_nodes.extract_message` | Chat logs, conversations |
| `text` | `extract_nodes.extract_text` | Articles, documents |
| `json` | `extract_nodes.extract_json` | Structured data, API responses |

## add_episode() Data Flow

### Method Signature
```python
async def add_episode(
    self,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    source: EpisodeType = EpisodeType.message,
    group_id: str | None = None,
    uuid: str | None = None,
    update_communities: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    previous_episode_uuids: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
) -> AddEpisodeResults
```

### Processing Steps

#### 1. Initialization (lines 682-720)
```python
start = time()
now = utc_now()

# Validation
validate_entity_types(entity_types)
validate_excluded_entity_types(excluded_entity_types, entity_types)
validate_group_id(group_id)
group_id = group_id or get_default_group_id(self.driver.provider)

# Retrieve context episodes
previous_episodes = (
    await self.retrieve_episodes(
        reference_time,
        last_n=RELEVANT_SCHEMA_LIMIT,  # From search_utils.py
        group_ids=[group_id],
        source=source,
    )
    if previous_episode_uuids is None
    else await EpisodicNode.get_by_uuids(self.driver, previous_episode_uuids)
)

# Create or retrieve episode
episode = (
    await EpisodicNode.get_by_uuid(self.driver, uuid)
    if uuid is not None
    else EpisodicNode(
        name=name,
        group_id=group_id,
        labels=[],
        source=source,
        content=episode_body,
        source_description=source_description,
        created_at=now,
        valid_at=reference_time,
    )
)
```

#### 2. Node Extraction (lines 730-740)
```python
extracted_nodes = await extract_nodes(
    self.clients,
    episode,
    previous_episodes,
    entity_types,
    excluded_entity_types
)

# Returns list[EntityNode] with:
# - name, uuid, group_id
# - labels (entity type classification)
# - summary (initially empty)
```

#### 3. Node Resolution (lines 734-740)
```python
nodes, uuid_map, _ = await resolve_extracted_nodes(
    self.clients,
    extracted_nodes,
    episode,
    previous_episodes,
    entity_types,
)

# Returns:
# - nodes: list[EntityNode] - deduplicated entities
# - uuid_map: dict[str, str] - maps extracted UUIDs to resolved UUIDs
# - duplicates: list[tuple[EntityNode, EntityNode]] - identified duplicates
```

#### 4. Edge Extraction and Resolution (lines 743-752)
```python
resolved_edges, invalidated_edges = await self._extract_and_resolve_edges(
    episode,
    extracted_nodes,
    previous_episodes,
    edge_type_map or edge_type_map_default,
    group_id,
    edge_types,
    nodes,
    uuid_map,
)
```

#### 5. Attribute Extraction (lines 755-757)
```python
hydrated_nodes = await extract_attributes_from_nodes(
    self.clients,
    nodes,
    episode,
    previous_episodes,
    entity_types
)
```

#### 6. Episode Data Processing (lines 762-764)
```python
entity_edges = resolved_edges + invalidated_edges

episodic_edges, episode = await self._process_episode_data(
    episode, hydrated_nodes, entity_edges, now
)

# Inside _process_episode_data (lines 413-436):
# 1. Build episodic edges (MENTIONS relationships)
episodic_edges = build_episodic_edges(nodes, episode.uuid, now)

# 2. Update episode with entity edge references
episode.entity_edges = [edge.uuid for edge in entity_edges]

# 3. Optionally clear raw content
if not self.store_raw_episode_content:
    episode.content = ''

# 4. Bulk save to database
await add_nodes_and_edges_bulk(
    self.driver,
    [episode],          # EpisodicNode
    episodic_edges,     # EpisodicEdge (MENTIONS)
    nodes,              # EntityNode
    entity_edges,       # EntityEdge (RELATES_TO)
    self.embedder,
)
```

#### 7. Community Updates (optional, lines 767-776)
```python
if update_communities:
    communities, community_edges = await semaphore_gather(
        *[
            update_community(
                self.driver,
                self.llm_client,
                self.embedder,
                node
            )
            for node in nodes
        ],
        max_coroutines=self.max_coroutines,
    )
```

## Database Save Operations

### Location: `graphiti_core/utils/bulk_utils.py`

The `add_nodes_and_edges_bulk()` function handles:

1. **Embedding Generation** (if missing)
   - Node name embeddings
   - Edge fact embeddings

2. **Graph Persistence**
   - Episodic nodes saved via `EpisodicNode.save()`
   - Entity nodes saved via `EntityNode.save()`
   - Episodic edges (MENTIONS) saved via `EpisodicEdge.save()`
   - Entity edges (RELATES_TO) saved via `EntityEdge.save()`

### Node Save Patterns

```python
# EpisodicNode.save() - graphiti_core/nodes.py:307-329
async def save(self, driver: GraphDriver):
    episode_args = {
        'uuid': self.uuid,
        'name': self.name,
        'group_id': self.group_id,
        'source_description': self.source_description,
        'content': self.content,
        'entity_edges': self.entity_edges,
        'created_at': self.created_at,
        'valid_at': self.valid_at,
        'source': self.source.value,
    }

    result = await driver.execute_query(
        get_episode_node_save_query(driver.provider),
        **episode_args
    )
```

## Context Window

**EPISODE_WINDOW_LEN = 3** (from `graph_data_operations.py:31`)

Previous episodes provide context for:
- Entity extraction
- Entity deduplication
- Edge extraction
- Temporal reasoning

## Return Value

```python
class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list[CommunityNode]
    community_edges: list[CommunityEdge]
```

## Integration Points for Custom Pipeline

### 1. **Episode Creation**
Replace automatic episode creation with custom EpisodicNode construction

### 2. **Node Extraction**
Override `extract_nodes()` to use DSPy instead of default prompts

### 3. **Node Resolution**
Control `resolve_extracted_nodes()` for custom deduplication logic

### 4. **Edge Extraction**
Override `extract_edges()` with DSPy-based relationship extraction

### 5. **Edge Resolution**
Control `resolve_extracted_edges()` for custom edge deduplication

### 6. **Attribute Extraction**
Override `extract_attributes_from_nodes()` with custom attribute logic

### 7. **Database Persistence**
Use `add_nodes_and_edges_bulk()` directly with custom data objects

## Key Observations for Custom Pipeline

1. **Separation of Concerns**: Each processing step is a distinct async function that can be called independently

2. **Pydantic Models**: All data objects are Pydantic models - can be reused in custom pipeline

3. **Embeddings**: Generated separately from extraction, can be controlled

4. **UUID Mapping**: Critical for maintaining referential integrity across extraction/resolution

5. **Bulk Operations**: Database saves are batched for efficiency

6. **FalkorDB Compatibility**: All query operations use `driver.execute_query()` which handles FalkorDB specifics
