# Node Extraction and Resolution Pipeline

**Source**: `graphiti_core/utils/maintenance/node_operations.py`
**Related Files**:
- `graphiti_core/utils/maintenance/dedup_helpers.py`
- `graphiti_core/prompts/extract_nodes.py`
- `graphiti_core/prompts/dedupe_nodes.py`
- `graphiti_core/nodes.py`

## Overview

The node extraction pipeline in graphiti-core follows a three-phase process:

1. **Extraction**: Extract entity names and classify entity types from episodes using reflexion loop
2. **Resolution**: Deduplicate extracted nodes against existing nodes using similarity + LLM
3. **Attribute Hydration**: Extract summaries and custom attributes for each resolved node

---

## 1. Node Extraction Flow

### Function: `extract_nodes()`

**Location**: `node_operations.py:88-208`

**Signature**:
```python
async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
) -> list[EntityNode]
```

**Parameters**:
- `clients`: Contains LLM client, graph driver, and embedder
- `episode`: The current episode being processed
- `previous_episodes`: Context from prior episodes for entity disambiguation
- `entity_types`: Custom Pydantic models for entity classification (optional)
- `excluded_entity_types`: List of entity type names to filter out (optional)

**Returns**: List of `EntityNode` objects with name, labels, and UUID

### Extraction Process

#### Step 1: Entity Type Context Building (Lines 102-121)

```python
entity_types_context = [
    {
        'entity_type_id': 0,
        'entity_type_name': 'Entity',
        'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
    }
]

entity_types_context += (
    [
        {
            'entity_type_id': i + 1,
            'entity_type_name': type_name,
            'entity_type_description': type_model.__doc__,
        }
        for i, (type_name, type_model) in enumerate(entity_types.items())
    ]
    if entity_types is not None
    else []
)
```

- Default entity type (ID=0) is always present
- Custom entity types get sequential IDs starting from 1
- Entity type descriptions come from Pydantic model docstrings

#### Step 2: Initial Extraction with Episode-Specific Prompts (Lines 132-154)

Supports three episode types with different extraction strategies:

```python
if episode.source == EpisodeType.message:
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_message(context),
        response_model=ExtractedEntities,
        group_id=episode.group_id,
        prompt_name='extract_nodes.extract_message',
    )
elif episode.source == EpisodeType.text:
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_text(context),
        response_model=ExtractedEntities,
        group_id=episode.group_id,
        prompt_name='extract_nodes.extract_text',
    )
elif episode.source == EpisodeType.json:
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_json(context),
        response_model=ExtractedEntities,
        group_id=episode.group_id,
        prompt_name='extract_nodes.extract_json',
    )
```

**Key Differences**:
- **Message**: Extracts speaker first, disambiguates pronouns
- **Text**: General entity extraction for unstructured text
- **JSON**: Targets "name" fields and structured data references

#### Step 3: Reflexion Loop for Missed Entities (Lines 160-174)

**Configuration**: `MAX_REFLEXION_ITERATIONS` (default: 0, from `helpers.py:37`)

```python
while entities_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
    # ... initial extraction ...

    reflexion_iterations += 1
    if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
        missing_entities = await extract_nodes_reflexion(
            llm_client,
            episode,
            previous_episodes,
            [entity.name for entity in extracted_entities],
            episode.group_id,
        )

        entities_missed = len(missing_entities) != 0

        custom_prompt = 'Make sure that the following entities are extracted: '
        for entity in missing_entities:
            custom_prompt += f'\n{entity},'
```

**Reflexion Function** (`node_operations.py:63-85`):
```python
async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
    group_id: str | None = None,
) -> list[str]:
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context),
        MissedEntities,
        group_id=group_id,
        prompt_name='extract_nodes.reflexion',
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities
```

- Asks LLM to identify entities that should have been extracted but weren't
- Re-runs extraction with custom prompt highlighting missed entities
- Continues until no missed entities or max iterations reached

#### Step 4: Entity Node Creation (Lines 176-206)

```python
filtered_extracted_entities = [entity for entity in extracted_entities if entity.name.strip()]

extracted_nodes = []
for extracted_entity in filtered_extracted_entities:
    type_id = extracted_entity.entity_type_id
    if 0 <= type_id < len(entity_types_context):
        entity_type_name = entity_types_context[extracted_entity.entity_type_id].get(
            'entity_type_name'
        )
    else:
        entity_type_name = 'Entity'

    # Check if this entity type should be excluded
    if excluded_entity_types and entity_type_name in excluded_entity_types:
        logger.debug(f'Excluding entity "{extracted_entity.name}" of type "{entity_type_name}"')
        continue

    labels: list[str] = list({'Entity', str(entity_type_name)})

    new_node = EntityNode(
        name=extracted_entity.name,
        group_id=episode.group_id,
        labels=labels,
        summary='',
        created_at=utc_now(),
    )
    extracted_nodes.append(new_node)
```

**Key Points**:
- Filters empty entity names
- Maps entity_type_id to entity_type_name
- Applies exclusion filter if configured
- Creates labels set with 'Entity' + custom type
- UUID is auto-generated by EntityNode constructor
- Summary starts empty (filled during attribute extraction)

---

## 2. Response Models

**Location**: `prompts/extract_nodes.py:28-64`

### ExtractedEntity

```python
class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )
```

### ExtractedEntities

```python
class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')
```

### MissedEntities

```python
class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")
```

### EntitySummary

```python
class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )
```

---

## 3. Node Resolution Flow

### Function: `resolve_extracted_nodes()`

**Location**: `node_operations.py:395-450`

**Signature**:
```python
async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
```

**Parameters**:
- `existing_nodes_override`: Optional list of known existing nodes to consider first

**Returns** (3-tuple):
1. `list[EntityNode]`: Resolved nodes (either new or existing)
2. `dict[str, str]`: UUID mapping from extracted UUID → resolved UUID
3. `list[tuple[EntityNode, EntityNode]]`: Duplicate pairs (extracted, existing) for creating DUPLICATE_OF edges

### Resolution Process

#### Step 1: Collect Candidate Nodes (`_collect_candidate_nodes`, Lines 211-243)

```python
async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Search per extracted name and return unique candidates with overrides honored in order."""
    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                search_filter=SearchFilters(),
                config=NODE_HYBRID_SEARCH_RRF,
            )
            for node in extracted_nodes
        ]
    )

    candidate_nodes: list[EntityNode] = [node for result in search_results for node in result.nodes]

    if existing_nodes_override is not None:
        candidate_nodes.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in candidate_nodes:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    return ordered_candidates
```

**Search Strategy**: `NODE_HYBRID_SEARCH_RRF` (from `search_config_recipes.py:156-161`)
```python
NODE_HYBRID_SEARCH_RRF = SearchConfig(
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    )
)
```

- Performs parallel searches for each extracted node name
- Combines BM25 (full-text) + cosine similarity (embedding-based) search
- Reranks using Reciprocal Rank Fusion (RRF)
- Appends `existing_nodes_override` if provided
- Deduplicates by UUID while preserving order

#### Step 2: Build Candidate Indexes (`_build_candidate_indexes`, `dedup_helpers.py:170-195`)

```python
@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures that drive entity deduplication heuristics."""

    existing_nodes: list[EntityNode]
    nodes_by_uuid: dict[str, EntityNode]
    normalized_existing: defaultdict[str, list[EntityNode]]
    shingles_by_candidate: dict[str, set[str]]
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]]
```

```python
def _build_candidate_indexes(existing_nodes: list[EntityNode]) -> DedupCandidateIndexes:
    """Precompute exact and fuzzy lookup structures once per dedupe run."""
    normalized_existing: defaultdict[str, list[EntityNode]] = defaultdict(list)
    nodes_by_uuid: dict[str, EntityNode] = {}
    shingles_by_candidate: dict[str, set[str]] = {}
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in existing_nodes:
        normalized = _normalize_string_exact(candidate.name)
        normalized_existing[normalized].append(candidate)
        nodes_by_uuid[candidate.uuid] = candidate

        shingles = _cached_shingles(_normalize_name_for_fuzzy(candidate.name))
        shingles_by_candidate[candidate.uuid] = shingles

        signature = _minhash_signature(shingles)
        for band_index, band in enumerate(_lsh_bands(signature)):
            lsh_buckets[(band_index, band)].append(candidate.uuid)

    return DedupCandidateIndexes(...)
```

**Index Components**:
- `normalized_existing`: Exact name matching (case-insensitive, whitespace-collapsed)
- `shingles_by_candidate`: 3-gram character shingles for fuzzy matching
- `lsh_buckets`: MinHash LSH bands for fast approximate similarity search

#### Step 3: Similarity-Based Resolution (`_resolve_with_similarity`, `dedup_helpers.py:198-246`)

```python
@dataclass
class DedupResolutionState:
    """Mutable resolution bookkeeping shared across deterministic and LLM passes."""

    resolved_nodes: list[EntityNode | None]
    uuid_map: dict[str, str]
    unresolved_indices: list[int]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]] = field(default_factory=list)
```

**Algorithm** (Lines 198-246):

```python
def _resolve_with_similarity(
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
) -> None:
    """Attempt deterministic resolution using exact name hits and fuzzy MinHash comparisons."""
    for idx, node in enumerate(extracted_nodes):
        normalized_exact = _normalize_string_exact(node.name)
        normalized_fuzzy = _normalize_name_for_fuzzy(node.name)

        # 1. Low entropy filter
        if not _has_high_entropy(normalized_fuzzy):
            state.unresolved_indices.append(idx)
            continue

        # 2. Exact match
        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        if len(existing_matches) == 1:
            match = existing_matches[0]
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue
        if len(existing_matches) > 1:
            state.unresolved_indices.append(idx)
            continue

        # 3. Fuzzy match via MinHash LSH
        shingles = _cached_shingles(normalized_fuzzy)
        signature = _minhash_signature(shingles)
        candidate_ids: set[str] = set()
        for band_index, band in enumerate(_lsh_bands(signature)):
            candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

        best_candidate: EntityNode | None = None
        best_score = 0.0
        for candidate_id in candidate_ids:
            candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
            score = _jaccard_similarity(shingles, candidate_shingles)
            if score > best_score:
                best_score = score
                best_candidate = indexes.nodes_by_uuid.get(candidate_id)

        if best_candidate is not None and best_score >= _FUZZY_JACCARD_THRESHOLD:
            state.resolved_nodes[idx] = best_candidate
            state.uuid_map[node.uuid] = best_candidate.uuid
            if best_candidate.uuid != node.uuid:
                state.duplicate_pairs.append((node, best_candidate))
            continue

        state.unresolved_indices.append(idx)
```

**Resolution Strategies**:

1. **Entropy Check**: Skip low-entropy names (short/repetitive like "AI", "IT")
   - Threshold: `_NAME_ENTROPY_THRESHOLD = 1.5` (Shannon entropy)
   - Minimum: 6 characters OR 2 tokens

2. **Exact Match**: Normalized string equality
   - Single match: Resolve immediately
   - Multiple matches: Escalate to LLM

3. **Fuzzy Match**: MinHash + LSH + Jaccard similarity
   - **MinHash**: 32 permutations (`_MINHASH_PERMUTATIONS = 32`)
   - **LSH**: 4-element bands (`_MINHASH_BAND_SIZE = 4`)
   - **Threshold**: 0.9 Jaccard similarity (`_FUZZY_JACCARD_THRESHOLD = 0.9`)

**Normalization Functions** (`dedup_helpers.py:39-49`):
```python
def _normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()

def _normalize_name_for_fuzzy(name: str) -> str:
    """Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles."""
    normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)
```

#### Step 4: LLM Escalation (`_resolve_with_llm`, Lines 246-393)

For entities that couldn't be resolved deterministically:

```python
async def _resolve_with_llm(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_types: dict[str, type[BaseModel]] | None,
) -> None:
    """Escalate unresolved nodes to the dedupe prompt so the LLM can select or reject duplicates."""
    if not state.unresolved_indices:
        return

    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': entity_types_dict.get(
                next((item for item in node.labels if item != 'Entity'), '')
            ).__doc__
            or 'Default Entity Type',
        }
        for i, node in enumerate(llm_extracted_nodes)
    ]

    existing_nodes_context = [
        {
            **{
                'idx': i,
                'name': candidate.name,
                'entity_types': candidate.labels,
            },
            **candidate.attributes,
        }
        for i, candidate in enumerate(indexes.existing_nodes)
    ]

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.nodes(context),
        response_model=NodeResolutions,
        prompt_name='dedupe_nodes.nodes',
    )
```

**LLM Response Model** (`prompts/dedupe_nodes.py:25-42`):
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
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')
```

**LLM Response Processing** (Lines 328-393):
```python
node_resolutions: list[NodeDuplicate] = NodeResolutions(**llm_response).entity_resolutions

valid_relative_range = range(len(state.unresolved_indices))
processed_relative_ids: set[int] = set()

# Validation logging
received_ids = {r.id for r in node_resolutions}
expected_ids = set(valid_relative_range)
missing_ids = expected_ids - received_ids
extra_ids = received_ids - expected_ids

if missing_ids:
    logger.warning('LLM did not return resolutions for IDs: %s', sorted(missing_ids))

if extra_ids:
    logger.warning('LLM returned invalid IDs: %s', sorted(extra_ids))

for resolution in node_resolutions:
    relative_id: int = resolution.id
    duplicate_idx: int = resolution.duplicate_idx

    # Guard: validate relative_id
    if relative_id not in valid_relative_range:
        logger.warning('Skipping invalid LLM dedupe id %d', relative_id)
        continue

    # Guard: prevent duplicate processing
    if relative_id in processed_relative_ids:
        logger.warning('Duplicate LLM dedupe id %s received; ignoring.', relative_id)
        continue
    processed_relative_ids.add(relative_id)

    original_index = state.unresolved_indices[relative_id]
    extracted_node = extracted_nodes[original_index]

    resolved_node: EntityNode
    if duplicate_idx == -1:
        resolved_node = extracted_node
    elif 0 <= duplicate_idx < len(indexes.existing_nodes):
        resolved_node = indexes.existing_nodes[duplicate_idx]
    else:
        logger.warning('Invalid duplicate_idx %s; treating as no duplicate.', duplicate_idx)
        resolved_node = extracted_node

    state.resolved_nodes[original_index] = resolved_node
    state.uuid_map[extracted_node.uuid] = resolved_node.uuid
    if resolved_node.uuid != extracted_node.uuid:
        state.duplicate_pairs.append((extracted_node, resolved_node))
```

**Guardrails**:
- Validates all LLM-provided IDs are within expected range
- Logs warnings for missing/extra/duplicate IDs
- Treats malformed responses as "no duplicate" to maintain determinism
- Never halts ingestion on LLM misbehavior

#### Step 5: Fallback and Final Mapping (Lines 432-450)

```python
for idx, node in enumerate(extracted_nodes):
    if state.resolved_nodes[idx] is None:
        state.resolved_nodes[idx] = node
        state.uuid_map[node.uuid] = node.uuid

new_node_duplicates: list[
    tuple[EntityNode, EntityNode]
] = await filter_existing_duplicate_of_edges(driver, state.duplicate_pairs)

return (
    [node for node in state.resolved_nodes if node is not None],
    state.uuid_map,
    new_node_duplicates,
)
```

- Any unresolved nodes (missed by similarity + LLM) become new entities
- Filters duplicate pairs to avoid creating duplicate edges
- Returns resolved nodes, UUID mapping, and duplicate pairs for edge creation

---

## 4. Attribute Extraction

### Function: `extract_attributes_from_nodes()`

**Location**: `node_operations.py:453-483`

**Signature**:
```python
async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> list[EntityNode]:
```

**Parameters**:
- `should_summarize_node`: Optional filter function to determine which nodes get summaries

**Process**:
```python
llm_client = clients.llm_client
embedder = clients.embedder

updated_nodes: list[EntityNode] = await semaphore_gather(
    *[
        extract_attributes_from_node(
            llm_client,
            node,
            episode,
            previous_episodes,
            (
                entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                if entity_types is not None
                else None
            ),
            should_summarize_node,
        )
        for node in nodes
    ]
)

await create_entity_node_embeddings(embedder, updated_nodes)

return updated_nodes
```

- Extracts attributes and summaries for all nodes in parallel
- Generates name embeddings for all nodes
- Embeddings are used for similarity search during resolution

### Function: `extract_attributes_from_node()`

**Location**: `node_operations.py:486-506`

```python
async def extract_attributes_from_node(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: type[BaseModel] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> EntityNode:
    # Extract attributes if entity type is defined and has attributes
    llm_response = await _extract_entity_attributes(
        llm_client, node, episode, previous_episodes, entity_type
    )

    # Extract summary if needed
    await _extract_entity_summary(
        llm_client, node, episode, previous_episodes, should_summarize_node
    )

    node.attributes.update(llm_response)

    return node
```

### Custom Attributes (`_extract_entity_attributes`, Lines 509-541)

```python
async def _extract_entity_attributes(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> dict[str, Any]:
    if entity_type is None or len(entity_type.model_fields) == 0:
        return {}

    attributes_context = _build_episode_context(
        # should not include summary
        node_data={
            'name': node.name,
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_attributes(attributes_context),
        response_model=entity_type,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_attributes',
    )

    # validate response
    entity_type(**llm_response)

    return llm_response
```

**Key Points**:
- Only extracts attributes if entity type Pydantic model has fields
- Uses small model size for cost efficiency
- Validates response against entity type schema
- Returns dict of attribute values

### Summary Extraction (`_extract_entity_summary`, Lines 544-573)

```python
async def _extract_entity_summary(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
) -> None:
    if should_summarize_node is not None and not await should_summarize_node(node):
        return

    summary_context = _build_episode_context(
        node_data={
            'name': node.name,
            'summary': truncate_at_sentence(node.summary, MAX_SUMMARY_CHARS),
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    summary_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_summary(summary_context),
        response_model=EntitySummary,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_summary',
    )

    node.summary = truncate_at_sentence(summary_response.get('summary', ''), MAX_SUMMARY_CHARS)
```

**Key Points**:
- Respects `should_summarize_node` filter if provided
- Truncates existing summary to MAX_SUMMARY_CHARS before sending to LLM
- Updates node summary field directly
- Uses small model size

---

## 5. EntityNode Pydantic Model

**Location**: `nodes.py:435-440`

```python
class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )
```

**Base Class** (`nodes.py:87-93`):
```python
class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())
```

**Field Descriptions**:
- `uuid`: Auto-generated UUIDv4 string
- `name`: Entity name (extracted from episode)
- `group_id`: Partition key (inherited from episode)
- `labels`: List of entity types (e.g., `['Entity', 'Person']`)
- `created_at`: Timestamp when node was created
- `name_embedding`: Vector embedding for similarity search (generated after extraction)
- `summary`: Accumulated context about entity across episodes
- `attributes`: Custom properties defined by entity type schema

**Storage** (`nodes.py:477-508`):
```python
async def save(self, driver: GraphDriver):
    entity_data: dict[str, Any] = {
        'uuid': self.uuid,
        'name': self.name,
        'name_embedding': self.name_embedding,
        'group_id': self.group_id,
        'summary': self.summary,
        'created_at': self.created_at,
    }

    if driver.provider == GraphProvider.KUZU:
        entity_data['attributes'] = json.dumps(self.attributes)
        entity_data['labels'] = list(set(self.labels + ['Entity']))
        result = await driver.execute_query(
            get_entity_node_save_query(driver.provider, labels=''),
            **entity_data,
        )
    else:
        entity_data.update(self.attributes or {})
        labels = ':'.join(self.labels + ['Entity'])

        result = await driver.execute_query(
            get_entity_node_save_query(driver.provider, labels),
            entity_data=entity_data,
        )
```

**Storage Strategy**:
- **Kuzu**: Attributes stored as JSON string in `attributes` field
- **Other backends**: Attributes flattened into node properties
- Labels always include 'Entity' base label

---

## 6. UUID Mapping for Referential Integrity

**Purpose**: Map temporary UUIDs from extraction to final resolved UUIDs

**Generation Point**: `extract_nodes()` creates EntityNode with auto-generated UUID
```python
new_node = EntityNode(
    name=extracted_entity.name,
    group_id=episode.group_id,
    labels=labels,
    summary='',
    created_at=utc_now(),
)
# UUID generated in Node.__init__ via Field(default_factory=lambda: str(uuid4()))
```

**Mapping Creation**: `resolve_extracted_nodes()` returns mapping dict
```python
state.uuid_map[extracted_node.uuid] = resolved_node.uuid
```

**Usage**: Edge extraction uses mapping to reference correct node UUIDs
- If node is new: `extracted_uuid → extracted_uuid`
- If node is duplicate: `extracted_uuid → existing_uuid`
- Ensures edge relationships point to canonical entity nodes

**Example**:
```python
extracted_node = EntityNode(name="Alice")  # uuid: "abc-123"
resolved_node = existing_node  # uuid: "def-456"

uuid_map = {"abc-123": "def-456"}

# When creating edge: use uuid_map["abc-123"] → "def-456"
```

---

## 7. Key Integration Points for Custom Pipeline

### 1. **Custom Entity Types**

Define Pydantic models with docstrings for classification:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """A human individual mentioned in the content."""
    age: int | None = Field(None, description="Age in years")
    occupation: str | None = Field(None, description="Job or profession")

class Organization(BaseModel):
    """A company, institution, or formal group."""
    founded: str | None = Field(None, description="Founding date")
    industry: str | None = Field(None, description="Primary industry")

entity_types = {
    "Person": Person,
    "Organization": Organization,
}

nodes = await extract_nodes(
    clients=clients,
    episode=episode,
    previous_episodes=previous_episodes,
    entity_types=entity_types,
)
```

**Key Points**:
- Docstring becomes `entity_type_description` in extraction prompt
- Field descriptions guide attribute extraction
- Use `excluded_entity_types=["Person"]` to filter after extraction

### 2. **Custom Summary Filter**

Control which nodes receive summaries:

```python
async def should_summarize_node(node: EntityNode) -> bool:
    # Only summarize Person and Organization types
    return any(label in ["Person", "Organization"] for label in node.labels)

nodes = await extract_attributes_from_nodes(
    clients=clients,
    nodes=resolved_nodes,
    episode=episode,
    previous_episodes=previous_episodes,
    entity_types=entity_types,
    should_summarize_node=should_summarize_node,
)
```

### 3. **Custom Deduplication Strategy**

Override candidate collection:

```python
# Provide known existing nodes directly
existing_nodes_override = await EntityNode.get_by_group_ids(
    driver=driver,
    group_ids=[episode.group_id],
)

resolved_nodes, uuid_map, duplicates = await resolve_extracted_nodes(
    clients=clients,
    extracted_nodes=extracted_nodes,
    episode=episode,
    previous_episodes=previous_episodes,
    entity_types=entity_types,
    existing_nodes_override=existing_nodes_override,
)
```

### 4. **Bypass Search-Based Resolution**

For deterministic pipelines without graph search:

```python
# Skip search by providing empty override
resolved_nodes, uuid_map, duplicates = await resolve_extracted_nodes(
    clients=clients,
    extracted_nodes=extracted_nodes,
    episode=None,  # No episode context
    previous_episodes=None,
    entity_types=entity_types,
    existing_nodes_override=[],  # Empty list = no candidates = all new nodes
)
```

### 5. **Custom Reflexion Configuration**

Control reflexion iterations via environment variable:

```bash
export MAX_REFLEXION_ITERATIONS=2
```

Or programmatically (before import):
```python
import os
os.environ['MAX_REFLEXION_ITERATIONS'] = '2'

from graphiti_core.utils.maintenance.node_operations import extract_nodes
```

### 6. **Access Intermediate State**

Inspect deduplication decisions:

```python
# After resolution
for extracted, existing in duplicate_pairs:
    print(f"Merged: {extracted.name} ({extracted.uuid}) → {existing.name} ({existing.uuid})")

for extracted_uuid, resolved_uuid in uuid_map.items():
    if extracted_uuid != resolved_uuid:
        print(f"Duplicate mapping: {extracted_uuid} → {resolved_uuid}")
```

---

## 8. Prompt Templates

### Message Extraction (`prompts/extract_nodes.py:86-132`)

**Key Instructions**:
- Always extract speaker first (part before `:`)
- Disambiguate pronouns using context
- Exclude pronouns like "you", "me", "he/she/they"
- Don't extract relationships or temporal info
- Use entity types for classification

### JSON Extraction (`prompts/extract_nodes.py:135-165`)

**Key Instructions**:
- Extract entities that JSON represents ("name" or "user" fields)
- Extract entities mentioned in all properties
- Don't extract date properties

### Text Extraction (`prompts/extract_nodes.py:168-196`)

**Key Instructions**:
- Extract explicitly or implicitly mentioned entities
- Avoid relationships or actions
- Avoid temporal information
- Be explicit, use full names

### Reflexion (`prompts/extract_nodes.py:199-220`)

**Task**:
Given previous messages, current message, and extracted entities, determine which entities were missed.

### Deduplication (`prompts/dedupe_nodes.py:117-185`)

**Key Rules**:
- Only mark as duplicates if same real-world object/concept
- Semantic equivalence counts (descriptive label = named entity)
- Don't mark related-but-distinct entities as duplicates
- Must return resolution for EVERY extracted entity ID
- Set `duplicate_idx=-1` if no duplicate found
- Use smallest idx as `duplicate_idx`, list all in `duplicates`

**Response Format**:
```json
{
  "entity_resolutions": [
    {
      "id": 0,
      "name": "Best full name",
      "duplicate_idx": 3,
      "duplicates": [3, 7]
    }
  ]
}
```

---

## Summary

**Pipeline Flow**:
1. Extract entity names + types via reflexion loop
2. Search for candidate matches using hybrid search
3. Resolve via similarity heuristics (exact match, fuzzy MinHash)
4. Escalate unresolved to LLM deduplication
5. Extract custom attributes + summaries per entity type
6. Generate embeddings for similarity search
7. Return resolved nodes + UUID mapping + duplicate pairs

**Key Design Patterns**:
- Reflexion loop for iterative extraction quality
- Deterministic similarity resolution with LLM fallback
- Defensive LLM response handling (never halt on misbehavior)
- UUID mapping for referential integrity across pipeline stages
- Parallel extraction/resolution/embedding for performance
- Entity type system via Pydantic models + docstrings
