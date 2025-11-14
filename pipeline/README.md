# Graphiti Pipeline Modules

**Goal**: Completely local MLX inference pipeline for knowledge graph ingestion, replicating graphiti-core's functionality without remote LLM APIs.

## Architecture

This pipeline reimplements graphiti-core's ingestion stages using:
- **DSPy modules** for each pipeline stage
- **dspy_outlines adapter** for structured output via Outlines constrained generation
- **MLX** for local LLM inference (no API calls)
- **graphiti-core utilities** for validators, deduplication, and graph operations (maximize code reuse)

### Design Pattern

Each stage follows a **two-layer architecture** for DSPy optimization:

1. **Pure `dspy.Module`**: LLM extraction only (sync, no DB/async). Optimizable with teleprompters.
2. **Orchestrator class**: Handles DB I/O, episode creation, post-processing. Accepts compiled modules.

This separation makes optimization 10-100x faster (no DB overhead) and metrics clearer (no post-processing dilution).

**Episode-First Convention**: Create full `EpisodicNode` BEFORE extraction begins (graphiti-core pattern).

## Quick Start

```python
import asyncio
from pipeline import add_journal

async def main():
    result = await add_journal(
        content="Today I met with Sarah at Stanford to discuss AI ethics...",
        group_id="user_123"
    )
    print(f"Episode: {result.episode.uuid}")
    print(f"Extracted {len(result.nodes)} entities")
    print(f"Extracted {len(result.edges)} relationships")

asyncio.run(main())
```

## Pipeline Entry Point

### `add_journal()` - Orchestrator

Analogous to graphiti-core's `add_episode()`. Entry point that accepts plain text and orchestrates all stages.

**Usage:**
```python
from pipeline import add_journal

result = await add_journal(
    content="Journal entry text...",
    group_id="user_123",  # Optional, defaults to FalkorDB default '\\_'
    entity_types=None,     # Optional custom entity schemas
    excluded_entity_types=None  # Optional types to exclude
)
# Returns: AddJournalResults(episode, nodes, edges, episodic_edges, uuid_map, metadata)
```

## Stage Data Flow

Based on verified graphiti-core implementation (`graphiti.py:611-813`):

### Previous Episodes Context

**Fetched ONCE** at the start of add_journal() and reused by all stages:
```python
previous_episodes = await fetch_recent_episodes(
    group_id,
    reference_time,
    limit=5  # RELEVANT_SCHEMA_LIMIT in graphiti-core
)
```

### Stage 1: Extract Nodes

**Returns:**
```python
@dataclass
class ExtractNodesOutput:
    episode: EpisodicNode
    extracted_nodes: list[EntityNode]  # Nodes with original UUIDs (for Stage 2 edge extraction)
    nodes: list[EntityNode]            # RESOLVED entities (canonical UUIDs)
    uuid_map: dict[str, str]           # provisional_uuid → canonical_uuid
    duplicate_pairs: list[tuple]       # For DUPLICATE_OF edges (Stage 5)
    metadata: dict[str, Any]           # Statistics: extracted_count, exact_matches, fuzzy_matches, new_entities
```

**Extractor**

- Stage 1 uses a single DSPy `EntityExtractor` module (same structure as graphiti-core for easy optimization).
- Reflexion/NER experiments have been rolled back; future iterative passes will plug into the existing stubs in `pipeline/extract_nodes.py` if needed.

### Stage 2: Extract Edges

**Input** (from Stage 1):
- episode: EpisodicNode
- extracted_nodes: list[EntityNode] (original UUIDs for LLM indexing)
- resolved_nodes: list[EntityNode] (canonical UUIDs for validation)
- uuid_map: dict[str, str] (for edge pointer remapping)
- previous_episodes: list[EpisodicNode] (context, unused currently)

**Returns:**
```python
@dataclass
class ExtractEdgesOutput:
    edges: list[EntityEdge]      # Resolved edges
    metadata: dict[str, Any]     # Statistics: extracted, new, merged counts
```

**Key requirement**: Edge extraction uses `extracted_nodes` (original UUIDs), then `resolve_edge_pointers()` remaps to canonical UUIDs from `uuid_map`.

### Stage 3: Extract Attributes

**graphiti-core signature** (`node_operations.py:453-460`):
```python
async def extract_attributes_from_nodes(
    nodes: list[EntityNode],           # RESOLVED nodes from Stage 1
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityNode]:
```

**What Stage 3 needs:**
- Resolved nodes from Stage 1
- Episode and previous episodes (reused)
- Entity type schemas

**Note:** Creates embeddings internally. Returns nodes with attributes, summaries, and embeddings populated.

## Pipeline Stages

```
Journal Text → add_journal() → ExtractNodes → ExtractEdges → ExtractAttributes → GenerateSummaries → [Future: Database]
```

### Stage 1: Extract Nodes

**Module**: `extract_nodes.py`
**Analogous to**: graphiti-core's `extract_nodes()` + `resolve_extracted_nodes()`

**Architecture** (two-layer pattern):
1. **`EntityExtractor`** (dspy.Module): Pure LLM extraction. Sync, no DB. Optimizable.
2. **`ExtractNodes`** (orchestrator): Full pipeline with DB, episode creation, MinHash LSH resolution.

**Default schema**: `pipeline/entity_edge_models.py` defines the four baseline entity types (Person, Place, Organization, Activity) and the curated relationship catalog (`MEETS_AT`, `WORKS_AT`, `HOSTS`, …). Keeping that file in sync with any schema tweaks ensures the runtime pipeline and all DSPy optimizers stay aligned without rerunning expensive compilation jobs.

```python
# Standard usage
extractor = ExtractNodes(group_id="user_123")
result = await extractor(content="Today I met with Sarah...")

# With optimized EntityExtractor
entity_extractor = EntityExtractor()
compiled = optimizer.compile(entity_extractor, trainset=examples)
extractor = ExtractNodes(group_id="user_123", entity_extractor=compiled)
result = await extractor(content="...")
```

**Important**: Stage 1 extracts entity names and types ONLY. Custom attributes (e.g., Person.relationship_type, Activity.activity_type) are extracted in Stage 3, following graphiti-core's separation of concerns.

**Pattern for Future Stages**: Repeat this two-layer design. Create a pure `dspy.Module` for each LLM operation, inject into orchestrator.

### Stage 2: Extract Edges

**Module**: `extract_edges.py`
**Analogous to**: graphiti-core's `extract_edges()` + `resolve_extracted_edges()`

**Purpose**: Extract relationships between resolved entities with supporting facts.

**Architecture** (two-layer pattern):
1. **`EdgeExtractor`** (dspy.Module): Pure LLM extraction. Sync, no DB. Optimizable.
2. **`ExtractEdges`** (orchestrator): Full pipeline with DB, edge building, exact match resolution.

**Design**:
- Extracts relationships + facts inline
- Follows graphiti-core's 3-step flow: extract → remap → resolve
- Uses exact match deduplication only (no fuzzy matching, no semantic search)
- Temporal metadata (valid_at/invalid_at) extracted via LLM
- Returns edges in-memory; Stage 5 writes them alongside episodes/nodes

**Data Flow**:
```
ExtractNodesOutput (from Stage 1)
    ↓ extracted_nodes (original UUIDs)
EdgeExtractor (DSPy) → ExtractedEdges (entity indices)
    ↓
build_entity_edges() → EntityEdges with provisional UUIDs
    ↓ uuid_map (from Stage 1)
resolve_edge_pointers() → EntityEdges with canonical UUIDs
    ↓ existing_edges (from DB)
resolve_edges() → Deduplicated edges
    ↓
ExtractEdgesOutput(edges, metadata)
```

**Usage:**
```python
# Standard usage
edge_extractor = ExtractEdges(group_id="user_123")
result = await edge_extractor(
    episode=extract_result.episode,
    extracted_nodes=extract_result.extracted_nodes,
    resolved_nodes=extract_result.nodes,
    uuid_map=extract_result.uuid_map,
    previous_episodes=extract_result.previous_episodes,
)

# With optimized EdgeExtractor
edge_extractor_module = EdgeExtractor()
compiled = optimizer.compile(edge_extractor_module, trainset=examples)
extractor = ExtractEdges(group_id="user_123", edge_extractor=compiled)
result = await extractor(...)
```

**Important Implementation Details**:

1. **Node Indexing**: LLM extracts edges using `extracted_nodes` (original UUIDs) because indices must match the node list passed to the LLM.

2. **UUID Remapping**: After extraction, `resolve_edge_pointers(edges, uuid_map)` remaps edge source/target UUIDs from provisional → canonical.

3. **Exact Match Deduplication**: Edges keyed by `(source_uuid, target_uuid, edge_name)`. If match exists, merge episode IDs. No fuzzy matching.

4. **Temporal Metadata**: LLM extracts `valid_at` and `invalid_at` as ISO 8601 strings. Simple datetime parsing only (no dateparser validation).

5. **Facts Inline**: Relationships and facts extracted in single LLM call.

**Deferred Features** (for later optimization):
- Contradiction detection (stub: import available but not called)
- Fuzzy edge matching (requires embeddings)
- LLM-based deduplication (requires embeddings)
- Reflexion loop for missed relationships

**Pattern for Future Stages**: Repeat this two-layer design. Create pure `dspy.Module` for each LLM operation, inject into orchestrator.

### Stage 3: Extract Attributes

**Module**: `extract_attributes.py`
**Analogous to**: graphiti-core's `extract_attributes_from_nodes()`

**Purpose**: Extract custom entity attributes based on entity type and episode context.

**Architecture** (two-layer pattern):
1. **`AttributeExtractor`** (dspy.Module): Pure LLM extraction. Sync, no DB. Optimizable.
2. **`ExtractAttributes`** (orchestrator): Full pipeline with entity type resolution, attribute merging.

**Design**:
- For each resolved entity from Stage 1, extract type-specific attributes:
  - **Person**: `relationship_type` (e.g., "friend", "colleague", "family")
  - **Activity**: `activity_type` (e.g., "walk", "therapy session")
  - **Place**: `category` (e.g., "park", "clinic")
  - **Organization**: `category` (e.g., "company", "nonprofit")
- Pass entity's Pydantic model (from `entity_edge_models.py`) as `response_model` to LLM
- LLM extracts attributes based on episode context and previous_episodes
- Results merged into `EntityNode.attributes` dict
- Validation: Pydantic model validation before merging

**Data Flow**:
```
ExtractNodesOutput (from Stage 1)
    ↓ nodes (resolved entities)
    ↓ episode, previous_episodes
AttributeExtractor (DSPy) → Extracted attributes per entity
    ↓
Merge attributes into node.attributes dict
    ↓
ExtractAttributesOutput(nodes, metadata)
```

**Usage:**
```python
# Standard usage
attribute_extractor = ExtractAttributes(group_id="user_123")
result = await attribute_extractor(
    nodes=extract_result.nodes,
    episode=extract_result.episode,
    previous_episodes=extract_result.previous_episodes,
    entity_types=entity_types,
)

# With optimized AttributeExtractor
attribute_extractor_module = AttributeExtractor()
compiled = optimizer.compile(attribute_extractor_module, trainset=examples)
extractor = ExtractAttributes(group_id="user_123", attribute_extractor=compiled)
result = await extractor(...)
```

**Example Flow:**

Stage 1 extracts:
```python
EntityNode(name="Sarah", labels=["Entity", "Person"], attributes={})
EntityNode(name="morning walk", labels=["Entity", "Activity"], attributes={})
```

Stage 3 enriches with type-specific attributes:
```python
# For episode: "Today I met with my friend Sarah. We took a brisk morning walk before work."

# Person entity enriched:
EntityNode(
    name="Sarah",
    labels=["Entity", "Person"],
    attributes={"relationship_type": "friend"}  # Extracted from "my friend Sarah"
)

# Activity entity enriched:
EntityNode(
    name="morning walk",
    labels=["Entity", "Activity"],
    attributes={"activity_type": "walk"}
)
```

**Important Implementation Details**:

1. **Entity Type Resolution**: Extract custom type from node.labels (skip "Entity" base label)

2. **Schema Checking**: Skip nodes with entity types that have no custom fields (`len(model.model_fields) == 0`)

3. **Context Building**: Format episode content and previous_episodes as context for LLM

4. **Attribute Merging**: Use `node.attributes.update()` to preserve existing attributes

5. **Validation**: Validate extracted attributes against Pydantic model before merging

6. **Embeddings**: Deferred to future integration (stub with TODO comment for local Qwen embedder)

**Pattern for Future Stages**: Repeat this two-layer design. Create pure `dspy.Module` for each LLM operation, inject into orchestrator.

### Stage 4: Generate Summaries

**Module**: `generate_summaries.py`
**Analogous to**: graphiti-core's summary generation in `extract_attributes_from_nodes()`

**Purpose**: Generate concise, factual summaries for all entities based on episode context and attributes.

**Architecture** (two-layer pattern):
1. **`SummaryGenerator`** (dspy.Module): Pure LLM extraction. Sync, no DB. Optimizable.
2. **`GenerateSummaries`** (orchestrator): Full pipeline with summary generation, truncation, and validation.

**Design**:
- Generate summaries for ALL entities (not filtered by type)
- Build a `summary_context` dict identical to graphiti-core's `_build_episode_context`:
  ```json
  {
    "node": {
      "name": "...",
      "summary": "<existing <=250 chars>",
      "entity_types": ["Entity", "..."],
      "attributes": {...}
    },
    "episode_content": "...",
    "previous_episodes": ["...", "..."]
  }
  ```
- Serialize the context to JSON and pass it to the DSPy signature so prompt structure stays aligned with graphiti-core.
- Summaries combine information from:
  - Current episode content
  - Previous episodes context
  - Entity attributes (extracted in Stage 3)
  - Existing summary (for updates)
- LLM follows strict guidelines:
  - Output only factual content (no meta-commentary)
  - State facts directly in under 250 characters
  - Combine new info with existing summary
- Summaries truncated at sentence boundaries using graphiti-core utility

**Data Flow**:
```
ExtractAttributesOutput (from Stage 3)
    ↓ nodes (with attributes populated)
    ↓ episode, previous_episodes
SummaryGenerator (DSPy) → EntitySummary per entity
    ↓
Truncate summary to 250 chars (sentence boundary)
    ↓
Update node.summary field
    ↓
GenerateSummariesOutput(nodes, metadata)
```

**Usage:**
```python
# Standard usage
summary_generator = GenerateSummaries(group_id="user_123")
result = await summary_generator(
    nodes=attributes_result.nodes,
    episode=extract_result.episode,
    previous_episodes=extract_result.previous_episodes,
)

# With optimized SummaryGenerator (context-only signature)
summary_generator_module = SummaryGenerator()
compiled = optimizer.compile(summary_generator_module, trainset=examples)
generator = GenerateSummaries(group_id="user_123", summary_generator=compiled)
result = await generator(...)
```

**Example Summaries:**

```python
# Good summaries (factual, concise, under 250 chars)
"Sarah is a friend. Met on 2025-01-06 to discuss AI ethics at Stanford."

"Morning walk on 2025-01-06 with Sarah around Lake Lynn before work."

# Bad summaries (avoid)
"This is the only activity in the context. The user met with Sarah. No other details were provided."
"Based on the messages, Sarah is mentioned. Due to length constraints, other details are omitted."
```

**Important Implementation Details**:

1. **Summary Guidelines**: Embedded directly in DSPy signature from graphiti-core's snippets.py

2. **Truncation**: Double truncation approach:
   - Truncate existing_summary on input (for context)
   - Truncate LLM output on storage (for consistency)

3. **Reuse graphiti-core utilities**:
   - `truncate_at_sentence()` from `graphiti_core.utils.text_utils`
   - `MAX_SUMMARY_CHARS = 250` constant

4. **Process all entities**: No filtering by entity type (summaries for Person, Activity, Place, Organization, generic Entity)

5. **Metadata tracking**: Nodes processed, average summary length, truncation count

6. **Embeddings**: Deferred to future integration (stub with TODO for name embedding generation after summaries)

### Stage 5: Database Persistence

`add_journal()` mirrors graphiti-core's `_process_episode_data()`:

1. Build episodic `MENTIONS` edges via `build_episodic_edges(nodes, episode.uuid, episode.created_at)`.
2. Attach resolved edge UUIDs to the episode (`episode.entity_edges = [edge.uuid ...]`).
3. Persist everything with `pipeline.falkordblite_driver.persist_episode_and_nodes()`, which wraps graphiti-core's `add_nodes_and_edges_bulk()` using the embedded FalkorDB-Lite `GraphDriver` so we stay API-compatible with upstream helpers (embeddings default to empty vectors until a local embedder lands).

Persistence is enabled by default (`persist=True`). Passing `persist=False` keeps the run in-memory and tags `metadata["persistence"] = {"status": "skipped"}` for downstream consumers.

## Optimization

Each pipeline stage can be optimized using DSPy's BootstrapFewShot optimizer to improve prompt quality.

### Pattern

All optimizers follow this structure:

1. **Location**: `pipeline/optimizers/<stage_name>_optimizer.py`
2. **Prompts**: Saved to `pipeline/prompts/<stage_name>.json`
3. **Auto-loading**: Stage modules automatically load optimized prompts if present
4. **Template**: 8-section functional template (no shared utilities)

### Running Optimizers

```bash
# Optimize Stage 1: Entity Extraction
python -m pipeline.optimizers.extract_nodes_optimizer

# Future stages will follow same pattern
# python -m pipeline.optimizers.extract_edges_optimizer
```

**What happens:**
1. Baseline evaluation on training set
2. BootstrapFewShot optimization (5 demos)
3. Optimized evaluation
4. Prompts saved to `pipeline/prompts/<stage>.json`
5. Stage auto-loads optimized prompts on next run

### Optimizer Structure

Each optimizer script contains:

1. **configure_dspy()** - Setup LM and adapter
2. **build_trainset()** - 15-25 training examples
3. **<stage>_metric()** - Stage-specific evaluation metric (e.g., F1 for entities)
4. **optimize()** - Run BootstrapFewShot
5. **evaluate()** - Measure quality
6. **main()** - Orchestrate: baseline → optimize → evaluate → save

### Adding New Stage Optimizers

1. Copy `pipeline/optimizers/extract_nodes_optimizer.py` as template
2. Customize `build_trainset()` with stage-specific examples
3. Implement stage-specific metric function
4. Update imports to use your stage's dspy.Module
5. Set correct `PROMPT_OUTPUT` path
6. Add auto-load logic to your stage's Module.__init__()

## Testing UI

### Gradio Interactive UI

Test individual pipeline stages interactively with `pipeline/gradio_ui.py`:

```bash
python -m pipeline.gradio_ui
```

**Features:**
- Extract entities from journal text
- Toggle deduplication on/off
- View extraction statistics (exact/fuzzy matches, new entities)
- Inspect UUID mappings (provisional → resolved)
- Write results to database
- Reset database for clean testing

**Use cases:**
- Validate entity extraction accuracy
- Test deduplication behavior with/without existing graph data
- Inspect DSPy signature outputs before committing to database

## Debugging

### Debug Logging

Set log level to see detailed execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Shows DSPy adapter fallbacks, generation params
logging.basicConfig(level=logging.INFO)   # Default: shows stage progress, extraction results
```

### Failed Generations

When constrained generation fails (truncated JSON, validation errors), the full output is saved to:

```
debug/failed_generation_YYYYMMDD_HHMMSS.json
```

**Common issues:**
- **Truncated JSON**: Check if `max_tokens` in `settings.py` is too low
- **Invalid schema**: Model may need better prompt or examples
- **Hallucinations**: Model extracting entities from `previous_context` instead of `episode_content`

The `debug/` directory is gitignored.

## Code Reuse Guidelines

When implementing pipeline stages, follow these rules to maximize graphiti-core code reuse:

### Always Import from graphiti-core

**Data structures:**
- `EntityNode`, `EpisodicNode`, `EntityEdge` - Core graph objects
- `EpisodeType` - Episode source classification
- Pydantic models for entity/edge types

**Utilities:**
- DateTime: `ensure_utc()`, `utc_now()`
- Deduplication: `_build_candidate_indexes()`, `_resolve_with_similarity()`
- Entity operations: `resolve_edge_pointers()`
- Validation: `validate_entity_types()` from `graphiti_core.utils.ontology_utils.entity_types_utils`
- Validation: `validate_group_id()`, `validate_excluded_entity_types()` from `graphiti_core.helpers`

**When to use graphiti-core vs. custom:**

| Scenario | Use graphiti-core | Write custom |
|----------|-------------------|--------------|
| Deterministic algorithms | ✓ (deduplication, UUID resolution) | Never |
| Data validation | ✓ (type validators) | Never |
| LLM prompts | Never (DSPy signatures) | ✓ |
| Database I/O | Custom (FalkorDB-specific) | ✓ |
| Graph queries | Custom (no Neo4j driver) | ✓ |
| Embeddings | Future (local Qwen) | ✓ when ready |

**Example decision tree:**
```
Need to deduplicate entities?
  → Import _resolve_with_similarity() from graphiti-core

Need to extract entities from text?
  → Write custom dspy.Module (LLM logic differs)

Need to validate entity type schema?
  → Import validate_entity_types() from graphiti-core

Need to fetch entities from database?
  → Write custom async function using FalkorDB driver
```

### DSPy Signature Design

For each LLM operation, create a custom `dspy.Signature`:
- **Input fields:** Episode content, context, type schemas (as strings or JSON)
- **Output fields:** Pydantic models compatible with graphiti-core (e.g., `ExtractedEntities`)
- **Descriptions:** Tailored for journal entry processing

**Do not** try to reuse graphiti-core's OpenAI prompt strings - DSPy signatures are structurally different.

## Implementation Notes

- **One DSPy config**: Configure once at module import, shared across all stages
- **Async/sync hybrid**: Database queries async, DSPy signatures sync (MLX_LOCK in adapter)
- **Single event loop**: All async stages share one event loop (no conflicts)
- **Code reuse first**: Import graphiti-core utilities before writing custom code
- **Test coverage**: Both end-to-end (`test_add_journal.py`) and per-stage (`test_extract_nodes.py`) tests
