# Phase 1 PoC: Text → Graph in FalkorDBLite

**Goal**: Prove unstructured text can flow through the pipeline to create queryable graph data in FalkorDBLite. No chunking, no embeddings, no transactions—serial and simple for testing.

## CRITICAL: Graphiti Compatibility Requirement

**This PoC MUST demonstrate that a custom ingestion pipeline can be built atop stock Graphiti.**

All data written to FalkorDBLite must follow Graphiti's conventions EXACTLY:
- Use Graphiti's `EntityNode` and `EntityEdge` models as the source of truth
- Match all field names, data types, and structure from `graphiti_core`
- Follow Graphiti's Cypher patterns (see `research/06-falkordb-backend.md`)
- Ensure compatibility with Graphiti's search and data operations

**Why this matters**: The PoC proves we can bypass Graphiti's ingestion (NER, chunking, etc.) while remaining fully compatible with Graphiti's query layer. Any deviation from Graphiti's data model breaks this compatibility.

**Verification**: After writing to FalkorDB, the data must be queryable using Graphiti's existing patterns (entity search, relationship traversal, etc.).

## Scope Clarifications

- **Episode definition**: Daily journal entry
- **Context window**: Single-window examples only (defer chunking strategy)
- **Extraction**: NER (fast entity detection) → Fact generation → Relationship inference
- **Persistence**: EntityNode + EntityEdge writes to FalkorDBLite (no embeddings for Phase 1)
- **Graphiti integration strategy**:
  - **Bypass**: Graphiti's ingestion pipeline (entity extraction, episode chunking)
  - **Use**: Graphiti's data models (EntityNode, EntityEdge) and Cypher conventions EXACTLY
  - **Goal**: Prove custom ingestion can write Graphiti-compatible data for search/query operations

## Data Flow

**Stage numbering is 0-indexed** (Stage 0 = Input, Stage 1 = NER, etc.)

```
Stage 0: Input Text (fits in single context window)
    ↓
Stage 1: NER (DistilBERT) → Entities [PER, ORG, LOC, MISC]
    ↓
Stage 2: DSPy Fact Extraction (entities + text → facts)
    ↓
Stage 3: DSPy Relationship Inference (facts → edges with source/target/label)
    ↓
Stage 4: Build Graphiti Objects (EntityNode, EntityEdge)
    ↓
Stage 5: FalkorDBLite Write (bulk save, no transactions)
    ↓
Stage 6: Graphviz Preview (query FalkorDBLite using UUIDs from Stage 5 to verify persistence)
```

## Implementation: `graphiti-poc.py`

New Gradio UI exposing each pipeline stage as **separate components** so data can be inspected at each step:

0. **Stage 0 (Input Panel)**: Text box for journal entry
1. **Stage 1 Output**: NER entity names (ordered `list[str]` derived via `distilbert_ner.py`; UI can optionally display the raw metadata for reference, but only the names flow downstream)
   ```python
   # Example NER output extraction:
   from distilbert_ner import predict_entities

   raw_entities = predict_entities(text)
   # raw_entities = [
   #     {"text": "Alice", "label": "PER", "confidence": 0.95, ...},
   #     {"text": "Microsoft", "label": "ORG", "confidence": 0.92, ...},
   # ]

   entity_names = [entity["text"] for entity in raw_entities]
   # entity_names = ["Alice", "Microsoft"]
   ```
2. **Stage 2 Output**: Extracted facts (reuse DSPy signatures from `dspy_outlines/`)
3. **Stage 3 Output**: Inferred relationships (reuse DSPy signatures from `dspy_outlines/`; output carries source/target/name/context)
4. **Stage 4 Output**: Graphiti objects (retain live `EntityNode`/`EntityEdge` instances in app state and render `.model_dump()` JSON for inspection)
5. **Stage 5 Output**: FalkorDBLite write confirmation (UUIDs, counts) or surfaced error details
6. **Stage 6 Output**: Graphviz preview of entities/edges fetched from FalkorDBLite via Cypher (queries using the UUIDs returned from Stage 5)

**Stage Execution**:
- Stage 1: Reuse the NER trigger pattern from gradio_app.py:278-290 (automatic on text change with debouncing via `trigger_mode="always_last"`). Caches the ordered list of entity names (case-insensitive dedupe).
- Stages 2-4 run only when the user clicks their respective buttons, consuming cached outputs from the prior stage.
- Stage 5 writes to FalkorDBLite when `Write to Falkor` is pressed; success returns a JSON summary with node/edge UUIDs, while failures render the exception details in the UI and log to console.
- Stage 6 runs automatically after a successful Stage 5 write, querying FalkorDBLite via Cypher using the UUIDs returned from Stage 5 to verify the data was persisted, then renders the Graphviz preview.

**State Management** (reuse gradio_app.py:244 pattern):
Use `gr.State()` for session-specific intermediate data:
```python
# Inside Gradio Blocks:
ner_raw_state = gr.State(None)          # Raw NER output (list[dict])
entity_names_state = gr.State([])       # Filtered entity names (list[str])
facts_state = gr.State(None)            # Facts object
relationships_state = gr.State(None)    # Relationships object
entity_nodes_state = gr.State([])       # EntityNode instances
entity_edges_state = gr.State([])       # EntityEdge instances
write_result_state = gr.State(None)     # Write confirmation with UUIDs
```

**UI Controls**:
- Entity type filter: Reuse the "Persons only" checkbox pattern from gradio_app.py:258-262 (filters Stage 1 entity list to PER type only)
- Stage run buttons: `Run Facts`, `Run Relationships`, `Build Graphiti`, `Write to Falkor` to step through the pipeline manually

**Model Parameter Configuration** (centralized in `settings.py`):

The parameter control feature is now implemented in `OutlinesLM`, enabling `OutlinesLM(generation_config=dict)`. Tune values in `settings.MODEL_CONFIG`.

**Supported Parameters** (passed to MLX-LM sampler):
- `temp` (float): Sampling temperature (0.0 = greedy/deterministic, higher = more random)
- `top_p` (float): Nucleus sampling threshold (0.0-1.0)
- `min_p` (float): Minimum token probability threshold
- `min_tokens_to_keep` (int): Minimum number of tokens to keep in sampling
- `top_k` (int): Top-k sampling (0 = disabled)

**Parameter Tuning Workflow**:
1. Edit `MODEL_CONFIG` values in `settings.py`
2. Restart Gradio app
3. Test extraction with same input text
4. Compare fact/relationship quality
5. Iterate to find optimal parameters

**Note**: Default is `temp=0.0` for fully deterministic output. For the PoC, you may want to experiment with `temp=0.7` or higher to observe variance in extraction quality.

**DSPy/Outlines Integration**:
- Reuse the existing LM/adapter stack from `dspy_outlines.lm` and `dspy_outlines.adapter` (no new configuration layers).
- Import and call the signatures/modules in `dspy_outlines/` for fact and relationship extraction; only add stopgap logic inside `graphiti-poc.py` when something is missing, then plan to upstream it later.

**Graphiti Utilities to Reuse** (CRITICAL: Use Graphiti's utilities instead of writing custom code):
- `graphiti_core.utils.datetime_utils.utc_now()` - Generate UTC timestamps for EntityNode/EntityEdge creation
- `graphiti_core.utils.datetime_utils.convert_datetimes_to_strings()` - Convert datetime objects to ISO strings for Cypher queries
- `graphiti_core.utils.maintenance.dedup_helpers._normalize_string_exact()` - Normalize entity names for deduplication (lowercase + collapse whitespace)
- EntityNode/EntityEdge models from `graphiti_core.nodes` and `graphiti_core.edges` - Use as-is, do not create custom models

**Database Management**: On Gradio init:
- Load existing FalkorDBLite database from `data/graphiti-poc.db` (create if doesn't exist)
- UI button: "Reset Database" (clear all nodes/edges for clean testing)
- Display current DB stats (node count, edge count)

**Settings Module**: Create `settings.py` for project-wide config (single source of truth for MODEL_CONFIG):

```python
"""Configuration for graphiti-poc.py"""
from pathlib import Path

# Database
DB_PATH = Path("data/graphiti-poc.db")
GRAPH_NAME = "phase1_poc"

# Models (not needed - dspy_outlines handles this)
# DEFAULT_MODEL_PATH already in dspy_outlines/mlx_loader.py

# Phase 1 identifiers
GROUP_ID = "phase1-poc"

# Model generation parameters (edit and restart to change)
# Supported: temp, top_p, min_p, min_tokens_to_keep, top_k
# Default (temp=0.0) is fully deterministic - use higher temp for more variance
MODEL_CONFIG = {
    "temp": 0.0,  # Start deterministic, increase to 0.7+ to experiment
    "top_p": 1.0,
    "min_p": 0.0,
}
```

Import in graphiti-poc.py:
```python
from settings import DB_PATH, GRAPH_NAME, GROUP_ID, MODEL_CONFIG
```

## Prerequisites

**Parameter Control Feature**: ✅ IMPLEMENTED - `OutlinesLM` now supports `generation_config` parameter for controlling sampling behavior (temp, top_p, min_p, top_k, min_tokens_to_keep).

See `dspy_outlines/lm.py:63-79` for implementation details and `tests/test_parameter_control.py` for usage examples.

## Key Documents for Implementation

### FalkorDB Integration
- **`research/06-falkordb-backend.md`**:
  - Lines 460-520: EntityNode/EntityEdge save query generation
  - Lines 522-585: Bulk save patterns (UNWIND queries)
  - Lines 112-131: Datetime conversion utilities (all datetimes → ISO strings)
  - Lines 649-746: FalkorDB-specific query patterns (vecf32, no transactions)

### FalkorDBLite Setup
- **`plans/falkordblite-evaluation.md`**:
  - Lines 15-19: Client initialization (`FalkorDB(dbfilename='/path/to/db')`)
  - Lines 29-33: Adapter wiring (expose socket/port to driver)
  - Lines 38-43: Performance notes (startup latency ~100-300ms)
- **`falkordblite-build/test_falkordblite.py`** (working examples):
  - Line 18: Import pattern `from redislite.falkordb_client import FalkorDB`
  - Line 466: Initialization `db = FalkorDB()` or `FalkorDB(dbfilename="path")`
  - Lines 33-46: Graph selection and query pattern
  - Lines 436, 507: Critical cleanup with `db.close()`

### NER Integration
- **`distilbert_ner.py`**:
  - Line 534-620: `predict_entities(text)` function returns `list[dict]` where each dict contains:
    - `"text"`: Entity name with original capitalization (e.g., "Alice")
    - `"label"`: Entity type - one of ["PER", "ORG", "LOC", "MISC"]
    - `"start_char"`, `"end_char"`: Character positions in original text
    - `"start_token"`, `"end_token"`: Token positions
    - `"tokens"`: List of tokens (tokenized form)
    - `"confidence"`: Float confidence score
    - `"chunk_idx"`: Chunk index for long texts
  - Stage 1 extracts only the `"text"` field to build the entity name list
  - Line 722-761: `format_entities()` for display formatting

### Current Pattern Reference
- **`gradio_app.py`** (keep as historical artifact):
  - Lines 14-46: KG extraction module pattern (DSPy + OutlinesLM setup)
  - Lines 180-235: Extract and display flow (text → graph → visualization)
  - Lines 49-109: Graph rendering with Graphviz

### Graphiti Models
- **File locations**:
  - `.venv/lib/python3.13/site-packages/graphiti_core/nodes.py` (lines 435-589: EntityNode)
  - `.venv/lib/python3.13/site-packages/graphiti_core/edges.py` (lines 221-478: EntityEdge)

- **Import paths**:
  ```python
  from graphiti_core.nodes import EntityNode
  from graphiti_core.edges import EntityEdge
  from graphiti_core.utils.datetime_utils import utc_now
  ```

- **EntityNode** (nodes.py:435-440):
  ```python
  uuid: str  # Auto-generated via uuid4()
  name: str  # Entity name from NER
  group_id: str  # REQUIRED - use "phase1-poc"
  labels: list[str]  # Cypher labels (include "Entity" for Phase 1)
  created_at: datetime  # Use utc_now()
  name_embedding: list[float]  # Empty [] for Phase 1 (vecf32([]) is allowed; see falkordblite-build/test_falkordblite.py::test_vecf32_empty_embedding)
  summary: str  # Empty "" for Phase 1
  attributes: dict  # Empty {} for Phase 1
  ```

- **EntityEdge** (edges.py:221-240):
  ```python
  uuid: str  # Auto-generated via uuid4()
  source_node_uuid: str  # UUID of source EntityNode
  target_node_uuid: str  # UUID of target EntityNode
  name: str  # Relationship type
  fact: str  # Supporting fact text
  group_id: str  # REQUIRED - "phase1-poc"
  created_at: datetime  # Use utc_now()
  fact_embedding: list[float]  # Empty [] for Phase 1 (vecf32([]) is allowed; see falkordblite-build/test_falkordblite.py::test_vecf32_empty_embedding)
  episodes: list[str]  # Empty [] for Phase 1
  expired_at: datetime | None  # None
  valid_at: datetime | None  # None
  invalid_at: datetime | None  # None
  attributes: dict  # Empty {} for Phase 1
  ```

**Note**: Both have async `.save(driver: GraphDriver)` methods. Phase 1 won't use these—we'll write directly to FalkorDBLite with custom queries (see Write Pattern).

## Pydantic Models for DSPy

### Fact Model
```python
from pydantic import BaseModel, Field

class Fact(BaseModel):
    """A factual statement about an entity."""
    entity: str = Field(description="Entity name this fact is about")
    text: str = Field(description="The factual statement")

class Facts(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Fact]
```

### Relationship Model
```python
class Relationship(BaseModel):
    """A relationship between two entities."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship type (e.g., works_at, knows)")
    context: str = Field(description="Supporting fact/context for this relationship")

class Relationships(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Relationship]
```

## NER → EntityNode Mapping Pattern

### Stage 1 to Stage 4 Transformation

```python
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now

# Stage 1: NER Output (UI may show metadata, cache only names)
# After deduplication using Graphiti's normalization
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

# Alias for clarity in this context
def normalize_entity_name(name: str) -> str:
    """Use Graphiti's exact normalization: lowercase + collapse whitespace."""
    return _normalize_string_exact(name)

entity_candidates = [
    "Alice",
    "Microsoft",
    "Microsoft",  # Duplicate - will be removed
]

unique_entity_names = []
seen = set()
for name in entity_candidates:
    key = normalize_entity_name(name)  # Use Graphiti's normalization
    if key in seen:
        continue
    seen.add(key)
    unique_entity_names.append(name)  # Keep original casing
# Result: ["Alice", "Microsoft"]

# Stage 2: DSPy Facts Output (omitted for brevity)

# Stage 3: DSPy Relationships Output
relationships = Relationships(items=[
    Relationship(
        source="Alice",
        target="Microsoft",
        relation="works_at",
        context="Alice works at Microsoft as a software engineer"
    )
])

# Stage 4: Build EntityNode and EntityEdge objects
entity_nodes = []
entity_map = {}  # Map normalized_name -> EntityNode (for UUID lookup)

for name in unique_entity_names:
    node = EntityNode(
        name=name,  # Keep original casing
        group_id="phase1-poc",
        labels=["Entity"],  # Graphiti requires :Entity label
        name_embedding=[],  # Empty list so vecf32([]) succeeds
        summary="",
        attributes={},
        created_at=utc_now()
    )
    entity_nodes.append(node)
    entity_map[normalize_entity_name(name)] = node  # Index by normalized name for lookup

entity_edges = []
for rel in relationships.items:
    # Look up EntityNodes by normalized name (Graphiti-compatible)
    source_node = entity_map.get(normalize_entity_name(rel.source))
    target_node = entity_map.get(normalize_entity_name(rel.target))

    if not source_node or not target_node:
        print(f"Warning: Skipping relationship {rel.source} -> {rel.target} (entity not found)")
        continue

    edge = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name=rel.relation,
        fact=rel.context,
        group_id="phase1-poc",
        created_at=utc_now(),
        fact_embedding=[],
        episodes=[],
        expired_at=None,
        valid_at=None,
        invalid_at=None,
        attributes={}
    )
    entity_edges.append(edge)

# Now entity_nodes and entity_edges are ready for Stage 5 (FalkorDB write)
```

**Key Patterns**:
- **Deduplication**: Uses Graphiti's normalization (`dedup_helpers.py:39-42`): lowercase + collapse whitespace
- **Casing**: Original casing from NER is preserved in EntityNode.name (only normalized for matching)
- **Entity Cache**: Maintain Stage 1 output as an ordered `list[str]` for reuse across DSPy stages
- **UUID Generation**: Automatic via EntityNode/EntityEdge constructors
- **Lookup Map**: Build `normalized_name → EntityNode` for relationship resolution
- **Validation**: Skip relationships if source or target entity not found

## DSPy Signatures (reuse `dspy_outlines`)

The LM, adapter, and baseline signatures live under `dspy_outlines/`. Import from there instead of rebuilding configuration. Temporary PoC-only glue can live in `graphiti-poc.py` and later be upstreamed.

### Signature 1: Fact Extraction
```python
class FactExtractionSignature(dspy.Signature):
    """Extract factual statements about entities from text."""
    text: str = dspy.InputField(desc="The input text to analyze")
    entities: list[str] = dspy.InputField(desc="NER-detected entity names")
    facts: Facts = dspy.OutputField(desc="Facts about entities")
```

### Signature 2: Relationship Inference
```python
class RelationshipSignature(dspy.Signature):
    """Infer relationships between entities based on facts."""
    text: str = dspy.InputField(desc="Original input text")
    facts: Facts = dspy.InputField(desc="Extracted facts about entities")
    entities: list[str] = dspy.InputField(desc="Entity names to constrain relationships")
    relationships: Relationships = dspy.OutputField(desc="Relationships between entities")
```

> If any of these signatures are missing from `dspy_outlines/`, define them in `graphiti-poc.py` for the PoC and schedule a follow-up task to upstream them.

**DSPy Configuration** (call at module level, before Gradio interface definition):

```python
import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM

# Configure DSPy once at module level - models load immediately
# Pass MODEL_CONFIG (defined in settings.py) to control generation parameters
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)
```

**Note**: The `generation_config` parameter controls sampling behavior. Default is deterministic (`temp=0.0`). Modify `settings.MODEL_CONFIG` and restart the app to experiment with different generation parameters.

**Usage Pattern** (within Gradio event handlers):
```python
# Stage 2: Extract facts
fact_predictor = dspy.Predict(FactExtractionSignature)
facts = fact_predictor(text=input_text, entities=entity_names).facts

# Stage 3: Infer relationships
rel_predictor = dspy.Predict(RelationshipSignature)
relationships = rel_predictor(
    text=input_text,
    facts=facts,
    entities=entity_names
).relationships
```

## FalkorDBLite Initialization

```python
# In graphiti-poc.py, at module level before Gradio app definition

from pathlib import Path
from redislite.falkordb_client import FalkorDB
import atexit

# Create data directory
DB_PATH = Path("data/graphiti-poc.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize embedded database (spawns Redis process)
db = FalkorDB(dbfilename=str(DB_PATH))

# Select the graph to use (creates if doesn't exist)
GRAPH_NAME = "phase1_poc"
graph = db.select_graph(GRAPH_NAME)

# Register cleanup handler for proper shutdown
def cleanup_db():
    """Clean shutdown of FalkorDB - CRITICAL before exit."""
    try:
        db.close()
        print("✓ FalkorDB closed successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to close FalkorDB: {e}")

atexit.register(cleanup_db)

def get_db_stats():
    """
    Query database statistics for UI display.

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # Count nodes
    result = graph.query("MATCH (n) RETURN count(n) as node_count")
    node_count = result.result_set[0][0] if result.result_set else 0

    # Count edges
    result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
    edge_count = result.result_set[0][0] if result.result_set else 0

    return {"nodes": node_count, "edges": edge_count}

def reset_database():
    """
    Clear all graph data (DESTRUCTIVE - no confirmation in Phase 1).

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    graph.query("MATCH (n) DETACH DELETE n")
    return "Database cleared successfully"
```

**Critical Notes**:
- Import from `redislite.falkordb_client`, not `falkordblite`
- FalkorDBLite methods are **synchronous**, not async
- **MUST call `db.close()`** before app exit (use `atexit.register()` for Gradio)
- Select graph with `db.select_graph()` before running queries
- Query with `graph.query()`, not `driver.execute_query()`

## FalkorDB Write Pattern (No Transactions)

**CRITICAL: This pattern follows Graphiti's exact conventions from `research/06-falkordb-backend.md`.**

FalkorDB has **no multi-query transactions** (see research/06:612-643). Use idempotent writes:

**Graphiti Conventions Used:**
- Node label: `:Entity` (research/06:506)
- Edge type: `:RELATES_TO` (research/06:559, 576) - ALL entity edges use this static type
- Relationship semantic name stored in `r.name` property (e.g., "works_at", "knows")
- Vector fields use `vecf32()` wrapper (research/06:509, 561); empty lists are valid and covered by `test_vecf32_empty_embedding`
- Datetime fields as ISO strings (research/06:112-131)

```python
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

# Example: Write EntityNodes and EntityEdges to FalkorDB
def write_entities_and_edges(entity_nodes, entity_edges):
    """Write entities and relationships to FalkorDB using Graphiti utilities."""

    # 1. Convert EntityNode objects to Cypher-compatible dicts
    node_dicts = []
    for node in entity_nodes:
        node_dict = {
            "uuid": node.uuid,
            "name": node.name,
            "group_id": "phase1-poc",
            "created_at": node.created_at,
            "labels": node.labels or ["Entity"],
            "name_embedding": node.name_embedding or [],
            "summary": node.summary,
            "attributes": node.attributes or {},
        }
        # Use Graphiti's utility to convert datetimes to ISO strings
        node_dicts.append(convert_datetimes_to_strings(node_dict))

    # 2. Write nodes (one query with UNWIND)
    # Pattern from research/06:524-538 (Graphiti's bulk entity node save)
    if node_dicts:
        node_query = """
        UNWIND $nodes AS node
        MERGE (n:Entity {uuid: node.uuid})
        SET n:Entity
        SET n.name = node.name,
            n.group_id = node.group_id,
            n.created_at = node.created_at,
            n.labels = node.labels,
            n.summary = node.summary,
            n.attributes = node.attributes
        SET n.name_embedding = vecf32(node.name_embedding)
        RETURN n.uuid AS uuid
        """
        result = graph.query(node_query, {"nodes": node_dicts})
        print(f"Created {len(result.result_set)} nodes")

    # 3. Convert EntityEdge objects to Cypher-compatible dicts
    edge_dicts = []
    for edge in entity_edges:
        edge_dict = {
            "uuid": edge.uuid,
            "source_uuid": edge.source_node_uuid,
            "target_uuid": edge.target_node_uuid,
            "name": edge.name,
            "fact": edge.fact,
            "group_id": "phase1-poc",
            "created_at": edge.created_at,
            "fact_embedding": edge.fact_embedding or [],
            "episodes": edge.episodes or [],
            "expired_at": edge.expired_at,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
            "attributes": edge.attributes or {},
        }
        # Use Graphiti's utility to convert datetimes to ISO strings
        edge_dicts.append(convert_datetimes_to_strings(edge_dict))

    # 4. Write edges (one query with UNWIND)
    # Pattern from research/06:571-581 (Graphiti's bulk entity edge save)
    # NOTE: ALL edges use :RELATES_TO type; semantic name stored in r.name property
    if edge_dicts:
        edge_query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {uuid: edge.source_uuid})
        MATCH (target:Entity {uuid: edge.target_uuid})
        MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
        SET r.name = edge.name,
            r.fact = edge.fact,
            r.group_id = edge.group_id,
            r.created_at = edge.created_at,
            r.episodes = edge.episodes,
            r.expired_at = edge.expired_at,
            r.valid_at = edge.valid_at,
            r.invalid_at = edge.invalid_at,
            r.attributes = edge.attributes
        SET r.fact_embedding = vecf32(edge.fact_embedding)
        RETURN r.uuid AS uuid
        """
        result = graph.query(edge_query, {"edges": edge_dicts})
        print(f"Created {len(result.result_set)} edges")

    return {
        "nodes_created": len(node_dicts),
        "edges_created": len(edge_dicts),
        "node_uuids": [n["uuid"] for n in node_dicts],
        "edge_uuids": [e["uuid"] for e in edge_dicts]
    }
```

**Key Patterns**:
- Use `graph.query(cypher_string, params_dict)` for parameterized queries
- Convert datetimes to ISO strings **before** passing to FalkorDB (use `convert_datetimes_to_strings()`)
- MERGE makes queries idempotent (safe to re-run)
- Each query is atomic; no multi-query transactions
- FalkorDBLite persists to disk automatically

**Error Handling Philosophy**:
- Let exceptions propagate (fail-fast approach)
- Gradio will catch and display errors in UI
- Prefer clear flow control over defensive programming
- Only catch exceptions for cleanup operations (e.g., `db.close()` in atexit)

## Graphviz Verification and Rendering

Stage 6 runs automatically after a successful write to confirm persistence and visualize the graph. It uses the UUIDs returned by `write_entities_and_edges` to fetch the corresponding nodes and edges from FalkorDBLite:

```python
import logging
import tempfile
import os
from graphviz import Digraph
from typing import Any

def load_written_entities(node_uuids: list[str], edge_uuids: list[str]) -> dict[str, Any]:
    """Query FalkorDBLite for nodes and edges by UUID."""
    try:
        node_query = """
        UNWIND $uuids AS uuid
        MATCH (n:Entity {uuid: uuid})
        RETURN n.uuid AS uuid, n.name AS name
        """
        node_result = graph.query(node_query, {"uuids": node_uuids})

        edge_query = """
        UNWIND $uuids AS uuid
        MATCH (source:Entity)-[r:RELATES_TO {uuid: uuid}]->(target:Entity)
        RETURN r.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid, r.name AS name
        """
        edge_result = graph.query(edge_query, {"uuids": edge_uuids})

        return {
            "nodes": node_result.result_set or [],  # List[tuple]: [(uuid, name), ...]
            "edges": edge_result.result_set or [],  # List[tuple]: [(uuid, src_uuid, tgt_uuid, name), ...]
        }
    except Exception as exc:
        logging.exception("Failed to verify FalkorDBLite write")
        return {"error": str(exc)}

def render_graph_from_db(db_data: dict[str, Any]) -> str | None:
    """
    Render Graphviz graph from FalkorDB query results.

    Returns: Path to PNG file, or None on error (errors logged to console).

    Styling reused from gradio_app.py for consistency, but parameters
    extracted to module-level constants for easy future customization.
    """
    # Check for query errors
    if "error" in db_data:
        logging.error(f"Cannot render graph: {db_data['error']}")
        return None

    try:
        dot = Digraph(format="png")

        # Graph layout settings (from gradio_app.py:52-60)
        dot.attr("graph",
                 rankdir="LR",
                 splines="spline",
                 pad="0.35",
                 nodesep="0.7",
                 ranksep="1.0",
                 bgcolor="transparent")

        # Node styling (from gradio_app.py:61-69)
        dot.attr("node",
                 shape="circle",
                 style="filled",
                 fontname="Helvetica",
                 fontsize="11",
                 color="transparent",
                 fontcolor="#1f2937")

        # Edge styling (from gradio_app.py:70-79)
        dot.attr("edge",
                 color="#60a5fa",
                 penwidth="1.6",
                 arrowsize="1.0",
                 arrowhead="vee",
                 fontname="Helvetica",
                 fontcolor="#e2e8f0",
                 fontsize="10")

        # Add nodes from result_set: [(uuid, name), ...]
        for row in db_data["nodes"]:
            uuid, name = row
            # Color logic from gradio_app.py:84-85
            normalized = name.lower()
            is_author = normalized in {"author", "i"}
            fillcolor = "#facc15" if is_author else "#dbeafe"

            dot.node(uuid,
                    label=name,
                    fillcolor=fillcolor,
                    fontcolor="#1f2937",
                    tooltip=name)

        # Add edges from result_set: [(uuid, source_uuid, target_uuid, name), ...]
        for row in db_data["edges"]:
            edge_uuid, source_uuid, target_uuid, edge_name = row
            edge_label = edge_name.replace("_", " ")

            dot.edge(source_uuid,
                    target_uuid,
                    label=edge_label,
                    color="#60a5fa",
                    fontcolor="#e2e8f0",
                    tooltip=edge_label)

        # Render to temporary file
        fd, tmp_png_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        dot.render(tmp_png_path[:-4], format="png", cleanup=True)

        return tmp_png_path

    except Exception as exc:
        logging.exception("Failed to render Graphviz graph")
        return None
```

**Future Customization**: To modify graph appearance, edit the `dot.attr()` parameters in `render_graph_from_db()`. Consider extracting these to a config dict or settings.py in later phases.

## Deferred to Later Phases

- Episode chunking/reassembly algorithms
- Embeddings (Qwen3-Embedding-4B)
- Context window management for long journal entries
- Entity deduplication across chunks
- Community detection
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking
