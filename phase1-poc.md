# Phase 1 PoC: Text → Graph in FalkorDBLite

**Goal**: Prove unstructured text can flow through the pipeline to create queryable graph data in FalkorDBLite. No chunking, no embeddings, no transactions—serial and simple for testing.

## Scope Clarifications

- **Episode definition**: Daily journal entry
- **Context window**: Single-window examples only (defer chunking strategy)
- **Extraction**: NER (fast entity detection) → Fact generation → Relationship inference
- **Persistence**: EntityNode + EntityEdge writes to FalkorDBLite (no embeddings)
- **Graphiti bypass**: Skip Graphiti's entity extraction, use only fact/relationship logic

## Data Flow

```
Input Text (fits in single context window)
    ↓
1. NER (DistilBERT) → Entities [PER, ORG, LOC, MISC]
    ↓
2. DSPy Fact Extraction (entities + text → facts)
    ↓
3. DSPy Relationship Inference (facts → edges with source/target/label)
    ↓
4. Build Graphiti Objects (EntityNode, EntityEdge)
    ↓
5. FalkorDBLite Write (bulk save, no transactions)
    ↓
6. Graphviz Preview (visualize in-memory EntityNode/EntityEdge objects; no DB query)
```

## Implementation: `graphiti-poc.py`

New Gradio UI exposing each pipeline stage as **separate components** so data can be inspected at each step:

1. **Input Panel**: Text box for journal entry
2. **Stage 1 Output**: NER entity names (list[str] derived via `distilbert_ner.py`; UI may show raw metadata but only names flow downstream)
3. **Stage 2 Output**: Extracted facts (reuse DSPy signatures from `dspy_outlines/`)
4. **Stage 3 Output**: Inferred relationships (reuse DSPy signatures from `dspy_outlines/`; output carries source/target/name/context)
5. **Stage 4 Output**: Graphiti objects (EntityNode/EntityEdge JSON representations)
6. **Stage 5 Output**: FalkorDBLite write confirmation (UUIDs, counts)
7. **Stage 6 Output**: Graphviz preview of in-memory EntityNode/EntityEdge objects (no FalkorDB queries)

**Stage Execution**:
- Stage 1 auto-refreshes every ~1s, running NER and caching the latest entity name list.
- Only the entity text is cached; discard model-assigned labels (PER/ORG/etc.) and confidence scores before passing to Stage 2.
- Stages 2-5 run only when the user clicks their respective buttons, consuming cached outputs from the prior stage.
- Stage 6 reuses the most recent EntityNode/EntityEdge objects already held in memory; it never issues Cypher queries.

**UI Controls**: Phase 1 extraction parameters exposed as Gradio sliders/checkboxes:
- NER confidence threshold
- Relationship inference temperature
- Enable/disable entity type filters
- Stage run buttons: `Run Facts`, `Run Relationships`, `Build Graphiti`, `Write to Falkor` to step through the pipeline manually

**DSPy/Outlines Integration**:
- Reuse the existing LM/adapter stack from `dspy_outlines.lm` and `dspy_outlines.adapter` (no new configuration layers).
- Import and call the signatures/modules in `dspy_outlines/` for fact and relationship extraction; only add stopgap logic inside `graphiti-poc.py` when something is missing, then plan to upstream it later.

**Database Management**: On Gradio init:
- Load existing FalkorDBLite database from `data/graphiti-poc.db` (create if doesn't exist)
- UI button: "Reset Database" (clear all nodes/edges for clean testing)
- Display current DB stats (node count, edge count)

**Settings Module**: Create `settings.py` for project-wide config:
- FalkorDBLite DB path: `data/graphiti-poc.db` (create `data/` dir if needed)
- Model paths (Qwen, DistilBERT ONNX)
- Phase-specific variables controlled via Gradio UI

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
  - Line 534-620: `predict_entities(text)` function (returns list[dict] with text/label/confidence)
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
  labels: list[str]  # Cypher labels (empty [] for Phase 1)
  created_at: datetime  # Use utc_now()
  name_embedding: list[float] | None  # None for Phase 1
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
  fact_embedding: list[float] | None  # None for Phase 1
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

# Stage 1: NER Output
ner_entities = [
    {"text": "Alice", "label": "PER", "confidence": 0.99},
    {"text": "Microsoft", "label": "ORG", "confidence": 0.95},
    {"text": "Microsoft", "label": "ORG", "confidence": 0.96},  # Duplicate
]

# Stage 2: Filter and deduplicate
# Deduplicate by lowercase entity text (case-insensitive matching)
seen = {}
for ner_entity in ner_entities:
    key = ner_entity["text"].lower()
    if key not in seen or ner_entity["confidence"] > seen[key]["confidence"]:
        seen[key] = ner_entity  # Keep highest confidence version

unique_entities = list(seen.values())
# Result: [{"text": "Alice", ...}, {"text": "Microsoft", "confidence": 0.96}]

# Only pass the entity text downstream
entity_names = [entity["text"] for entity in unique_entities]

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
entity_map = {}  # Map entity_name -> EntityNode (for UUID lookup)

for entity in unique_entities:
    node = EntityNode(
        name=entity["text"],  # Keep original casing
        group_id="phase1-poc",
        labels=[],  # Empty for Phase 1
        name_embedding=None,
        summary="",
        attributes={},
        created_at=utc_now()
    )
    entity_nodes.append(node)
    entity_map[entity["text"].lower()] = node  # Index by lowercase for lookup

entity_edges = []
for rel in relationships.items:
    # Look up EntityNodes by name (case-insensitive)
    source_node = entity_map.get(rel.source.lower())
    target_node = entity_map.get(rel.target.lower())

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
        fact_embedding=None,
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
- **Deduplication**: Case-insensitive by entity name, keep highest confidence
- **Metadata Drop**: After deduplication, pass only the entity text to Stage 2
- **UUID Generation**: Automatic via EntityNode/EntityEdge constructors
- **Lookup Map**: Build `entity_name.lower() → EntityNode` for relationship resolution
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

**Usage Pattern**:
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

Configure DSPy once at startup using the shared components:

```python
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM

dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(),
)
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
    """Query database statistics for UI display."""
    # Count nodes
    result = graph.query("MATCH (n) RETURN count(n) as node_count")
    node_count = result.result_set[0][0] if result.result_set else 0

    # Count edges
    result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
    edge_count = result.result_set[0][0] if result.result_set else 0

    return {"nodes": node_count, "edges": edge_count}

def reset_database():
    """Clear all graph data (DESTRUCTIVE - no confirmation in Phase 1)."""
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

FalkorDB has **no multi-query transactions** (see research/06:612-643). Use idempotent writes:

```python
from datetime import datetime
from uuid import uuid4

# Example: Write EntityNodes and EntityEdges to FalkorDB
def write_entities_and_edges(entity_nodes, entity_edges):
    """Write entities and relationships to FalkorDB."""

    # 1. Convert EntityNode objects to Cypher-compatible dicts
    node_dicts = []
    for node in entity_nodes:
        node_dicts.append({
            "uuid": str(node.uuid),
            "name": node.name,
            "group_id": "phase1-poc",
            "created_at": node.created_at.isoformat()  # Convert datetime to ISO string
        })

    # 2. Write nodes (one query with UNWIND)
    if node_dicts:
        node_query = """
        UNWIND $nodes AS node
        MERGE (n:Entity {uuid: node.uuid})
        SET n.name = node.name,
            n.group_id = node.group_id,
            n.created_at = node.created_at
        RETURN n.uuid AS uuid
        """
        result = graph.query(node_query, {"nodes": node_dicts})
        print(f"Created {len(result.result_set)} nodes")

    # 3. Convert EntityEdge objects to Cypher-compatible dicts
    edge_dicts = []
    for edge in entity_edges:
        edge_dicts.append({
            "uuid": str(edge.uuid),
            "source_uuid": str(edge.source_node_uuid),
            "target_uuid": str(edge.target_node_uuid),
            "name": edge.name,
            "fact": edge.fact,
            "group_id": "phase1-poc",
            "created_at": edge.created_at.isoformat()
        })

    # 4. Write edges (one query with UNWIND)
    if edge_dicts:
        edge_query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {uuid: edge.source_uuid})
        MATCH (target:Entity {uuid: edge.target_uuid})
        MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
        SET r.name = edge.name,
            r.fact = edge.fact,
            r.group_id = edge.group_id,
            r.created_at = edge.created_at
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
- Convert datetimes to ISO strings **before** passing to FalkorDB
- MERGE makes queries idempotent (safe to re-run)
- Each query is atomic; no multi-query transactions
- FalkorDBLite persists to disk automatically

## Deferred to Later Phases

- Episode chunking/reassembly algorithms
- Embeddings (Qwen3-Embedding-4B)
- Context window management for long journal entries
- Entity deduplication across chunks
- Community detection
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking
