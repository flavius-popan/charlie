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
6. Query & Explore (Cypher queries, graph visualization)
```

## Implementation: `graphiti-poc.py`

New Gradio UI exposing each pipeline stage as **separate components** so data can be inspected at each step:

1. **Input Panel**: Text box for journal entry
2. **Stage 1 Output**: NER entities (JSON with labels/confidence from `distilbert_ner.py`)
3. **Stage 2 Output**: Extracted facts (DSPy signature output)
4. **Stage 3 Output**: Inferred relationships (DSPy signature output with source/target/label)
5. **Stage 4 Output**: Graphiti objects (EntityNode/EntityEdge JSON representations)
6. **Stage 5 Output**: FalkorDBLite write confirmation (UUIDs, counts)
7. **Stage 6 Output**: Query results + graph visualization (Graphviz rendering)

**UI Controls**: Phase 1 extraction parameters exposed as Gradio sliders/checkboxes:
- NER confidence threshold
- Relationship inference temperature
- Enable/disable entity type filters

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
- **`.venv/lib/python3.13/site-packages/graphiti_core/models/`**:
  - `nodes/entity_node.py`: EntityNode schema (uuid, name, labels, created_at)
  - `edges/entity_edge.py`: EntityEdge schema (uuid, source_uuid, target_uuid, name, fact, created_at)
  - Note: Skip embedding fields (name_embedding, fact_embedding) in Phase 1

## DSPy Signatures to Create

### Signature 1: Fact Extraction
```python
class FactExtractionSignature(dspy.Signature):
    """Extract facts about entities from text."""
    text: str = dspy.InputField()
    entities: list[str] = dspy.InputField(desc="NER-detected entities")
    facts: list[Fact] = dspy.OutputField()  # Pydantic: entity, fact_text
```

### Signature 2: Relationship Inference
```python
class RelationshipSignature(dspy.Signature):
    """Infer relationships between entities from facts."""
    facts: list[Fact] = dspy.InputField()
    relationships: list[Relationship] = dspy.OutputField()  # Pydantic: source, target, label, fact
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

