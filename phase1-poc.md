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

## FalkorDB Write Pattern (No Transactions)

FalkorDB has **no multi-query transactions** (see research/06:612-643). Use idempotent writes:

```python
# 1. Create nodes (one query with UNWIND)
node_query = """
UNWIND $nodes AS node
MERGE (n:Entity {uuid: node.uuid})
SET n.name = node.name,
    n.created_at = node.created_at
RETURN n.uuid AS uuid
"""

# 2. Create edges (one query with UNWIND)
edge_query = """
UNWIND $edges AS edge
MATCH (source:Entity {uuid: edge.source_uuid})
MATCH (target:Entity {uuid: edge.target_uuid})
MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
SET r.name = edge.name,
    r.fact = edge.fact,
    r.created_at = edge.created_at
RETURN r.uuid AS uuid
"""
```

Each query is atomic; partial failures require re-running (MERGE makes idempotent).

### Database Reset Pattern

```python
# Clear all graph data (for "Reset Database" button)
driver.execute_query("MATCH (n) DETACH DELETE n")
```

FalkorDBLite persists data to disk; deletion is immediate and permanent.

## Deferred to Later Phases

- Episode chunking/reassembly algorithms
- Embeddings (Qwen3-Embedding-4B)
- Context window management for long journal entries
- Entity deduplication across chunks
- Community detection
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking

