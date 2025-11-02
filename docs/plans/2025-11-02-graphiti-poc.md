# Graphiti PoC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Gradio UI (`graphiti-poc.py`) that demonstrates text-to-graph pipeline with FalkorDBLite, following Graphiti's data model conventions exactly.

**Architecture:** Six-stage pipeline exposed as separate UI components: (0) Input text → (1) NER entities → (2) DSPy facts → (3) DSPy relationships → (4) Graphiti objects → (5) FalkorDB write → (6) Graphviz verification. Each stage's output is cached in `gr.State()` for downstream consumption.

**Tech Stack:** Gradio, DSPy + Outlines + MLX (via `dspy_outlines/`), FalkorDBLite, Graphiti-core models, DistilBERT NER, Graphviz

---

## Task 1: Create Settings Configuration

**Files:**
- Create: `settings.py`

**Step 1: Write settings.py**

Create the configuration module with all project-wide settings:

```python
"""Configuration for graphiti-poc.py"""
from pathlib import Path

# Database
DB_PATH = Path("data/graphiti-poc.db")
GRAPH_NAME = "phase1_poc"

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

**Step 2: Verify import works**

Run: `python -c "from settings import DB_PATH, GRAPH_NAME, GROUP_ID, MODEL_CONFIG; print('Settings loaded:', MODEL_CONFIG)"`

Expected: `Settings loaded: {'temp': 0.0, 'top_p': 1.0, 'min_p': 0.0}`

**Step 3: Commit**

```bash
git add settings.py
git commit -m "feat: add settings.py for PoC configuration"
```

---

## Task 2: Create Pydantic Models for DSPy

**Files:**
- Create: `models.py`

**Step 1: Write Pydantic models**

Create models for facts and relationships:

```python
"""Pydantic models for DSPy signature outputs."""
from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A factual statement about an entity."""
    entity: str = Field(description="Entity name this fact is about")
    text: str = Field(description="The factual statement")


class Facts(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Fact]


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

**Step 2: Verify import works**

Run: `python -c "from models import Fact, Facts, Relationship, Relationships; print('Models loaded')"`

Expected: `Models loaded`

**Step 3: Commit**

```bash
git add models.py
git commit -m "feat: add Pydantic models for facts and relationships"
```

---

## Task 3: Create DSPy Signatures

**Files:**
- Create: `signatures.py`

**Step 1: Write DSPy signatures**

Create signatures for fact extraction and relationship inference:

```python
"""DSPy signatures for knowledge graph extraction."""
import dspy
from models import Facts, Relationships


class FactExtractionSignature(dspy.Signature):
    """Extract factual statements about entities from text."""
    text: str = dspy.InputField(desc="The input text to analyze")
    entities: list[str] = dspy.InputField(desc="NER-detected entity names")
    facts: Facts = dspy.OutputField(desc="Facts about entities")


class RelationshipSignature(dspy.Signature):
    """Infer relationships between entities based on facts."""
    text: str = dspy.InputField(desc="Original input text")
    facts: Facts = dspy.InputField(desc="Extracted facts about entities")
    entities: list[str] = dspy.InputField(desc="Entity names to constrain relationships")
    relationships: Relationships = dspy.OutputField(desc="Relationships between entities")
```

**Step 2: Verify import works**

Run: `python -c "from signatures import FactExtractionSignature, RelationshipSignature; print('Signatures loaded')"`

Expected: `Signatures loaded`

**Step 3: Commit**

```bash
git add signatures.py
git commit -m "feat: add DSPy signatures for extraction pipeline"
```

---

## Task 4: Create FalkorDB Initialization Module

**Files:**
- Create: `falkordb_utils.py`

**Step 1: Write FalkorDB initialization code**

Create module with database initialization and cleanup:

```python
"""FalkorDB initialization and utility functions."""
import atexit
import logging
from pathlib import Path
from redislite.falkordb_client import FalkorDB
from settings import DB_PATH, GRAPH_NAME

# Create data directory
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize embedded database (spawns Redis process)
db = FalkorDB(dbfilename=str(DB_PATH))

# Select the graph to use (creates if doesn't exist)
graph = db.select_graph(GRAPH_NAME)


def cleanup_db():
    """Clean shutdown of FalkorDB - CRITICAL before exit."""
    try:
        db.close()
        print("✓ FalkorDB closed successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to close FalkorDB: {e}")


# Register cleanup handler for proper shutdown
atexit.register(cleanup_db)


def get_db_stats() -> dict[str, int]:
    """
    Query database statistics for UI display.

    Returns:
        dict with 'nodes' and 'edges' counts

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # Count nodes
    result = graph.query("MATCH (n) RETURN count(n) as node_count")
    node_count = result.result_set[0][0] if result.result_set else 0

    # Count edges
    result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
    edge_count = result.result_set[0][0] if result.result_set else 0

    return {"nodes": node_count, "edges": edge_count}


def reset_database() -> str:
    """
    Clear all graph data (DESTRUCTIVE - no confirmation in Phase 1).

    Returns:
        Success message

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    graph.query("MATCH (n) DETACH DELETE n")
    return "Database cleared successfully"
```

**Step 2: Verify database initialization**

Run: `python -c "from falkordb_utils import get_db_stats; print('DB Stats:', get_db_stats())"`

Expected: `DB Stats: {'nodes': 0, 'edges': 0}` (and `✓ FalkorDB closed successfully` on exit)

**Step 3: Commit**

```bash
git add falkordb_utils.py data/.gitkeep
git commit -m "feat: add FalkorDB initialization and utilities"
```

---

## Task 5: Create FalkorDB Write Functions

**Files:**
- Modify: `falkordb_utils.py`

**Step 1: Add write function to falkordb_utils.py**

Add the function to write EntityNode and EntityEdge objects to FalkorDB:

```python
# Add these imports at the top
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from typing import Any


# Add this function at the end of the file
def write_entities_and_edges(
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge]
) -> dict[str, Any]:
    """
    Write entities and relationships to FalkorDB using Graphiti utilities.

    Args:
        entity_nodes: List of EntityNode objects
        entity_edges: List of EntityEdge objects

    Returns:
        dict with counts and UUIDs of created nodes/edges

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # 1. Convert EntityNode objects to Cypher-compatible dicts
    node_dicts = []
    for node in entity_nodes:
        node_dict = {
            "uuid": node.uuid,
            "name": node.name,
            "group_id": node.group_id,
            "created_at": node.created_at,
            "labels": node.labels or ["Entity"],
            "name_embedding": node.name_embedding or [],
            "summary": node.summary,
            "attributes": node.attributes or {},
        }
        # Use Graphiti's utility to convert datetimes to ISO strings
        node_dicts.append(convert_datetimes_to_strings(node_dict))

    # 2. Write nodes (one query with UNWIND)
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
        logging.info(f"Created {len(result.result_set)} nodes")

    # 3. Convert EntityEdge objects to Cypher-compatible dicts
    edge_dicts = []
    for edge in entity_edges:
        edge_dict = {
            "uuid": edge.uuid,
            "source_uuid": edge.source_node_uuid,
            "target_uuid": edge.target_node_uuid,
            "name": edge.name,
            "fact": edge.fact,
            "group_id": edge.group_id,
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
        logging.info(f"Created {len(result.result_set)} edges")

    return {
        "nodes_created": len(node_dicts),
        "edges_created": len(edge_dicts),
        "node_uuids": [n["uuid"] for n in node_dicts],
        "edge_uuids": [e["uuid"] for e in edge_dicts]
    }
```

**Step 2: Verify function imports**

Run: `python -c "from falkordb_utils import write_entities_and_edges; print('Write function loaded')"`

Expected: `Write function loaded`

**Step 3: Commit**

```bash
git add falkordb_utils.py
git commit -m "feat: add FalkorDB write function for Graphiti objects"
```

---

## Task 6: Create Graphviz Rendering Module

**Files:**
- Create: `graphviz_utils.py`

**Step 1: Write Graphviz rendering functions**

Create module for graph visualization:

```python
"""Graphviz visualization utilities."""
import logging
import tempfile
import os
from graphviz import Digraph
from typing import Any
from falkordb_utils import graph


def load_written_entities(node_uuids: list[str], edge_uuids: list[str]) -> dict[str, Any]:
    """
    Query FalkorDBLite for nodes and edges by UUID.

    Args:
        node_uuids: List of node UUIDs to fetch
        edge_uuids: List of edge UUIDs to fetch

    Returns:
        dict with 'nodes' and 'edges' result sets, or 'error' key on failure
    """
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

    Args:
        db_data: Dict with 'nodes' and 'edges' keys from load_written_entities()

    Returns:
        Path to PNG file, or None on error (errors logged to console)
    """
    # Check for query errors
    if "error" in db_data:
        logging.error(f"Cannot render graph: {db_data['error']}")
        return None

    try:
        dot = Digraph(format="png")

        # Graph layout settings
        dot.attr("graph",
                 rankdir="LR",
                 splines="spline",
                 pad="0.35",
                 nodesep="0.7",
                 ranksep="1.0",
                 bgcolor="transparent")

        # Node styling
        dot.attr("node",
                 shape="circle",
                 style="filled",
                 fontname="Helvetica",
                 fontsize="11",
                 color="transparent",
                 fontcolor="#1f2937")

        # Edge styling
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
            # Color logic: highlight "author" or "I"
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

**Step 2: Verify import works**

Run: `python -c "from graphviz_utils import load_written_entities, render_graph_from_db; print('Graphviz utils loaded')"`

Expected: `Graphviz utils loaded`

**Step 3: Commit**

```bash
git add graphviz_utils.py
git commit -m "feat: add Graphviz rendering utilities"
```

---

## Task 7: Create Entity Processing Utilities

**Files:**
- Create: `entity_utils.py`

**Step 1: Write entity normalization and deduplication functions**

Create utilities for entity name processing:

```python
"""Entity processing utilities using Graphiti conventions."""
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now
from models import Relationships
from settings import GROUP_ID


def normalize_entity_name(name: str) -> str:
    """Use Graphiti's exact normalization: lowercase + collapse whitespace."""
    return _normalize_string_exact(name)


def deduplicate_entities(entity_candidates: list[str]) -> list[str]:
    """
    Deduplicate entity names using Graphiti's normalization.

    Args:
        entity_candidates: List of entity names (may contain duplicates)

    Returns:
        List of unique entity names (preserving original casing)
    """
    unique_entity_names = []
    seen = set()
    for name in entity_candidates:
        key = normalize_entity_name(name)
        if key in seen:
            continue
        seen.add(key)
        unique_entity_names.append(name)  # Keep original casing
    return unique_entity_names


def build_entity_nodes(entity_names: list[str]) -> tuple[list[EntityNode], dict[str, EntityNode]]:
    """
    Build EntityNode objects from entity names.

    Args:
        entity_names: Deduplicated list of entity names

    Returns:
        Tuple of (list of EntityNode objects, dict mapping normalized_name -> EntityNode)
    """
    entity_nodes = []
    entity_map = {}

    for name in entity_names:
        node = EntityNode(
            name=name,
            group_id=GROUP_ID,
            labels=["Entity"],
            name_embedding=[],
            summary="",
            attributes={},
            created_at=utc_now()
        )
        entity_nodes.append(node)
        entity_map[normalize_entity_name(name)] = node

    return entity_nodes, entity_map


def build_entity_edges(
    relationships: Relationships,
    entity_map: dict[str, EntityNode]
) -> list[EntityEdge]:
    """
    Build EntityEdge objects from relationships.

    Args:
        relationships: Relationships object from DSPy
        entity_map: Dict mapping normalized entity name -> EntityNode

    Returns:
        List of EntityEdge objects (relationships with missing entities are skipped)
    """
    entity_edges = []

    for rel in relationships.items:
        # Look up EntityNodes by normalized name
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
            group_id=GROUP_ID,
            created_at=utc_now(),
            fact_embedding=[],
            episodes=[],
            expired_at=None,
            valid_at=None,
            invalid_at=None,
            attributes={}
        )
        entity_edges.append(edge)

    return entity_edges
```

**Step 2: Verify import works**

Run: `python -c "from entity_utils import normalize_entity_name, deduplicate_entities, build_entity_nodes, build_entity_edges; print('Entity utils loaded')"`

Expected: `Entity utils loaded`

**Step 3: Commit**

```bash
git add entity_utils.py
git commit -m "feat: add entity processing utilities"
```

---

## Task 8: Create Gradio UI Skeleton

**Files:**
- Create: `graphiti-poc.py`

**Step 1: Write basic Gradio app structure**

Create the main application file with UI skeleton:

```python
"""Phase 1 PoC: Text → Graph in FalkorDBLite with Gradio UI."""
import gradio as gr
import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM
from settings import MODEL_CONFIG, DB_PATH
from falkordb_utils import get_db_stats, reset_database

# Configure DSPy once at module level
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)

# Build Gradio interface
with gr.Blocks(title="Phase 1 PoC: Graphiti Pipeline") as app:
    gr.Markdown("# Phase 1 PoC: Text → Graph in FalkorDBLite")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    # Database stats display
    with gr.Row():
        db_stats_display = gr.Textbox(
            label="Database Stats",
            value=lambda: str(get_db_stats()),
            interactive=False
        )
        reset_btn = gr.Button("Reset Database", variant="stop")

    # Stage 0: Input
    gr.Markdown("## Stage 0: Input Text")
    input_text = gr.Textbox(
        label="Journal Entry",
        placeholder="Enter text here...",
        lines=5
    )

    # Stage 1: NER
    gr.Markdown("## Stage 1: NER Entities")
    ner_output = gr.Textbox(label="Entity Names", interactive=False)
    persons_only_filter = gr.Checkbox(label="Persons only", value=False)

    # Stage 2: Facts
    gr.Markdown("## Stage 2: Fact Extraction")
    run_facts_btn = gr.Button("Run Facts", variant="primary")
    facts_output = gr.JSON(label="Extracted Facts")

    # Stage 3: Relationships
    gr.Markdown("## Stage 3: Relationship Inference")
    run_relationships_btn = gr.Button("Run Relationships", variant="primary")
    relationships_output = gr.JSON(label="Inferred Relationships")

    # Stage 4: Graphiti Objects
    gr.Markdown("## Stage 4: Build Graphiti Objects")
    build_graphiti_btn = gr.Button("Build Graphiti Objects", variant="primary")
    graphiti_output = gr.JSON(label="EntityNode + EntityEdge Objects")

    # Stage 5: FalkorDB Write
    gr.Markdown("## Stage 5: Write to FalkorDB")
    write_falkor_btn = gr.Button("Write to Falkor", variant="primary")
    write_output = gr.JSON(label="Write Confirmation")

    # Stage 6: Graphviz Preview
    gr.Markdown("## Stage 6: Graphviz Verification")
    graphviz_output = gr.Image(label="Graph Visualization")

    # State management
    ner_raw_state = gr.State(None)
    entity_names_state = gr.State([])
    facts_state = gr.State(None)
    relationships_state = gr.State(None)
    entity_nodes_state = gr.State([])
    entity_edges_state = gr.State([])
    write_result_state = gr.State(None)

    # Event handlers (placeholder functions)
    def on_reset_db():
        msg = reset_database()
        return str(get_db_stats())

    reset_btn.click(
        on_reset_db,
        outputs=[db_stats_display]
    )

if __name__ == "__main__":
    app.launch()
```

**Step 2: Test Gradio app launches**

Run: `python graphiti-poc.py`

Expected: Gradio app opens in browser with all UI elements visible (no functionality yet)

**Step 3: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: add Gradio UI skeleton for PoC"
```

---

## Task 9: Implement Stage 1 (NER Integration)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add NER processing function**

Add the NER stage handler before the `with gr.Blocks()` section:

```python
# Add this import at the top
from distilbert_ner import predict_entities

# Add this function before the Gradio interface definition
def process_ner(text: str, persons_only: bool):
    """
    Stage 1: Extract entities using NER.

    Returns: (entity_names_list, raw_ner_output, display_string)
    """
    if not text.strip():
        return [], None, ""

    # Run NER
    raw_entities = predict_entities(text)

    # Filter by type if requested
    if persons_only:
        filtered = [e for e in raw_entities if e["label"] == "PER"]
    else:
        filtered = raw_entities

    # Extract entity names and deduplicate
    from entity_utils import deduplicate_entities
    entity_names = [e["text"] for e in filtered]
    unique_names = deduplicate_entities(entity_names)

    # Format for display
    display = "\n".join(unique_names) if unique_names else "(no entities found)"

    return unique_names, raw_entities, display
```

**Step 2: Wire up NER event handler**

Replace the placeholder event handlers section with:

```python
    # Event handlers

    # Stage 1: NER (automatic on text change)
    def on_text_change(text, persons_only):
        entity_names, raw_ner, display = process_ner(text, persons_only)
        return display, entity_names, raw_ner

    input_text.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state],
        trigger_mode="always_last"
    )

    persons_only_filter.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state]
    )

    def on_reset_db():
        msg = reset_database()
        return str(get_db_stats())

    reset_btn.click(
        on_reset_db,
        outputs=[db_stats_display]
    )
```

**Step 3: Test NER stage**

Run: `python graphiti-poc.py`

Expected: Type text in input box → entity names appear in Stage 1 output

Example input: "Alice works at Microsoft in Seattle."

Expected output:
```
Alice
Microsoft
Seattle
```

**Step 4: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 1 NER integration"
```

---

## Task 10: Implement Stage 2 (Fact Extraction)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add fact extraction function**

Add before the Gradio interface definition:

```python
# Add this import at the top
from signatures import FactExtractionSignature

# Add this function
def extract_facts(text: str, entity_names: list[str]):
    """
    Stage 2: Extract facts using DSPy.

    Returns: (Facts object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        fact_predictor = dspy.Predict(FactExtractionSignature)
        facts = fact_predictor(text=text, entities=entity_names).facts

        # Convert to JSON for display
        facts_json = {
            "items": [
                {"entity": f.entity, "text": f.text}
                for f in facts.items
            ]
        }

        return facts, facts_json
    except Exception as e:
        return None, {"error": str(e)}
```

**Step 2: Wire up Stage 2 button**

Add to event handlers section:

```python
    # Stage 2: Fact Extraction
    def on_run_facts(text, entity_names):
        facts, facts_json = extract_facts(text, entity_names)
        return facts_json, facts

    run_facts_btn.click(
        on_run_facts,
        inputs=[input_text, entity_names_state],
        outputs=[facts_output, facts_state]
    )
```

**Step 3: Test fact extraction**

Run: `python graphiti-poc.py`

Expected:
1. Enter text: "Alice works at Microsoft in Seattle."
2. Wait for NER entities to appear
3. Click "Run Facts" button
4. See facts JSON appear in Stage 2 output

**Step 4: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 2 fact extraction"
```

---

## Task 11: Implement Stage 3 (Relationship Inference)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add relationship inference function**

Add before the Gradio interface definition:

```python
# Add this import at the top
from signatures import RelationshipSignature

# Add this function
def infer_relationships(text: str, facts, entity_names: list[str]):
    """
    Stage 3: Infer relationships using DSPy.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not facts or not entity_names:
        return None, {"error": "Need text, facts, and entities"}

    try:
        rel_predictor = dspy.Predict(RelationshipSignature)
        relationships = rel_predictor(
            text=text,
            facts=facts,
            entities=entity_names
        ).relationships

        # Convert to JSON for display
        rels_json = {
            "items": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "context": r.context
                }
                for r in relationships.items
            ]
        }

        return relationships, rels_json
    except Exception as e:
        return None, {"error": str(e)}
```

**Step 2: Wire up Stage 3 button**

Add to event handlers section:

```python
    # Stage 3: Relationship Inference
    def on_run_relationships(text, facts, entity_names):
        relationships, rels_json = infer_relationships(text, facts, entity_names)
        return rels_json, relationships

    run_relationships_btn.click(
        on_run_relationships,
        inputs=[input_text, facts_state, entity_names_state],
        outputs=[relationships_output, relationships_state]
    )
```

**Step 3: Test relationship inference**

Run: `python graphiti-poc.py`

Expected:
1. Enter text and run Stages 1-2
2. Click "Run Relationships"
3. See relationships JSON with source/target/relation/context

**Step 4: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 3 relationship inference"
```

---

## Task 12: Implement Stage 4 (Build Graphiti Objects)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add Graphiti object builder function**

Add before the Gradio interface definition:

```python
# Add this import at the top
from entity_utils import build_entity_nodes, build_entity_edges

# Add this function
def build_graphiti_objects(entity_names: list[str], relationships):
    """
    Stage 4: Build EntityNode and EntityEdge objects.

    Returns: (entity_nodes, entity_edges, JSON for display)
    """
    if not entity_names or not relationships:
        return [], [], {"error": "Need entities and relationships"}

    try:
        # Build nodes and entity map
        entity_nodes, entity_map = build_entity_nodes(entity_names)

        # Build edges
        entity_edges = build_entity_edges(relationships, entity_map)

        # Convert to JSON for display
        graphiti_json = {
            "nodes": [n.model_dump() for n in entity_nodes],
            "edges": [e.model_dump() for e in entity_edges]
        }

        return entity_nodes, entity_edges, graphiti_json
    except Exception as e:
        return [], [], {"error": str(e)}
```

**Step 2: Wire up Stage 4 button**

Add to event handlers section:

```python
    # Stage 4: Build Graphiti Objects
    def on_build_graphiti(entity_names, relationships):
        nodes, edges, graphiti_json = build_graphiti_objects(entity_names, relationships)
        return graphiti_json, nodes, edges

    build_graphiti_btn.click(
        on_build_graphiti,
        inputs=[entity_names_state, relationships_state],
        outputs=[graphiti_output, entity_nodes_state, entity_edges_state]
    )
```

**Step 3: Test Graphiti object building**

Run: `python graphiti-poc.py`

Expected:
1. Run through Stages 1-3
2. Click "Build Graphiti Objects"
3. See JSON with nodes (containing UUIDs, names, etc.) and edges (with source/target UUIDs)

**Step 4: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 4 Graphiti object building"
```

---

## Task 13: Implement Stage 5 (FalkorDB Write)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add FalkorDB write function**

Add before the Gradio interface definition:

```python
# Add this import at the top
from falkordb_utils import write_entities_and_edges

# Add this function
def write_to_falkordb(entity_nodes, entity_edges):
    """
    Stage 5: Write EntityNode and EntityEdge objects to FalkorDB.

    Returns: Write result dict (with UUIDs)
    """
    if not entity_nodes:
        return {"error": "No entities to write"}

    try:
        result = write_entities_and_edges(entity_nodes, entity_edges)
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
```

**Step 2: Wire up Stage 5 button**

Add to event handlers section:

```python
    # Stage 5: Write to FalkorDB
    def on_write_falkor(entity_nodes, entity_edges):
        result = write_to_falkordb(entity_nodes, entity_edges)
        # Update database stats
        new_stats = str(get_db_stats())
        return result, result, new_stats

    write_falkor_btn.click(
        on_write_falkor,
        inputs=[entity_nodes_state, entity_edges_state],
        outputs=[write_output, write_result_state, db_stats_display]
    )
```

**Step 3: Test FalkorDB write**

Run: `python graphiti-poc.py`

Expected:
1. Run through Stages 1-4
2. Click "Write to Falkor"
3. See write confirmation JSON with node/edge counts and UUIDs
4. Database stats should update showing new node/edge counts

**Step 4: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 5 FalkorDB write"
```

---

## Task 14: Implement Stage 6 (Graphviz Verification)

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add Graphviz rendering function**

Add before the Gradio interface definition:

```python
# Add this import at the top
from graphviz_utils import load_written_entities, render_graph_from_db

# Add this function
def render_verification_graph(write_result):
    """
    Stage 6: Verify FalkorDB write by querying and rendering graph.

    Returns: Path to PNG file, or None on error
    """
    if not write_result or "error" in write_result:
        return None

    # Load entities from DB using UUIDs
    node_uuids = write_result.get("node_uuids", [])
    edge_uuids = write_result.get("edge_uuids", [])

    if not node_uuids:
        return None

    db_data = load_written_entities(node_uuids, edge_uuids)
    return render_graph_from_db(db_data)
```

**Step 2: Wire up Stage 6 to trigger automatically after Stage 5**

Update the Stage 5 event handler:

```python
    # Stage 5: Write to FalkorDB (triggers Stage 6)
    def on_write_falkor(entity_nodes, entity_edges):
        result = write_to_falkordb(entity_nodes, entity_edges)
        # Update database stats
        new_stats = str(get_db_stats())
        # Render verification graph
        graph_img = render_verification_graph(result)
        return result, result, new_stats, graph_img

    write_falkor_btn.click(
        on_write_falkor,
        inputs=[entity_nodes_state, entity_edges_state],
        outputs=[write_output, write_result_state, db_stats_display, graphviz_output]
    )
```

**Step 3: Test end-to-end pipeline**

Run: `python graphiti-poc.py`

Expected:
1. Enter text: "Alice works at Microsoft in Seattle. She reports to Bob."
2. Wait for NER (Stage 1) to complete automatically
3. Click "Run Facts" (Stage 2)
4. Click "Run Relationships" (Stage 3)
5. Click "Build Graphiti Objects" (Stage 4)
6. Click "Write to Falkor" (Stage 5)
7. See graph visualization appear in Stage 6 showing Alice, Microsoft, Seattle, Bob with relationships

**Step 4: Verify database persistence**

Click "Reset Database" button → stats should show 0 nodes/edges

Run pipeline again → stats should increment

**Step 5: Stop server and commit**

```bash
git add graphiti-poc.py
git commit -m "feat: implement Stage 6 Graphviz verification"
```

---

## Task 15: Add Error Handling and Polish

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add error display improvements**

Update the Gradio interface to show errors more clearly:

```python
    # Update Stage 2 output to show errors
    facts_output = gr.JSON(label="Extracted Facts (or error)")

    # Update Stage 3 output
    relationships_output = gr.JSON(label="Inferred Relationships (or error)")

    # Update Stage 4 output
    graphiti_output = gr.JSON(label="EntityNode + EntityEdge Objects (or error)")

    # Update Stage 5 output
    write_output = gr.JSON(label="Write Confirmation (or error with traceback)")
```

**Step 2: Add example text**

Add an example text box above Stage 0:

```python
    # Add this after the database stats section
    gr.Markdown("### Example Text")
    example_text = gr.Textbox(
        value="Alice works at Microsoft in Seattle. She reports to Bob, who manages the engineering team.",
        interactive=False,
        show_label=False
    )

    # Add a button to load example
    load_example_btn = gr.Button("Load Example", size="sm")

    # Add event handler
    def on_load_example():
        return example_text.value

    load_example_btn.click(
        on_load_example,
        outputs=[input_text]
    )
```

**Step 3: Test error handling**

Run: `python graphiti-poc.py`

Test error cases:
1. Click "Run Facts" without entering text → should show error JSON
2. Click "Run Relationships" without running facts → should show error JSON
3. Verify errors are readable in UI

Test example loading:
1. Click "Load Example" → text should populate
2. Run through full pipeline → should work end-to-end

**Step 4: Commit**

```bash
git add graphiti-poc.py
git commit -m "feat: add error handling and example text"
```

---

## Task 16: Add Logging Configuration

**Files:**
- Modify: `graphiti-poc.py`

**Step 1: Add logging setup at module level**

Add at the top of the file after imports:

```python
# Add this import
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Starting Graphiti PoC with MODEL_CONFIG: {MODEL_CONFIG}")
logger.info(f"Database path: {DB_PATH}")
```

**Step 2: Add info logging to key stages**

Update each stage function to log execution:

```python
# In process_ner():
logger.info(f"Stage 1: Extracted {len(unique_names)} unique entities")

# In extract_facts():
logger.info(f"Stage 2: Extracted {len(facts.items)} facts")

# In infer_relationships():
logger.info(f"Stage 3: Inferred {len(relationships.items)} relationships")

# In build_graphiti_objects():
logger.info(f"Stage 4: Built {len(entity_nodes)} nodes and {len(entity_edges)} edges")

# In write_to_falkordb():
logger.info(f"Stage 5: Writing {len(entity_nodes)} nodes and {len(entity_edges)} edges to FalkorDB")
```

**Step 3: Test logging output**

Run: `python graphiti-poc.py`

Expected: Console output showing INFO logs for each stage execution

**Step 4: Commit**

```bash
git add graphiti-poc.py
git commit -m "feat: add logging configuration"
```

---

## Task 17: Final Integration Test

**Files:**
- None (testing only)

**Step 1: Reset database**

Run: `python graphiti-poc.py`

Click "Reset Database" → verify stats show 0/0

**Step 2: Run full pipeline with example**

1. Click "Load Example"
2. Verify Stage 1 NER runs automatically
3. Click through Stages 2-5
4. Verify graph visualization appears
5. Verify database stats update

**Step 3: Test "Persons only" filter**

1. Check "Persons only" checkbox
2. Verify Stage 1 output updates (only shows person names)
3. Run pipeline with filtered entities
4. Verify graph only contains person nodes

**Step 4: Test with custom input**

Enter new text with different entities and relationships. Verify pipeline works.

**Step 5: Test persistence**

1. Note current database stats
2. Run pipeline again with different text
3. Verify stats increment (database persists across runs)

**Step 6: Document test results**

If all tests pass, proceed to final commit. If issues found, fix and re-test.

---

## Task 18: Final Commit and Documentation

**Files:**
- Create: `docs/phase1-poc-status.md`

**Step 1: Create status document**

```markdown
# Phase 1 PoC Status

**Status:** ✅ Complete

**Completion Date:** 2025-11-02

## Implemented Features

- ✅ Stage 0: Text input
- ✅ Stage 1: NER entity extraction with person filter
- ✅ Stage 2: DSPy fact extraction
- ✅ Stage 3: DSPy relationship inference
- ✅ Stage 4: Graphiti object building (EntityNode + EntityEdge)
- ✅ Stage 5: FalkorDBLite write with Graphiti conventions
- ✅ Stage 6: Graphviz verification
- ✅ Database management (stats display, reset)
- ✅ Error handling and logging
- ✅ Example text loading

## Files Created

- `settings.py` - Configuration
- `models.py` - Pydantic models
- `signatures.py` - DSPy signatures
- `falkordb_utils.py` - Database initialization and write functions
- `graphviz_utils.py` - Visualization utilities
- `entity_utils.py` - Entity processing utilities
- `graphiti-poc.py` - Main Gradio application

## Testing

All manual tests passed:
- End-to-end pipeline execution
- Error handling for missing inputs
- Database persistence
- Person filter functionality
- Graph visualization rendering

## Usage

```bash
python graphiti-poc.py
```

Then:
1. Click "Load Example" or enter custom text
2. Wait for NER to complete
3. Click through stages 2-5
4. View graph visualization in stage 6

## Known Limitations

- Single context window only (no chunking)
- No embeddings (empty vectors)
- No episode management
- Serial execution (no parallelization)
- Basic error messages (no user-friendly validation)

## Next Steps (Future Phases)

- Episode chunking strategy
- Embedding generation (Qwen3-Embedding-4B)
- Multi-document ingestion
- Entity deduplication across documents
- Hybrid search implementation
```

**Step 2: Commit status document**

```bash
git add docs/phase1-poc-status.md
git commit -m "docs: add Phase 1 PoC completion status"
```

**Step 3: Final verification**

Run: `git log --oneline -20`

Verify all commits are present with clear messages.

**Step 4: Announce completion**

Report to user: "Phase 1 PoC implementation complete. All 18 tasks executed successfully."

---

## Execution Notes

**Estimated Time:** 2-3 hours for sequential execution

**Prerequisites:**
- Existing `dspy_outlines/` module working
- `distilbert_ner.py` functional
- FalkorDBLite installed and tested

**Testing Strategy:**
- Manual UI testing after each stage implementation
- Verify state propagation between stages
- Check database persistence
- Validate Graphiti convention compliance

**Common Issues:**
- MLX model loading: Ensure models in `.models/` directory
- FalkorDB connection: Verify `data/` directory writable
- NER debouncing: May take 1-2 seconds to update
- Graphviz rendering: Requires graphviz system package installed
