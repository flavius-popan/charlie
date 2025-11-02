# PoC Remediation Plan: Achieve Full Graphiti Compatibility

## Executive Summary

This plan remediates `graphiti-poc.py` to achieve 1:1 parity with `Graphiti.add_episode()`, ensuring downstream Graphiti tooling can operate unchanged on the data written by our custom pipeline.

**Critical gaps identified in `poc-analysis.md`:**
1. Missing `EpisodicNode` creation
2. Missing `EpisodicEdge` (MENTIONS) creation
3. `EntityEdge.episodes` field empty (should contain episode UUID)
4. `episodes` field JSON-serialized as string (should be native list)
5. Relationship names not normalized to SCREAMING_SNAKE_CASE

**Research findings confirm Graphiti's flow:**
- EpisodicNode → EntityNode extraction → EntityEdge extraction (with `episodes=[episode.uuid]`) → EpisodicEdge creation (MENTIONS links) → bulk save
- All edges use `:RELATES_TO` Cypher type; semantic name stored in `r.name` property
- Episode UUIDs propagate to `EntityEdge.episodes` list and enable provenance tracking

---

## Implementation Tasks

### Task 1: Create Edge Name Normalization Utility

**File:** `entity_utils.py`
**Location:** After line 96 (after `build_entity_edges()`)

**Add function:**
```python
def normalize_edge_name(name: str) -> str:
    """
    Normalize relationship name to SCREAMING_SNAKE_CASE.

    Mirrors Graphiti's prompt-based convention (extract_edges.py:26, 120).

    Examples:
        "works at" → "WORKS_AT"
        "knows" → "KNOWS"
        "WorksAt" → "WORKS_AT"
    """
    # Remove extra whitespace
    name = " ".join(name.split())
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Convert to uppercase
    return name.upper()
```

**Validation:**
- Test: `normalize_edge_name("works at")` → `"WORKS_AT"`
- Test: `normalize_edge_name("  knows  ")` → `"KNOWS"`
- Test: `normalize_edge_name("has_relationship_with")` → `"HAS_RELATIONSHIP_WITH"`

---

### Task 2: Apply Edge Name Normalization in build_entity_edges()

**File:** `entity_utils.py`
**Line:** 96

**Current code:**
```python
edge = EntityEdge(
    source_node_uuid=source_node.uuid,
    target_node_uuid=target_node.uuid,
    name=rel.relation,  # ← Raw DSPy output
    fact=rel.context,
    ...
)
```

**Modified code:**
```python
edge = EntityEdge(
    source_node_uuid=source_node.uuid,
    target_node_uuid=target_node.uuid,
    name=normalize_edge_name(rel.relation),  # ← Normalize to SCREAMING_SNAKE_CASE
    fact=rel.context,
    ...
)
```

**Validation:**
- Verify DSPy output like `"works_at"` becomes `"WORKS_AT"` in EntityEdge.name
- Check Gradio UI Stage 4 output shows normalized names

---

### Task 3: Populate EntityEdge.episodes Field

**File:** `entity_utils.py`
**Function:** `build_entity_edges()` (line 80)

**Add parameter:**
```python
def build_entity_edges(
    relationships,
    entity_map: dict,
    episode_uuid: str  # ← Add this parameter
) -> list[EntityEdge]:
```

**Modify edge creation (line 96):**
```python
edge = EntityEdge(
    source_node_uuid=source_node.uuid,
    target_node_uuid=target_node.uuid,
    name=normalize_edge_name(rel.relation),
    fact=rel.context,
    group_id="phase1-poc",
    created_at=utc_now(),
    fact_embedding=[],
    episodes=[episode_uuid],  # ← Add episode UUID to list
    expired_at=None,
    valid_at=None,
    invalid_at=None,
    attributes={}
)
```

**Update caller in graphiti-poc.py (line 148):**
```python
# Stage 4: Build Graphiti Objects
def on_build_graphiti(entity_names, relationships, episode_uuid):
    nodes, edges, graphiti_json = build_graphiti_objects(
        entity_names,
        relationships,
        episode_uuid  # ← Pass episode UUID
    )
    return graphiti_json, nodes, edges
```

**Validation:**
- Check EntityEdge.episodes contains exactly one UUID (the episode)
- Verify Gradio Stage 4 JSON output shows `"episodes": ["<uuid>"]`

---

### Task 4: Create Episode Creation Utility

**File:** `entity_utils.py`
**Location:** After `build_entity_edges()`

**Add imports at top:**
```python
from graphiti_core.nodes import EpisodicNode
from graphiti_core.edges import EpisodicEdge
from graphiti_core.utils.datetime_utils import utc_now
from datetime import datetime
from uuid import uuid4
```

**Add function:**
```python
def build_episodic_node(
    content: str,
    reference_time: datetime,
    group_id: str = "phase1-poc"
) -> EpisodicNode:
    """
    Create EpisodicNode for a journal entry.

    Mirrors graphiti.py:706-720 (EpisodicNode creation).

    Args:
        content: Raw journal entry text
        reference_time: Timestamp when entry was created
        group_id: Graph partition identifier

    Returns:
        EpisodicNode with entity_edges initially empty (populated after edge creation)
    """
    from graphiti_core.nodes import EpisodeType

    episode = EpisodicNode(
        name=f"Journal Entry {reference_time.isoformat()}",
        group_id=group_id,
        source=EpisodeType.text,  # Journal entries are text type
        source_description="Daily journal entry",
        content=content,
        valid_at=reference_time,
        created_at=utc_now(),
        labels=[],
        entity_edges=[]  # Will be populated after EntityEdge creation
    )
    return episode


def build_episodic_edges(
    episode: EpisodicNode,
    entity_nodes: list
) -> list[EpisodicEdge]:
    """
    Create MENTIONS edges from episode to each entity.

    Mirrors edge_operations.py:51-68 (build_episodic_edges).

    Args:
        episode: The EpisodicNode
        entity_nodes: List of EntityNode objects extracted from episode

    Returns:
        List of EpisodicEdge objects (one per entity)
    """
    episodic_edges = []

    for node in entity_nodes:
        edge = EpisodicEdge(
            source_node_uuid=episode.uuid,  # Episode mentions entity
            target_node_uuid=node.uuid,
            group_id=episode.group_id,
            created_at=utc_now()
        )
        episodic_edges.append(edge)

    return episodic_edges
```

**Validation:**
- Test with sample journal entry → should create EpisodicNode
- Verify `source=EpisodeType.text` and content populated
- Test MENTIONS edges → one per entity, correct UUIDs

---

### Task 5: Integrate Episode Creation into Pipeline

**File:** `graphiti-poc.py`
**Function:** `build_graphiti_objects()` (line 134)

**Modify function signature:**
```python
def build_graphiti_objects(
    input_text: str,  # ← Add input text
    entity_names: list[str],
    relationships,
    reference_time: datetime  # ← Add reference time
):
```

**Modify function body:**
```python
def build_graphiti_objects(input_text, entity_names, relationships, reference_time):
    """
    Stage 4: Build EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects.

    Now mirrors graphiti.py:413-436 (_process_episode_data).
    """
    if not entity_names or not relationships:
        return None, [], [], [], {
            "error": "Need entities and relationships"
        }

    try:
        from entity_utils import (
            build_entity_nodes,
            build_entity_edges,
            build_episodic_node,
            build_episodic_edges
        )

        # 1. Create EpisodicNode (mirrors graphiti.py:706-720)
        episode = build_episodic_node(
            content=input_text,
            reference_time=reference_time
        )

        # 2. Build EntityNodes
        entity_nodes, entity_map = build_entity_nodes(entity_names)

        # 3. Build EntityEdges with episode UUID (mirrors edge_operations.py:220-230)
        entity_edges = build_entity_edges(
            relationships,
            entity_map,
            episode_uuid=episode.uuid  # ← Pass episode UUID
        )

        # 4. Link episode to entity edges (mirrors graphiti.py:422)
        episode.entity_edges = [edge.uuid for edge in entity_edges]

        # 5. Build EpisodicEdges (MENTIONS) (mirrors edge_operations.py:51-68)
        episodic_edges = build_episodic_edges(episode, entity_nodes)

        logger.info(
            f"Stage 4: Built episode {episode.uuid}, "
            f"{len(entity_nodes)} nodes, "
            f"{len(entity_edges)} entity edges, "
            f"{len(episodic_edges)} episodic edges"
        )

        # Convert to JSON for display
        graphiti_json = {
            "episode": episode.model_dump(),
            "nodes": [n.model_dump() for n in entity_nodes],
            "entity_edges": [e.model_dump() for e in entity_edges],
            "episodic_edges": [e.model_dump() for e in episodic_edges]
        }

        return episode, entity_nodes, entity_edges, episodic_edges, graphiti_json

    except Exception as e:
        return None, [], [], [], {"error": str(e)}
```

**Update Gradio state management (line 269):**
```python
# Add new state variables
episode_state = gr.State(None)             # EpisodicNode
episodic_edges_state = gr.State([])        # EpisodicEdge list
```

**Update event handler (line 318):**
```python
def on_build_graphiti(text, entity_names, relationships):
    from datetime import datetime
    reference_time = datetime.now()

    episode, nodes, entity_edges, episodic_edges, json_output = build_graphiti_objects(
        input_text=text,
        entity_names=entity_names,
        relationships=relationships,
        reference_time=reference_time
    )
    return json_output, episode, nodes, entity_edges, episodic_edges

build_graphiti_btn.click(
    on_build_graphiti,
    inputs=[input_text, entity_names_state, relationships_state],
    outputs=[
        graphiti_output,
        episode_state,           # ← New
        entity_nodes_state,
        entity_edges_state,
        episodic_edges_state     # ← New
    ]
)
```

**Validation:**
- Stage 4 JSON output includes all four components (episode, nodes, entity_edges, episodic_edges)
- Verify episode.entity_edges contains all EntityEdge UUIDs
- Verify one EpisodicEdge per EntityNode

---

### Task 6: Fix FalkorDB Write - Remove JSON Serialization

**File:** `falkordb_utils.py`
**Line:** 246

**Current code (INCORRECT):**
```python
"episodes": json.dumps(edge['episodes'])  # ← Serializes list to JSON string
```

**Fixed code:**
```python
"episodes": edge['episodes']  # ← Keep as native list
```

**Why:** FalkorDB supports native lists (confirmed by research). Graphiti stores `episodes` as a list property, and Neptune is the only backend that needs string conversion (which happens in Neptune's specific query generation).

**Validation:**
- After write, query: `MATCH ()-[r:RELATES_TO]->() RETURN r.episodes LIMIT 1`
- Verify result is a list, not a string like `"[\"uuid1\"]"`

---

### Task 7: Write Episodes and MENTIONS Edges to FalkorDB

**File:** `falkordb_utils.py`
**Function:** `write_entities_and_edges()` (line 12)

**Modify function signature:**
```python
def write_entities_and_edges(
    episode,           # ← Add EpisodicNode
    entity_nodes,
    entity_edges,
    episodic_edges     # ← Add EpisodicEdge list
):
```

**Add episodic node write (after line 65, before entity node write):**
```python
# 1. Write EpisodicNode (mirrors bulk_utils.py:217)
episode_dict = {
    "uuid": episode.uuid,
    "name": episode.name,
    "group_id": episode.group_id,
    "source": episode.source.value,  # Convert enum to string
    "source_description": episode.source_description,
    "content": episode.content,
    "valid_at": episode.valid_at,
    "created_at": episode.created_at,
    "entity_edges": episode.entity_edges,
    "labels": episode.labels or [],
}
episode_dict = convert_datetimes_to_strings(episode_dict)

episode_query = """
MERGE (e:Episodic {uuid: $episode.uuid})
SET e:Episodic
SET e.name = $episode.name,
    e.group_id = $episode.group_id,
    e.source = $episode.source,
    e.source_description = $episode.source_description,
    e.content = $episode.content,
    e.valid_at = $episode.valid_at,
    e.created_at = $episode.created_at,
    e.entity_edges = $episode.entity_edges,
    e.labels = $episode.labels
RETURN e.uuid AS uuid
"""
result = graph.query(episode_query, {"episode": episode_dict})
logger.info(f"Created episode node: {episode.uuid}")
```

**Add episodic edges write (after entity edges write, around line 120):**
```python
# 4. Write EpisodicEdges (MENTIONS) (mirrors bulk_utils.py:221)
episodic_edge_dicts = []
for edge in episodic_edges:
    edge_dict = {
        "uuid": edge.uuid,
        "source_uuid": edge.source_node_uuid,  # Episode UUID
        "target_uuid": edge.target_node_uuid,  # Entity UUID
        "group_id": edge.group_id,
        "created_at": edge.created_at,
    }
    episodic_edge_dicts.append(convert_datetimes_to_strings(edge_dict))

if episodic_edge_dicts:
    episodic_edge_query = """
    UNWIND $edges AS edge
    MATCH (episode:Episodic {uuid: edge.source_uuid})
    MATCH (entity:Entity {uuid: edge.target_uuid})
    MERGE (episode)-[r:MENTIONS {uuid: edge.uuid}]->(entity)
    SET r.group_id = edge.group_id,
        r.created_at = edge.created_at
    RETURN r.uuid AS uuid
    """
    result = graph.query(episodic_edge_query, {"edges": episodic_edge_dicts})
    logger.info(f"Created {len(result.result_set)} MENTIONS edges")
```

**Update return statement:**
```python
return {
    "episode_uuid": episode.uuid,
    "node_uuids": [n["uuid"] for n in node_dicts],
    "edge_uuids": [e["uuid"] for e in edge_dicts],
    "episodic_edge_uuids": [e["uuid"] for e in episodic_edge_dicts],
    "nodes_created": len(node_dicts),
    "edges_created": len(edge_dicts),
    "episodic_edges_created": len(episodic_edge_dicts)
}
```

**Update caller in graphiti-poc.py (line 329):**
```python
def on_write_falkor(episode, entity_nodes, entity_edges, episodic_edges):
    result = write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges)
    # ... rest of handler

write_falkor_btn.click(
    on_write_falkor,
    inputs=[
        episode_state,           # ← Add
        entity_nodes_state,
        entity_edges_state,
        episodic_edges_state     # ← Add
    ],
    outputs=[write_output, write_result_state, db_stats_display, graphviz_output]
)
```

**Validation:**
- Query: `MATCH (e:Episodic) RETURN count(e)` → should return 1 after write
- Query: `MATCH ()-[r:MENTIONS]->() RETURN count(r)` → should return count of entities
- Verify `episode.entity_edges` list populated in database

---

### Task 8: Update Graphviz Verification to Include Episodes

**File:** `graphviz_utils.py`
**Function:** `load_written_entities()` (line 5)

**Add episode query:**
```python
def load_written_entities(node_uuids: list[str], edge_uuids: list[str], episode_uuid: str = None):
    """Query FalkorDBLite for episode, nodes, edges, and MENTIONS edges by UUID."""
    try:
        # 1. Load episode (if provided)
        episode_data = None
        if episode_uuid:
            episode_query = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content
            """
            episode_result = graph.query(episode_query, {"uuid": episode_uuid})
            episode_data = episode_result.result_set[0] if episode_result.result_set else None

        # 2. Load entities (existing code)
        node_query = """
        UNWIND $uuids AS uuid
        MATCH (n:Entity {uuid: uuid})
        RETURN n.uuid AS uuid, n.name AS name
        """
        node_result = graph.query(node_query, {"uuids": node_uuids})

        # 3. Load entity edges (existing code)
        edge_query = """
        UNWIND $uuids AS uuid
        MATCH (source:Entity)-[r:RELATES_TO {uuid: uuid}]->(target:Entity)
        RETURN r.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid,
               r.name AS name, r.episodes AS episodes
        """
        edge_result = graph.query(edge_query, {"uuids": edge_uuids})

        # 4. Load MENTIONS edges (new)
        mentions_query = """
        MATCH (episode:Episodic {uuid: $episode_uuid})-[r:MENTIONS]->(entity:Entity)
        RETURN r.uuid AS uuid, episode.uuid AS source_uuid, entity.uuid AS target_uuid
        """ if episode_uuid else "RETURN null AS uuid LIMIT 0"
        mentions_result = graph.query(mentions_query, {"episode_uuid": episode_uuid} if episode_uuid else {})

        return {
            "episode": episode_data,
            "nodes": node_result.result_set or [],
            "edges": edge_result.result_set or [],
            "mentions_edges": mentions_result.result_set or []
        }
    except Exception as exc:
        logging.exception("Failed to verify FalkorDBLite write")
        return {"error": str(exc)}
```

**Update render function (line 25):**
```python
def render_graph_from_db(db_data: dict[str, Any]) -> str | None:
    """Render Graphviz graph including episode node."""

    if "error" in db_data:
        logging.error(f"Cannot render graph: {db_data['error']}")
        return None

    try:
        dot = Digraph(format="png")

        # ... existing styling code ...

        # Add episode node (if present)
        if db_data.get("episode"):
            uuid, name, content = db_data["episode"]
            # Truncate content for display
            label = f"{name}\\n{content[:50]}..." if len(content) > 50 else f"{name}\\n{content}"
            dot.node(
                uuid,
                label=label,
                fillcolor="#fef3c7",  # Light yellow for episodes
                shape="box",
                fontcolor="#1f2937",
                tooltip=content
            )

        # Add entity nodes (existing code)
        for row in db_data["nodes"]:
            uuid, name = row
            # ... existing node code ...

        # Add MENTIONS edges (episode → entities)
        for row in db_data.get("mentions_edges", []):
            edge_uuid, source_uuid, target_uuid = row
            dot.edge(
                source_uuid,
                target_uuid,
                label="MENTIONS",
                color="#a78bfa",  # Purple for episode links
                fontcolor="#e2e8f0",
                style="dashed",
                tooltip="Episode mentions entity"
            )

        # Add entity edges (existing code)
        for row in db_data["edges"]:
            edge_uuid, source_uuid, target_uuid, edge_name, episodes = row
            edge_label = edge_name.replace("_", " ")

            # Show episode provenance in tooltip
            tooltip = f"{edge_label} (from {len(episodes)} episode(s))"

            dot.edge(
                source_uuid,
                target_uuid,
                label=edge_label,
                color="#60a5fa",
                fontcolor="#e2e8f0",
                tooltip=tooltip
            )

        # ... rest of rendering code ...
```

**Update caller in graphiti-poc.py (line 186):**
```python
def render_verification_graph(write_result):
    """Stage 6: Verify using episode UUID."""
    if not write_result or "error" in write_result:
        return None

    episode_uuid = write_result.get("episode_uuid")
    node_uuids = write_result.get("node_uuids", [])
    edge_uuids = write_result.get("edge_uuids", [])

    db_data = load_written_entities(node_uuids, edge_uuids, episode_uuid)  # ← Pass episode UUID
    return render_graph_from_db(db_data)
```

**Validation:**
- Graphviz output shows episode node (light yellow box)
- Dashed purple arrows from episode to entities (MENTIONS)
- Solid blue arrows between entities (RELATES_TO)
- Hover tooltips show episode provenance

---

### Task 9: Add Graphiti Compatibility Validation

**File:** Create new file `validation.py`

**Content:**
```python
"""Validation that PoC data is compatible with Graphiti query layer."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_graphiti_compatibility(write_result: dict[str, Any], db) -> dict[str, Any]:
    """
    Validate written data is queryable using Graphiti-style patterns.

    Tests:
    1. Episode node exists with correct structure
    2. MENTIONS edges link episode to entities
    3. EntityEdge.episodes field is a list (not string)
    4. Entity edges use :RELATES_TO type with semantic name in .name property

    Args:
        write_result: Result from write_entities_and_edges()
        db: FalkorDB client (for graph.query access)

    Returns:
        Dict with validation results
    """
    results = {
        "episode_exists": False,
        "mentions_edges_correct": False,
        "episodes_field_is_list": False,
        "edge_naming_correct": False,
        "all_valid": False,
        "errors": []
    }

    try:
        graph = db.select_graph("phase1_poc")
        episode_uuid = write_result.get("episode_uuid")

        # Test 1: Episode exists with entity_edges field
        episode_query = """
        MATCH (e:Episodic {uuid: $uuid})
        RETURN e.uuid, e.entity_edges
        """
        result = graph.query(episode_query, {"uuid": episode_uuid})
        if result.result_set:
            uuid, entity_edges = result.result_set[0]
            results["episode_exists"] = True
            logger.info(f"✓ Episode node exists with {len(entity_edges)} entity_edges")
        else:
            results["errors"].append("Episode node not found in database")

        # Test 2: MENTIONS edges exist
        mentions_query = """
        MATCH (e:Episodic {uuid: $uuid})-[:MENTIONS]->(entity:Entity)
        RETURN count(entity) AS count
        """
        result = graph.query(mentions_query, {"uuid": episode_uuid})
        mentions_count = result.result_set[0][0] if result.result_set else 0
        expected_count = len(write_result.get("node_uuids", []))

        if mentions_count == expected_count:
            results["mentions_edges_correct"] = True
            logger.info(f"✓ MENTIONS edges correct: {mentions_count} edges")
        else:
            results["errors"].append(
                f"MENTIONS edges mismatch: expected {expected_count}, got {mentions_count}"
            )

        # Test 3: episodes field is list, not string
        edge_episodes_query = """
        MATCH ()-[r:RELATES_TO]->()
        RETURN r.episodes LIMIT 1
        """
        result = graph.query(edge_episodes_query, {})
        if result.result_set:
            episodes_value = result.result_set[0][0]
            if isinstance(episodes_value, list):
                results["episodes_field_is_list"] = True
                logger.info(f"✓ episodes field is list: {episodes_value}")
            else:
                results["errors"].append(
                    f"episodes field is {type(episodes_value).__name__}, not list: {episodes_value}"
                )

        # Test 4: Edge naming (uses :RELATES_TO with semantic name in .name)
        edge_naming_query = """
        MATCH ()-[r:RELATES_TO]->()
        RETURN type(r) AS edge_type, r.name AS semantic_name LIMIT 1
        """
        result = graph.query(edge_naming_query, {})
        if result.result_set:
            edge_type, semantic_name = result.result_set[0]
            if edge_type == "RELATES_TO" and semantic_name:
                results["edge_naming_correct"] = True
                logger.info(f"✓ Edge naming correct: type={edge_type}, name={semantic_name}")
            else:
                results["errors"].append(
                    f"Edge naming incorrect: type={edge_type}, name={semantic_name}"
                )

        # Overall validation
        results["all_valid"] = all([
            results["episode_exists"],
            results["mentions_edges_correct"],
            results["episodes_field_is_list"],
            results["edge_naming_correct"]
        ])

        if results["all_valid"]:
            logger.info("✓ ALL VALIDATIONS PASSED - Full Graphiti compatibility achieved")
        else:
            logger.warning(f"⚠ Validation failed: {results['errors']}")

    except Exception as exc:
        logger.exception("Validation error")
        results["errors"].append(str(exc))

    return results
```

**Integrate into graphiti-poc.py (after Stage 5 write, line 335):**
```python
def on_write_falkor(episode, entity_nodes, entity_edges, episodic_edges):
    result = write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges)

    # Validate Graphiti compatibility
    from validation import validate_graphiti_compatibility
    validation_result = validate_graphiti_compatibility(result, db)

    # Add validation to result for UI display
    result["validation"] = validation_result

    # Update database stats
    new_stats = str(get_db_stats())

    # Render verification graph
    graph_img = render_verification_graph(result)

    return result, result, new_stats, graph_img
```

**Update Gradio UI to show validation (line 260):**
```python
# After write_output JSON display, add:
validation_output = gr.JSON(label="Graphiti Compatibility Validation")

# Update write handler outputs:
write_falkor_btn.click(
    on_write_falkor,
    inputs=[episode_state, entity_nodes_state, entity_edges_state, episodic_edges_state],
    outputs=[
        write_output,
        write_result_state,
        db_stats_display,
        graphviz_output,
        validation_output  # ← New
    ]
)
```

**Validation:**
- All 4 validation checks pass (green checkmarks in logs)
- UI shows `"all_valid": true` in validation JSON
- No errors in validation results

---

## Testing Checklist

After implementing all tasks:

### Unit Tests
- [ ] `normalize_edge_name()` converts to SCREAMING_SNAKE_CASE
- [ ] `build_episodic_node()` creates valid EpisodicNode
- [ ] `build_episodic_edges()` creates one MENTIONS edge per entity
- [ ] `EntityEdge.episodes` contains episode UUID after build

### Integration Tests
- [ ] Run full pipeline with example text
- [ ] Verify Stage 4 output includes episode, nodes, entity_edges, episodic_edges
- [ ] Verify Stage 5 writes all components to FalkorDB
- [ ] Verify Graphviz shows episode node + MENTIONS edges

### Validation Tests
- [ ] All 4 Graphiti compatibility checks pass
- [ ] Query: `MATCH (e:Episodic) RETURN e.entity_edges` → returns list of edge UUIDs
- [ ] Query: `MATCH ()-[r:RELATES_TO]->() RETURN r.episodes` → returns list (not string)
- [ ] Query: `MATCH ()-[r:MENTIONS]->() RETURN count(r)` → matches entity count

### Graphiti Interoperability Tests
- [ ] Use Graphiti's `EntityEdge.get_by_uuid()` to load edge from FalkorDB
- [ ] Verify loaded edge has `episodes` as list
- [ ] Use Graphiti's search helpers on written data (if available for FalkorDB)

---

## Success Criteria

1. **Episode provenance complete**: Every EntityEdge links back to originating episode via `episodes` field
2. **MENTIONS graph exists**: Episode nodes connected to extracted entities
3. **Naming conventions match**: Edge names in SCREAMING_SNAKE_CASE, :RELATES_TO type universal
4. **Field types correct**: `episodes` stored as list, not JSON string
5. **Validation passes**: All 4 compatibility checks return `True`
6. **Graphviz visualizes correctly**: Shows episode → entities → relationships

---

## Rollout Strategy

**Recommended order:**
1. Implement Tasks 1-3 (normalization) → verify in isolation
2. Implement Tasks 4-5 (episode creation) → test with mock data
3. Implement Task 6 (fix serialization) → verify database queries
4. Implement Tasks 7-8 (write pipeline + viz) → end-to-end test
5. Implement Task 9 (validation) → verify compatibility

**Risk mitigation:**
- Each task includes validation steps
- Test with "Reset Database" between iterations
- Keep original `graphiti-poc.py` as `graphiti-poc-v1.py` backup before changes

---

## Expected Outcome

After remediation:
- PoC writes data identical to `Graphiti.add_episode()`
- Downstream Graphiti tools (search, query, analytics) work unchanged on PoC-generated graphs
- Phase 1 goal achieved: **custom pipeline with full Graphiti support**
