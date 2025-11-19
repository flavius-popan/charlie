# Extract Nodes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement V2 backend entity extraction operation that identifies entities from journal entries and resolves them against existing graph entities using MinHash LSH deduplication.

**Architecture:** Two-layer pattern in single file (`backend/graph/extract_nodes.py`): (1) `EntityExtractor(dspy.Module)` - pure LLM extraction, (2) `extract_nodes()` async function - orchestrates DB I/O, deduplication, persistence.

**Tech Stack:** DSPy, graphiti-core deduplication helpers, FalkorDB, asyncio.

---

## Overview

The implementation follows V2 architecture goals:
- Operations work on existing episodes (no episode creation)
- Discrete units with no data passing between stages
- Reuses graphiti-core utilities for deduplication
- Compatible with future Huey task integration
- DSPy module importable by optimizer scripts

## Component Structure

### Files to Create
- `backend/graph/__init__.py` - Module init
- `backend/graph/entities_edges.py` - Entity type definitions (Person, Place, Organization, Activity)
- `backend/graph/extract_nodes.py` - Main implementation

### Files to Modify
- `backend/database/persistence.py` - Add entity/edge persistence
- `backend/__init__.py` - Export new functions
- `tests/test_backend/test_database.py` - Add tests

---

## Design Decision: SELF Entity Handling

**Key architectural decision:** SELF entity is NOT extracted during extract_nodes operation.

**Rationale:**
- SELF pre-seeded in database on journal creation (already implemented in backend)
- MENTIONS edges to SELF would be redundant with `episode.group_id`
- Most journal entries use first-person pronouns → thousands of MENTIONS edges to single node
- SELF's value is in ENTITY edges (relationships): Self → SPENT_TIME_WITH → Sarah
- Those edges created during extract_edges operation (future), not extract_nodes

**Implementation approach:**
- extract_nodes: Skip Self extraction entirely, only extract other entities (Person, Place, etc.)
- extract_edges: Include Self in relationship extraction when building entity-to-entity edges
- MENTIONS edges: Only created for non-Self entities

**This means:**
- No `self_entity.py` module needed
- No pronoun detection in extract_nodes
- Simpler extraction logic (one less special case)
- SELF handling deferred to extract_edges operation

---

## Task 1: Entity Persistence Layer

**Goal:** Add functions to persist entities and edges to FalkorDB.

**Test (add to `tests/test_backend/test_database.py`):**

```python
@pytest.mark.asyncio
async def test_persist_entities_and_edges(isolated_graph):
    """Test persisting entities and episodic edges."""
    from backend.database.persistence import persist_entities_and_edges
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EpisodicEdge
    from graphiti_core.utils.datetime_utils import utc_now
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL

    episode_uuid = await add_journal_entry("Test content")

    person_node = EntityNode(
        name="Sarah",
        group_id=DEFAULT_JOURNAL,
        labels=["Entity", "Person"],
        summary="",
        created_at=utc_now(),
        name_embedding=[],
    )

    episodic_edge = EpisodicEdge(
        source_node_uuid=episode_uuid,
        target_node_uuid=person_node.uuid,
        created_at=utc_now(),
        fact_embedding=[],
    )

    await persist_entities_and_edges(
        nodes=[person_node],
        edges=[],
        episodic_edges=[episodic_edge],
        journal=DEFAULT_JOURNAL,
    )

    # Verify entity exists
    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity {name: 'Sarah'}) RETURN count(e)"
    result = driver.execute_query(query)
    assert result[0][0] >= 1
```

**Implementation (add to `backend/database/persistence.py`):**

Signature:
```python
async def persist_entities_and_edges(
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    episodic_edges: list[EpisodicEdge],
    journal: str,
) -> None:
    """Persist entities and relationships using graphiti-core bulk writer.

    Ensures embeddings are set, creates NullEmbedder, calls add_nodes_and_edges_bulk.
    Thread-safe via per-journal locking.
    """
```

Also add:
```python
async def update_episode_attributes(
    episode_uuid: str,
    attributes: dict,
    journal: str,
) -> None:
    """Update episode.attributes field for temporary state storage (uuid_map).

    Uses EpisodicNode.get_by_uuid() + .save() pattern from update_episode().
    """
```

Update `backend/__init__.py` exports.

---

## Task 2: Entity Type Definitions

**Goal:** Define entity types for journaling context.

**Test (add to `tests/test_backend/test_database.py`):**

```python
def test_entity_types_format():
    """Test entity types formatting for LLM."""
    from backend.graph.entities_edges import format_entity_types_for_llm, entity_types
    import json

    result = format_entity_types_for_llm(entity_types)
    types_list = json.loads(result)

    assert len(types_list) == 5  # Base + 4 custom (no Self)
    assert types_list[0]["entity_type_name"] == "Entity"

    person_type = next(t for t in types_list if t["entity_type_name"] == "Person")
    assert person_type["entity_type_id"] == 1
```

**Implementation (`backend/graph/entities_edges.py`):**

```python
"""Entity and edge type definitions for V2 backend.

No attributes per V2_PLAN.md line 41. Entity types are simple Pydantic models
for LLM classification only.

SELF entity handling: SELF is NOT extracted during extract_nodes. It's pre-seeded
in the database and only used during extract_edges for relationship extraction.
"""

from pydantic import BaseModel

class Person(BaseModel):
    """A person mentioned in journal entries."""
    pass

class Place(BaseModel):
    """A location or venue."""
    pass

class Organization(BaseModel):
    """A company, team, or community group."""
    pass

class Activity(BaseModel):
    """An event, outing, or recurring routine."""
    pass

entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Activity": Activity,
}

def format_entity_types_for_llm(types: dict | None = None) -> str:
    """Convert type definitions to JSON with enhanced descriptions for journaling."""
    # Returns JSON array: [{"entity_type_id": 0, "entity_type_name": "Entity", ...}, ...]

def get_type_name_from_id(type_id: int, types: dict | None = None) -> str:
    """Map entity_type_id back to type name."""
```

---

## Task 3: DSPy Entity Extractor

**Goal:** Pure LLM extraction module with no dependencies.

**Test (add to `tests/test_backend/test_database.py`):**

```python
def test_entity_extractor_pydantic_models():
    """Test Pydantic models for DSPy."""
    from backend.graph.extract_nodes import ExtractedEntity, ExtractedEntities

    entity = ExtractedEntity(name="Sarah", entity_type_id=1)
    assert entity.name == "Sarah"

    # Test list coercion
    entities = ExtractedEntities([{"name": "Sarah", "entity_type_id": 1}])
    assert len(entities.extracted_entities) == 1
```

**Implementation (`backend/graph/extract_nodes.py`):**

```python
"""Entity extraction for V2 backend."""

import dspy
from pydantic import BaseModel, Field, model_validator

# Pydantic models
class ExtractedEntity(BaseModel):
    """Entity with name and type."""
    name: str
    entity_type_id: int

class ExtractedEntities(BaseModel):
    """Collection with list coercion validator."""
    extracted_entities: list[ExtractedEntity]

# DSPy signature
class EntityExtractionSignature(dspy.Signature):
    """Extract entities from journal entry."""
    episode_content: str = dspy.InputField(...)
    entity_types: str = dspy.InputField(...)
    extracted_entities: ExtractedEntities = dspy.OutputField(...)

# DSPy Module
class EntityExtractor(dspy.Module):
    """Pure LLM extraction - zero dependencies on runtime infrastructure.

    This module has NO dependencies on Model Manager or queue infrastructure.
    The LLM is configured via dspy.context() by the caller.

    In production (via Huey tasks):
        - extract_nodes() calls get_model('llm') from Model Manager
        - Wraps EntityExtractor instantiation in dspy.context(lm=...)

    In optimizer scripts:
        - Import DspyLM directly from backend.inference
        - Configure via dspy.context(lm=DspyLM())

    Optimizable with DSPy teleprompters.
    Auto-loads prompts from prompts/extract_nodes.json if exists.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)
        # Auto-load optimized prompts

    def forward(self, episode_content: str, entity_types: str) -> ExtractedEntities:
        """Extract entities from text."""
```

---

## Task 4: Extract Nodes Orchestrator

**Goal:** Main operation that coordinates extraction pipeline.

**Test (add to `tests/test_backend/test_database.py`):**

```python
@pytest.mark.asyncio
async def test_extract_nodes_basic(isolated_graph):
    """Test end-to-end entity extraction."""
    import dspy
    if dspy.settings.lm is None:
        pytest.skip("No LLM configured")

    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    content = "Today I met Sarah at Central Park."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result.episode_uuid == episode_uuid
    assert result.extracted_count > 0
    # Note: Self NOT extracted, so count should be ~2 (Sarah, Central Park)

    # Verify entities in database (no Self entity from extraction)
    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE e.name <> 'Self' RETURN count(e)"
    count = driver.execute_query(query)[0][0]
    assert count >= 2
```

**Implementation (add to `backend/graph/extract_nodes.py`):**

```python
import dspy
from dataclasses import dataclass

@dataclass
class ExtractNodesResult:
    """Extraction metadata."""
    episode_uuid: str
    extracted_count: int
    resolved_count: int
    new_entities: int
    exact_matches: int
    fuzzy_matches: int
    entity_uuids: list[str]

async def extract_nodes(
    episode_uuid: str,
    journal: str,
    entity_types: dict | None = None,
    dedupe_enabled: bool = True,
) -> ExtractNodesResult:
    """Extract and resolve entities from existing episode.

    Pipeline:
    1. Fetch episode from database
    2. Get LLM from Model Manager (production) or context (optimizer)
    3. Extract entities via EntityExtractor with dspy.context()
    4. Fetch existing entities for deduplication
    5. Resolve duplicates (MinHash LSH + exact matching)
    6. Create MENTIONS edges (episode → entities, excluding Self)
    7. Persist entities and edges (using NullEmbedder - no embedding generation)
    8. Store uuid_map in episode.attributes
    9. Return metadata

    Model Access Pattern (production):
        from backend.inference.manager import get_model
        llm = get_model('llm')  # Returns DspyLM instance
        with dspy.context(lm=llm):
            extractor = EntityExtractor()
            result = extractor.forward(...)

    Model Access Pattern (optimizer scripts):
        from backend.inference import DspyLM
        llm = DspyLM()
        with dspy.context(lm=llm):
            extractor = EntityExtractor()
            # Optimize with teleprompters

    Note: SELF entity is NOT extracted. It's pre-seeded in the database
    and only used during extract_edges operation for relationship extraction.

    Embeddings: Uses NullEmbedder - no embedding generation in this operation.
    """
    # Implementation uses helpers:
    # - _fetch_existing_entities()
    # - _resolve_entities()
    # - _resolve_exact_names()
```

Helper functions:
```python
async def _fetch_existing_entities(driver, group_id: str) -> dict[str, EntityNode]:
    """Fetch all entities using EntityNode.get_by_group_ids()."""

async def _resolve_entities(
    provisional_nodes: list[EntityNode],
    existing_nodes: dict[str, EntityNode],
    driver,
    dedupe_enabled: bool,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Resolve using graphiti-core dedup helpers."""

def _resolve_exact_names(
    provisional_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
) -> None:
    """Case-insensitive exact matching (runs after fuzzy)."""
```

---

## Task 5: API Exports

**Goal:** Export from backend module.

Update `backend/graph/__init__.py`:
```python
from backend.graph.extract_nodes import extract_nodes, ExtractNodesResult

__all__ = ["extract_nodes", "ExtractNodesResult"]
```

Update `backend/__init__.py`:
```python
from backend.graph import extract_nodes, ExtractNodesResult
# Add to __all__
```

Test:
```python
def test_extract_nodes_exported():
    """Test extract_nodes is importable from backend."""
    from backend import extract_nodes
    assert callable(extract_nodes)
```

---

## Task 6: Integration Test

**Goal:** Verify complete workflow including deduplication.

**Test (add to `tests/test_backend/test_database.py`):**

```python
@pytest.mark.asyncio
async def test_extract_nodes_deduplication(isolated_graph):
    """Test entity deduplication across multiple entries."""
    import dspy
    if dspy.settings.lm is None:
        pytest.skip("No LLM configured")

    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    # First entry
    uuid1 = await add_journal_entry("I met with Sarah today.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    # Second entry mentioning Sarah again
    uuid2 = await add_journal_entry("Sarah and I had coffee.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    # Should have deduplicated Sarah
    assert result2.exact_matches > 0 or result2.fuzzy_matches > 0

    # Verify only one Sarah node exists
    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Person) WHERE e.name CONTAINS 'Sarah' RETURN count(e)"
    count = driver.execute_query(query)[0][0]
    assert count == 1

    # Verify uuid_map stored
    from backend.database import get_episode
    episode = await get_episode(uuid2, DEFAULT_JOURNAL)
    assert "uuid_map" in episode.attributes
```

---

## Verification

After implementation:

1. **Run tests:**
   ```bash
   pytest tests/test_backend/test_database.py -v -k extract_nodes
   ```

2. **Check imports:**
   ```python
   from backend import extract_nodes, ExtractNodesResult
   from backend.graph.extract_nodes import EntityExtractor
   ```

3. **Manual test (if LLM configured):**
   ```python
   import asyncio
   from backend import add_journal_entry, extract_nodes

   async def test():
       uuid = await add_journal_entry("Today I met Sarah at Central Park.")
       result = await extract_nodes(uuid, "default")
       print(f"Extracted: {result.extracted_count}, New: {result.new_entities}")
       # Should extract Sarah and Central Park, but NOT Self

   asyncio.run(test())
   ```

---

## Success Criteria

- [ ] All tests passing (or skipped if no LLM)
- [ ] `extract_nodes()` exported from backend
- [ ] Entity/edge persistence working
- [ ] Deduplication working (MinHash LSH + exact)
- [ ] SELF entity NOT extracted (deferred to extract_edges)
- [ ] uuid_map stored in episode.attributes
- [ ] EntityExtractor importable for optimizer scripts
- [ ] Compatible with future Huey integration

---

## Notes

**SELF Entity Architecture:**
- Pre-seeded in database on journal creation (already implemented in backend)
- NOT extracted during extract_nodes operation
- Will be used during extract_edges operation for relationship extraction
- MENTIONS edges only created for non-Self entities (avoids thousands of redundant edges)
- Episode ownership implicit via `episode.group_id` field

**Huey Integration (future):**
```python
# backend/services/tasks.py
@huey.task()
async def extract_nodes(episode_uuid: str, journal: str):
    result = await extract_nodes(episode_uuid, journal)
    return {"entities": result.resolved_count}
```

**Extract Edges (future):**
- Reads uuid_map from episode.attributes
- Extracts relationships between entities (including Self)
- Cleans up uuid_map after completion

**Optimization (future):**
- Import EntityExtractor in optimizer scripts
- Train with MIPROv2
- Save to `backend/graph/prompts/extract_nodes.json`
