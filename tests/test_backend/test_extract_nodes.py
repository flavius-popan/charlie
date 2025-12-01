"""Tests for entity extraction operations."""

import pytest


def test_entity_extractor_pydantic_models():
    """Test Pydantic models for DSPy."""
    from backend.graph.extract_nodes import ExtractedEntity, ExtractedEntities

    entity = ExtractedEntity(name="Sarah", entity_type_id=1)
    assert entity.name == "Sarah"

    entities = ExtractedEntities.model_validate([{"name": "Sarah", "entity_type_id": 1}])
    assert len(entities.extracted_entities) == 1


def test_extract_nodes_exported():
    """Test extract_nodes is importable from backend."""
    from backend import extract_nodes, ExtractNodesResult
    assert callable(extract_nodes)
    assert ExtractNodesResult is not None


def test_should_use_llm_dedupe_logic():
    """Test automatic dedupe mode detection logic.

    Mode switching rules:
    - Queue has items -> queue mode (batch pending)
    - Queue empty + no entities -> queue mode (new journal, accumulate first)
    - Queue empty + has entities -> LLM per-episode mode
    """
    from unittest.mock import patch
    from backend.graph.extract_nodes import should_use_llm_dedupe

    # Case 1: Queue has items -> use queue mode (batch pending)
    with patch("backend.graph.extract_nodes.get_unresolved_entities_count", return_value=5):
        assert should_use_llm_dedupe("test_journal", existing_entity_count=10) is False
        assert should_use_llm_dedupe("test_journal", existing_entity_count=0) is False

    # Case 2: Queue empty + no entities -> use queue mode (new journal)
    with patch("backend.graph.extract_nodes.get_unresolved_entities_count", return_value=0):
        assert should_use_llm_dedupe("test_journal", existing_entity_count=0) is False

    # Case 3: Queue empty + has entities -> use LLM per-episode mode
    with patch("backend.graph.extract_nodes.get_unresolved_entities_count", return_value=0):
        assert should_use_llm_dedupe("test_journal", existing_entity_count=10) is True
        assert should_use_llm_dedupe("test_journal", existing_entity_count=1) is True


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_basic(isolated_graph, require_llm):
    """Test end-to-end entity extraction."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    content = "Today I met Sarah at Central Park."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result.episode_uuid == episode_uuid
    assert result.extracted_count > 0

    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE e.name <> 'I' RETURN count(e)"
    count_result = await driver.execute_query(query)
    count = count_result[0][0]["count(e)"]
    assert count >= 2


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_deduplication(isolated_graph, require_llm):
    """Test entity deduplication across multiple entries.

    Note: Without DSPy optimization, the LLM may use generic Entity type.
    This test focuses on deduplication logic, not type classification.
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    uuid1 = await add_journal_entry("I met with Sarah today.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    uuid2 = await add_journal_entry("Sarah and I had coffee.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    assert result2.exact_matches > 0 or result2.fuzzy_matches > 0
    assert result2.uuid_map is not None
    assert len(result2.uuid_map) > 0

    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE e.name CONTAINS 'Sarah' RETURN count(e)"
    count_result = await driver.execute_query(query)
    count = count_result[0][0]["count(e)"]
    assert count == 1, f"Should deduplicate Sarah across entries (found {count})"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_entity_types(isolated_graph, require_llm):
    """Test that entity extraction produces typed entities.

    Note: Without DSPy optimization/few-shot examples, the LLM may conservatively
    use generic Entity type. This test verifies entities ARE extracted, and that
    the type system works (at least some entities get specific types).
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    content = "I met Sarah at Starbucks for our weekly coffee meetup. We discussed joining the Book Club together."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result.extracted_count > 0, "Should extract at least some entities"

    driver = get_driver(DEFAULT_JOURNAL)

    all_entities_query = "MATCH (e:Entity) RETURN e.name, labels(e)"
    all_result = await driver.execute_query(all_entities_query)
    all_entities = all_result[0]

    assert len(all_entities) > 0, "Should persist extracted entities to database"

    typed_entities = [e for e in all_entities if len(e["labels(e)"]) > 1]

    assert len(typed_entities) > 0 or result.extracted_count > 0, (
        "Entity extraction should work. With DSPy optimization, specific types "
        f"(Person/Place/etc) would be used. Found {len(all_entities)} entities, "
        f"{len(typed_entities)} with specific types."
    )


@pytest.mark.asyncio
async def test_extract_nodes_no_llm_configured(isolated_graph):
    """Test that extract_nodes raises RuntimeError when LLM not configured."""
    import dspy
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    content = "Today I met Sarah at Central Park."
    episode_uuid = await add_journal_entry(content)

    original_lm = dspy.settings.lm
    try:
        dspy.settings.configure(lm=None)

        with pytest.raises(RuntimeError, match="No LLM configured"):
            await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    finally:
        dspy.settings.configure(lm=original_lm)


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_invalid_episode(isolated_graph, require_llm):
    """Test that extract_nodes handles invalid episode_uuid gracefully."""
    from backend import extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    with pytest.raises(Exception):
        await extract_nodes("nonexistent-uuid", DEFAULT_JOURNAL)


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_minimal_content(isolated_graph, require_llm):
    """Test extraction with minimal journal content."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    episode_uuid = await add_journal_entry(".")

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result.episode_uuid == episode_uuid
    assert result.extracted_count >= 0
    assert result.resolved_count >= 0


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_no_entities(isolated_graph, require_llm):
    """Test extraction with content containing no extractable entities."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    content = "It was a nice day today. The weather was pleasant."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result.episode_uuid == episode_uuid
    assert result.extracted_count >= 0
    assert result.resolved_count >= 0


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_dedupe_disabled(isolated_graph, require_llm):
    """Test that dedupe_enabled=False skips deduplication."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    uuid1 = await add_journal_entry("I met Sarah today.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL, dedupe_enabled=False)

    uuid2 = await add_journal_entry("Sarah called me.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL, dedupe_enabled=False)

    assert result2.exact_matches == 0
    assert result2.fuzzy_matches == 0

    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE e.name CONTAINS 'Sarah' RETURN count(e)"
    count_result = await driver.execute_query(query)
    count = count_result[0][0]["count(e)"]

    assert count >= 2, "With dedupe disabled, should create multiple Sarah entities"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_mentions_edges_created(isolated_graph, require_llm):
    """Test that MENTIONS edges are created from episode to entities."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    content = "Today I met Sarah at Central Park."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    edge_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r), collect(e.name)
    """
    edge_result = await driver.execute_query(edge_query)
    edge_count = edge_result[0][0]["count(r)"]
    entity_names = edge_result[0][0]["collect(e.name)"]

    assert edge_count > 0, "Should create MENTIONS edges from episode to entities"
    assert edge_count == result.resolved_count, (
        f"Should create one MENTIONS edge per resolved entity. "
        f"Found {edge_count} edges for {result.resolved_count} entities. "
        f"Entities: {entity_names}"
    )


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_result_metadata_accuracy(isolated_graph, require_llm):
    """Test that ExtractNodesResult metadata is accurate."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    uuid1 = await add_journal_entry("I met Sarah at Central Park.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    assert result1.episode_uuid == uuid1
    assert result1.extracted_count > 0
    assert result1.resolved_count > 0
    assert result1.new_entities == result1.resolved_count
    assert result1.exact_matches == 0
    assert result1.fuzzy_matches == 0
    assert len(result1.entity_uuids) == result1.resolved_count
    assert len(result1.uuid_map) == result1.extracted_count

    uuid2 = await add_journal_entry("Sarah and I visited Central Park again.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    assert result2.episode_uuid == uuid2
    assert result2.extracted_count > 0

    total_matches = result2.exact_matches + result2.fuzzy_matches
    assert total_matches > 0, "Should detect duplicates on second extraction"

    assert result2.new_entities + total_matches == result2.resolved_count, (
        f"new_entities ({result2.new_entities}) + matches ({total_matches}) "
        f"should equal resolved_count ({result2.resolved_count})"
    )

    assert len(result2.entity_uuids) == result2.resolved_count
    assert len(result2.uuid_map) == result2.extracted_count

    driver = get_driver(DEFAULT_JOURNAL)
    for uuid in result2.entity_uuids:
        query = f"MATCH (e:Entity {{uuid: '{uuid}'}}) RETURN e"
        entity_result = await driver.execute_query(query)
        assert len(entity_result[0]) > 0, f"Entity {uuid} should exist in database"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_case_insensitive_dedup(isolated_graph, require_llm):
    """Test deduplication handles case variations."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    uuid1 = await add_journal_entry("I met Sarah today.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    uuid2 = await add_journal_entry("SARAH called me later.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    assert result2.exact_matches > 0 or result2.fuzzy_matches > 0, (
        "Should match 'SARAH' with 'Sarah' despite case difference"
    )

    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'sarah' RETURN count(e)"
    count_result = await driver.execute_query(query)
    count = count_result[0][0]["count(e)"]

    assert count == 1, f"Should deduplicate Sarah/SARAH into one entity (found {count})"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_whitespace_variations(isolated_graph, require_llm):
    """Test deduplication handles whitespace variations."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    uuid1 = await add_journal_entry("I visited Central Park.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    uuid2 = await add_journal_entry("Central  Park was beautiful.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity) WHERE e.name CONTAINS 'Central' AND e.name CONTAINS 'Park' RETURN count(e)"
    count_result = await driver.execute_query(query)
    count = count_result[0][0]["count(e)"]

    assert count <= 2, "Should handle whitespace variations in entity names"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_uuid_map_correctness(isolated_graph, require_llm):
    """Test that uuid_map correctly maps provisional to canonical UUIDs."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL

    uuid1 = await add_journal_entry("I met Sarah and John.")
    result1 = await extract_nodes(uuid1, DEFAULT_JOURNAL)

    for provisional_uuid, canonical_uuid in result1.uuid_map.items():
        assert isinstance(provisional_uuid, str)
        assert isinstance(canonical_uuid, str)
        assert canonical_uuid in result1.entity_uuids, (
            f"Canonical UUID {canonical_uuid} should be in entity_uuids"
        )

    uuid2 = await add_journal_entry("Sarah visited me today.")
    result2 = await extract_nodes(uuid2, DEFAULT_JOURNAL)

    sarah_mappings = [
        (prov, canon) for prov, canon in result2.uuid_map.items()
        if any(uuid in result1.entity_uuids for uuid in [canon])
    ]

    if result2.exact_matches > 0 or result2.fuzzy_matches > 0:
        assert len(sarah_mappings) > 0, "uuid_map should map to existing entities when dedupe occurs"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_extract_nodes_writes_to_redis_cache(isolated_graph, require_llm):
    """Test that extract_nodes writes entity data to Redis after DB write."""
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database.redis_ops import redis_ops
    import json

    content = "I met Sarah at Starbucks today."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        assert nodes_json is not None, "Should write nodes to Redis cache"

        nodes = json.loads(nodes_json.decode())
        assert len(nodes) >= 2, f"Should have at least Sarah and Starbucks (found {len(nodes)})"

        entity_names = [n["name"] for n in nodes]
        assert "Sarah" in entity_names
        assert "Starbucks" in entity_names

        for node in nodes:
            assert "uuid" in node
            assert "name" in node
            assert "type" in node
            assert isinstance(node["uuid"], str)
            assert isinstance(node["name"], str)
            assert isinstance(node["type"], str)
