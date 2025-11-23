"""Tests for global entity suppression feature.

Tests verify that deleted entities are globally suppressed across the journal,
preventing them from being re-extracted in any episode.
"""

import json
import pytest


@pytest.fixture(autouse=True)
def clear_suppression():
    """Clear suppression list before each test to prevent cross-test contamination."""
    from backend.settings import DEFAULT_JOURNAL
    from backend.database.redis_ops import redis_ops

    with redis_ops() as r:
        suppression_key = f"journal:{DEFAULT_JOURNAL}:suppressed_entities"
        r.delete(suppression_key)

    yield

    with redis_ops() as r:
        suppression_key = f"journal:{DEFAULT_JOURNAL}:suppressed_entities"
        r.delete(suppression_key)


@pytest.mark.inference
@pytest.mark.asyncio
async def test_deletion_suppresses_globally_across_episodes(isolated_graph, require_llm):
    """Test that deleting an entity from one episode suppresses it globally.

    Verifies:
    - Episode A extracts Alice and Bob
    - Deleting Bob from Episode A adds Bob to global suppression list
    - Episode B extraction filters out Bob (only Charlie extracted)
    - Suppression list contains "bob" (lowercase)
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import get_suppressed_entities

    content_a = "Today I met Alice and Bob at the park."
    episode_a_uuid = await add_journal_entry(content_a)

    result_a = await extract_nodes(episode_a_uuid, DEFAULT_JOURNAL)
    assert result_a.extracted_count >= 2, "Episode A should extract at least Alice and Bob"

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query_a = f"""
    MATCH (ep:Episodic {{uuid: '{episode_a_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name
    """
    entities_result_a = await driver.execute_query(entities_query_a)
    entities_a = entities_result_a[0]

    bob_entity = next((e for e in entities_a if "bob" in e["e.name"].lower()), None)
    assert bob_entity is not None, "Bob should be extracted in Episode A"

    bob_uuid = bob_entity["e.uuid"]

    await delete_entity_mention(episode_a_uuid, bob_uuid, DEFAULT_JOURNAL)

    suppressed = get_suppressed_entities(DEFAULT_JOURNAL)
    assert "bob" in suppressed, "Bob should be in global suppression list after deletion"

    content_b = "Bob and Charlie went to the store."
    episode_b_uuid = await add_journal_entry(content_b)

    result_b = await extract_nodes(episode_b_uuid, DEFAULT_JOURNAL)

    entities_query_b = f"""
    MATCH (ep:Episodic {{uuid: '{episode_b_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.name
    """
    entities_result_b = await driver.execute_query(entities_query_b)
    entities_b = entities_result_b[0]
    entity_names_b = {e["e.name"] for e in entities_b}

    has_bob = any("bob" in name.lower() for name in entity_names_b)
    assert not has_bob, "Bob should NOT be extracted in Episode B (globally suppressed)"

    has_charlie = any("charlie" in name.lower() for name in entity_names_b)
    assert has_charlie, "Charlie should be extracted in Episode B"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_reextraction_respects_suppression(isolated_graph, require_llm):
    """Test that re-extraction filters out suppressed entities.

    Verifies:
    - Episode initially has Alice and Bob
    - Delete Bob (adds to suppression)
    - Edit episode to mention Alice, Bob, and Charlie
    - Re-extraction should only extract Alice and Charlie (Bob suppressed)
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.lifecycle import _ensure_graph
    from backend.database.utils import to_cypher_literal
    from graphiti_core.nodes import EpisodicNode

    content_v1 = "Alice and Bob went to lunch."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count >= 2

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name
    """
    entities_result = await driver.execute_query(entities_query)
    entities = entities_result[0]

    bob_entity = next((e for e in entities if "bob" in e["e.name"].lower()), None)
    assert bob_entity is not None, "Bob should be extracted initially"

    bob_uuid = bob_entity["e.uuid"]

    await delete_entity_mention(episode_uuid, bob_uuid, DEFAULT_JOURNAL)

    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "Alice and Bob and Charlie had dinner together."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        query = f"""
        MATCH (ep:Episodic {{uuid: {to_cypher_literal(episode_uuid)}}})
        SET ep.content = {to_cypher_literal(episode.content)}
        RETURN ep.uuid
        """
        with lock:
            result = graph.query(query)
        return result

    import asyncio
    await asyncio.to_thread(_update_episode_sync)

    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    entities_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.name
    """
    entities_result2 = await driver.execute_query(entities_query2)
    entities2 = entities_result2[0]
    entity_names = {e["e.name"] for e in entities2}

    has_alice = any("alice" in name.lower() for name in entity_names)
    assert has_alice, "Alice should be extracted after re-extraction"

    has_bob = any("bob" in name.lower() for name in entity_names)
    assert not has_bob, "Bob should NOT be extracted (still suppressed)"

    has_charlie = any("charlie" in name.lower() for name in entity_names)
    assert has_charlie, "Charlie should be extracted after re-extraction"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_cache_state_after_deletion(isolated_graph, require_llm):
    """Test that all cache keys are properly updated after entity deletion.

    Verifies:
    - Episode extracts Alice and Bob (cache populated)
    - Delete Bob
    - nodes cache updated (Bob removed)
    - mentions_edges cache updated (Bob's edge removed)
    - uuid_map cache updated (Bob's mappings removed)
    - suppressed_entities contains "bob"
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import redis_ops, get_suppressed_entities

    content = "Alice and Bob are friends."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result.extracted_count >= 2

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name, r.uuid as edge_uuid
    """
    entities_result = await driver.execute_query(entities_query)
    entities = entities_result[0]

    bob_entity = next((e for e in entities if "bob" in e["e.name"].lower()), None)
    assert bob_entity is not None, "Bob should be extracted"

    bob_uuid = bob_entity["e.uuid"]
    bob_edge_uuid = bob_entity["edge_uuid"]

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"

        nodes_before_json = r.hget(cache_key, "nodes")
        assert nodes_before_json is not None, "nodes cache should be populated"
        nodes_before = json.loads(nodes_before_json.decode())
        assert len(nodes_before) >= 2, "Should have at least 2 nodes cached"

        edges_before_json = r.hget(cache_key, "mentions_edges")
        assert edges_before_json is not None, "mentions_edges cache should be populated"
        edges_before = json.loads(edges_before_json.decode())
        assert len(edges_before) >= 2, "Should have at least 2 edges cached"
        assert bob_edge_uuid in edges_before, "Bob's edge should be in cache before deletion"

    await delete_entity_mention(episode_uuid, bob_uuid, DEFAULT_JOURNAL)

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"

        nodes_after_json = r.hget(cache_key, "nodes")
        assert nodes_after_json is not None, "nodes cache should still exist"
        nodes_after = json.loads(nodes_after_json.decode())

        bob_in_cache = any(n["uuid"] == bob_uuid for n in nodes_after)
        assert not bob_in_cache, "Bob should be removed from nodes cache"

        edges_after_json = r.hget(cache_key, "mentions_edges")
        assert edges_after_json is not None, "mentions_edges cache should still exist"
        edges_after = json.loads(edges_after_json.decode())
        assert bob_edge_uuid not in edges_after, "Bob's edge should be removed from cache"

    suppressed = get_suppressed_entities(DEFAULT_JOURNAL)
    assert "bob" in suppressed, "Bob should be in suppressed_entities"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_case_insensitive_suppression(isolated_graph, require_llm):
    """Test that suppression matching is case-insensitive.

    Verifies:
    - Delete "bob" (lowercase stored)
    - Extract content with "Bob" (capitalized)
    - "Bob" should be filtered out despite different case
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import get_suppressed_entities

    content_v1 = "I met bob at the store."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count >= 1

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name
    """
    entities_result = await driver.execute_query(entities_query)
    entities = entities_result[0]

    bob_entity = next((e for e in entities if "bob" in e["e.name"].lower()), None)
    assert bob_entity is not None, "bob should be extracted"

    bob_uuid = bob_entity["e.uuid"]

    await delete_entity_mention(episode_uuid, bob_uuid, DEFAULT_JOURNAL)

    suppressed = get_suppressed_entities(DEFAULT_JOURNAL)
    assert "bob" in suppressed, "bob should be in suppression list (lowercase)"

    content_v2 = "Bob and Alice went to the park."
    episode2_uuid = await add_journal_entry(content_v2)

    result2 = await extract_nodes(episode2_uuid, DEFAULT_JOURNAL)

    entities_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode2_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.name
    """
    entities_result2 = await driver.execute_query(entities_query2)
    entities2 = entities_result2[0]
    entity_names = {e["e.name"] for e in entities2}

    has_bob = any("bob" in name.lower() for name in entity_names)
    assert not has_bob, "Bob (capitalized) should be filtered out despite case difference"

    has_alice = any("alice" in name.lower() for name in entity_names)
    assert has_alice, "Alice should be extracted"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_all_entities_suppressed(isolated_graph, require_llm):
    """Test that extraction handles all entities being suppressed.

    Verifies:
    - Episode has Alice and Bob
    - Extract, then delete both
    - Re-extract same content
    - Extraction result is empty (0 entities)
    - Cache updated with empty arrays
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import redis_ops

    content = "Alice and Bob went shopping."
    episode_uuid = await add_journal_entry(content)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count >= 2

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name
    """
    entities_result = await driver.execute_query(entities_query)
    entities = entities_result[0]

    for entity in entities:
        await delete_entity_mention(episode_uuid, entity["e.uuid"], DEFAULT_JOURNAL)

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    from backend.database.utils import to_cypher_literal

    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        query = f"""
        MATCH (ep:Episodic {{uuid: {to_cypher_literal(episode_uuid)}}})
        SET ep.content = {to_cypher_literal(episode.content)}
        RETURN ep.uuid
        """
        with lock:
            result = graph.query(query)
        return result

    import asyncio
    await asyncio.to_thread(_update_episode_sync)

    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result2.resolved_count == 0, "Should have 0 entities after all suppressed"

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"

        nodes_json = r.hget(cache_key, "nodes")
        assert nodes_json is not None, "nodes cache should exist"
        nodes = json.loads(nodes_json.decode())
        assert nodes == [], "nodes cache should be empty array"

        edges_json = r.hget(cache_key, "mentions_edges")
        assert edges_json is not None, "mentions_edges cache should exist"
        edges = json.loads(edges_json.decode())
        assert edges == [], "mentions_edges cache should be empty array"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_uuid_map_cleanup(isolated_graph, require_llm):
    """Test that uuid_map cache is cleaned up after entity deletion.

    Verifies:
    - Extract entities (uuid_map populated in result)
    - Delete one entity
    - uuid_map no longer contains deleted entity UUID
    - Other mappings preserved
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention

    content = "Alice, Bob, and Charlie are colleagues."
    episode_uuid = await add_journal_entry(content)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count >= 2

    uuid_map_before = result1.uuid_map
    assert len(uuid_map_before) >= 2, "uuid_map should have mappings"

    driver = get_driver(DEFAULT_JOURNAL)

    entities_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN e.uuid, e.name
    """
    entities_result = await driver.execute_query(entities_query)
    entities = entities_result[0]

    bob_entity = next((e for e in entities if "bob" in e["e.name"].lower()), None)
    if bob_entity:
        bob_uuid = bob_entity["e.uuid"]

        bob_mappings_before = [prov for prov, canon in uuid_map_before.items() if canon == bob_uuid]
        assert len(bob_mappings_before) > 0, "Bob should have uuid mappings before deletion"

        await delete_entity_mention(episode_uuid, bob_uuid, DEFAULT_JOURNAL)

        result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

        uuid_map_after = result2.uuid_map

        bob_mappings_after = [prov for prov, canon in uuid_map_after.items() if canon == bob_uuid]
        assert len(bob_mappings_after) == 0, "Bob's UUID should not appear in uuid_map after deletion"

        other_mappings = [prov for prov, canon in uuid_map_after.items() if canon != bob_uuid]
        assert len(other_mappings) > 0, "Other entity mappings should be preserved"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_multiple_edge_deletion(isolated_graph, require_llm):
    """Test deletion when entity might have multiple mentions in same episode.

    Verifies:
    - Episode mentions entity (potentially multiple times)
    - Delete entity
    - All MENTIONS edges to that entity are removed from cache
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import redis_ops

    content = "Alice met Bob. Later, Alice called Bob again. Bob was helpful."
    episode_uuid = await add_journal_entry(content)

    result = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result.extracted_count >= 2

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) CONTAINS 'bob'
    RETURN e.uuid, count(r) as edge_count, collect(r.uuid) as edge_uuids
    """
    mentions_result = await driver.execute_query(mentions_query)

    if len(mentions_result[0]) > 0:
        bob_data = mentions_result[0][0]
        bob_uuid = bob_data["e.uuid"]
        bob_edge_uuids = set(bob_data["edge_uuids"])

        with redis_ops() as r:
            cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
            edges_before_json = r.hget(cache_key, "mentions_edges")
            assert edges_before_json is not None
            edges_before = set(json.loads(edges_before_json.decode()))

            for edge_uuid in bob_edge_uuids:
                assert edge_uuid in edges_before, f"Bob's edge {edge_uuid} should be in cache before deletion"

        await delete_entity_mention(episode_uuid, bob_uuid, DEFAULT_JOURNAL)

        with redis_ops() as r:
            cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
            edges_after_json = r.hget(cache_key, "mentions_edges")

            if edges_after_json:
                edges_after = set(json.loads(edges_after_json.decode()))

                for edge_uuid in bob_edge_uuids:
                    assert edge_uuid not in edges_after, f"Bob's edge {edge_uuid} should be removed from cache after deletion"
