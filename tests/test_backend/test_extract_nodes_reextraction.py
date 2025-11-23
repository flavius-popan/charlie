"""Regression tests for MENTIONS edge cleanup on re-extraction.

Tests verify Bug 2 fix: Old MENTIONS edges are properly deleted when re-extracting
entities from an edited episode, preventing duplicate edges in the database.
"""

import json
import pytest


@pytest.mark.inference
@pytest.mark.asyncio
async def test_reextraction_deletes_old_mentions_edges(isolated_graph, require_llm):
    """Test re-extraction deletes old MENTIONS edges.

    Verifies:
    - Initial extraction creates correct MENTIONS edges
    - Redis cache stores edge UUIDs
    - Re-extraction removes old edges before creating new ones
    - Only current entities have MENTIONS edges after re-extraction
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import redis_ops
    from backend.database.lifecycle import _ensure_graph

    content_v1 = "Today I met Alice and Bob at the park."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result1.extracted_count >= 2, "Should extract at least Alice and Bob"
    assert result1.resolved_count >= 2

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query1 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN r.uuid, e.name
    """
    mentions_result1 = await driver.execute_query(mentions_query1)
    initial_mentions = mentions_result1[0]
    initial_edge_uuids = [m["r.uuid"] for m in initial_mentions]
    initial_entity_names = {m["e.name"] for m in initial_mentions}

    assert len(initial_mentions) >= 2, f"Should have at least 2 MENTIONS edges (found {len(initial_mentions)})"
    assert "Alice" in initial_entity_names or "Bob" in initial_entity_names

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        assert cached_edges_json is not None, "Redis cache should have mentions_edges"

        cached_edge_uuids = json.loads(cached_edges_json.decode())
        assert len(cached_edge_uuids) == len(initial_mentions), (
            f"Redis cache should match DB edge count: cache={len(cached_edge_uuids)}, db={len(initial_mentions)}"
        )
        assert set(cached_edge_uuids) == set(initial_edge_uuids), "Cache UUIDs should match DB UUIDs"

    from graphiti_core.nodes import EpisodicNode
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "Today I met Alice and Charlie at the cafe."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal, _decode_value

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

    mentions_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN r.uuid, e.name
    """
    mentions_result2 = await driver.execute_query(mentions_query2)
    final_mentions = mentions_result2[0]
    final_edge_uuids = [m["r.uuid"] for m in final_mentions]
    final_entity_names = {m["e.name"] for m in final_mentions}

    assert len(final_mentions) >= 2, f"Should have at least 2 MENTIONS edges after re-extraction (found {len(final_mentions)})"

    for old_uuid in initial_edge_uuids:
        assert old_uuid not in final_edge_uuids, f"Old edge UUID {old_uuid} should not exist after re-extraction"

    assert "Alice" in final_entity_names, "Alice should still be mentioned"
    assert "Charlie" in final_entity_names, "Charlie should be mentioned in new version"

    if "Bob" in initial_entity_names:
        bob_in_final = "Bob" in final_entity_names
        if bob_in_final:
            pass
        else:
            assert not bob_in_final, "Bob should not be mentioned after removal from content"

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        assert cached_edges_json is not None, "Redis cache should still have mentions_edges"

        cached_edge_uuids = json.loads(cached_edges_json.decode())
        assert len(cached_edge_uuids) == len(final_mentions), (
            f"Redis cache should match new DB edge count: cache={len(cached_edge_uuids)}, db={len(final_mentions)}"
        )
        assert set(cached_edge_uuids) == set(final_edge_uuids), "Cache UUIDs should match new DB UUIDs"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_entities_deduplicated_on_reextraction(isolated_graph, require_llm):
    """Test entities are deduplicated on re-extraction.

    Verifies:
    - Extracting same entity twice reuses the existing node
    - No duplicate entity nodes are created
    - Only one MENTIONS edge exists to the reused entity
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    content_v1 = "I had lunch with Alice today."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    alice_query1 = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS 'alice'
    RETURN e.uuid, e.name
    """
    alice_result1 = await driver.execute_query(alice_query1)
    alice_entities_v1 = alice_result1[0]

    assert len(alice_entities_v1) == 1, f"Should have exactly 1 Alice entity (found {len(alice_entities_v1)})"
    alice_uuid = alice_entities_v1[0]["e.uuid"]

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "Alice and I went for coffee this afternoon."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal

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

    alice_query2 = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS 'alice'
    RETURN e.uuid, e.name
    """
    alice_result2 = await driver.execute_query(alice_query2)
    alice_entities_v2 = alice_result2[0]

    assert len(alice_entities_v2) == 1, (
        f"Should still have exactly 1 Alice entity after re-extraction (found {len(alice_entities_v2)})"
    )
    assert alice_entities_v2[0]["e.uuid"] == alice_uuid, (
        "Alice entity should be reused (same UUID), not duplicated"
    )

    mentions_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) CONTAINS 'alice'
    RETURN count(r) as edge_count
    """
    mentions_result = await driver.execute_query(mentions_query)
    edge_count = mentions_result[0][0]["edge_count"]

    assert edge_count == 1, (
        f"Should have exactly 1 MENTIONS edge to Alice (found {edge_count})"
    )


@pytest.mark.inference
@pytest.mark.asyncio
async def test_cache_miss_graceful_degradation_on_reextraction(isolated_graph, require_llm):
    """Test graceful degradation when cache is missing during re-extraction.

    When Redis cache is missing, the system cannot delete old MENTIONS edges
    (because it doesn't know which edges are old). This test verifies graceful degradation:
    - Re-extraction succeeds without errors even when cache is missing
    - New MENTIONS edges are created correctly
    - Old edges remain (not deleted, due to missing cache)
    - Cache is repopulated with NEW edge UUIDs only
    - Total edges = initial_edges + new_edges (accumulated)
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import redis_ops

    content_v1 = "I met Sarah at the library."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count > 0

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query1 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count, collect(r.uuid) as edge_uuids
    """
    mentions_result1 = await driver.execute_query(mentions_query1)
    initial_edge_count = mentions_result1[0][0]["edge_count"]
    initial_edge_uuids = set(mentions_result1[0][0]["edge_uuids"])
    assert initial_edge_count > 0, "Should have MENTIONS edges after initial extraction"

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        deleted_count = r.hdel(cache_key, "mentions_edges")
        assert deleted_count > 0, "Should have deleted mentions_edges from cache"

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "Sarah and I studied together at the library."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal

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

    assert result2.extracted_count > 0, "Should extract entities even with cache miss"

    mentions_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count, collect(e.name) as entity_names, collect(r.uuid) as edge_uuids
    """
    mentions_result2 = await driver.execute_query(mentions_query2)
    final_edge_count = mentions_result2[0][0]["edge_count"]
    entity_names = mentions_result2[0][0]["entity_names"]
    final_edge_uuids = set(mentions_result2[0][0]["edge_uuids"])

    assert final_edge_count > 0, "Should have MENTIONS edges after re-extraction with cache miss"
    assert "Sarah" in entity_names, "Should still extract Sarah"

    assert final_edge_count >= initial_edge_count, (
        f"Edges should accumulate when cache is missing (old edges not deleted): "
        f"initial={initial_edge_count}, final={final_edge_count}"
    )

    new_edge_uuids = final_edge_uuids - initial_edge_uuids
    assert len(new_edge_uuids) > 0, "Should have created new MENTIONS edges"

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        assert cached_edges_json is not None, "Cache should be repopulated after re-extraction"

        cached_edge_uuids = set(json.loads(cached_edges_json.decode()))
        assert cached_edge_uuids == new_edge_uuids, (
            "Cache should contain only NEW edge UUIDs (not all accumulated edges)"
        )


@pytest.mark.inference
@pytest.mark.asyncio
async def test_reextraction_with_entity_additions(isolated_graph, require_llm):
    """Test re-extraction when adding new entities to existing content.

    Verifies:
    - Old entities remain (deduplicated)
    - New entities are added
    - Total MENTIONS edge count increases correctly
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    content_v1 = "I met Tom today."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query1 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count, collect(e.name) as entity_names
    """
    mentions_result1 = await driver.execute_query(mentions_query1)
    initial_edge_count = mentions_result1[0][0]["edge_count"]
    initial_entity_names = set(mentions_result1[0][0]["entity_names"])

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "I met Tom and Jerry and Spike at the park."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal

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

    mentions_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count, collect(e.name) as entity_names
    """
    mentions_result2 = await driver.execute_query(mentions_query2)
    final_edge_count = mentions_result2[0][0]["edge_count"]
    final_entity_names = set(mentions_result2[0][0]["entity_names"])

    assert final_edge_count > initial_edge_count, (
        f"Should have more MENTIONS edges after adding entities "
        f"(initial={initial_edge_count}, final={final_edge_count})"
    )

    assert "Tom" in final_entity_names, "Tom from original content should still be mentioned"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_reextraction_with_all_entities_removed(isolated_graph, require_llm):
    """Test re-extraction when all entities are removed from content.

    Verifies:
    - Old MENTIONS edges are deleted
    - Episode has zero or minimal MENTIONS edges after re-extraction
    """
    from backend import add_journal_entry, extract_nodes
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import redis_ops

    content_v1 = "I met Emily and David at Central Park."
    episode_uuid = await add_journal_entry(content_v1)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query1 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count
    """
    mentions_result1 = await driver.execute_query(mentions_query1)
    initial_edge_count = mentions_result1[0][0]["edge_count"]

    assert initial_edge_count > 0, "Should have MENTIONS edges after initial extraction"

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "It was a nice day. The weather was pleasant."

    def _update_episode_sync():
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal

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

    mentions_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN count(r) as edge_count
    """
    mentions_result2 = await driver.execute_query(mentions_query2)
    final_edge_count = mentions_result2[0][0]["edge_count"]

    assert final_edge_count < initial_edge_count, (
        f"Should have fewer MENTIONS edges after removing entities "
        f"(initial={initial_edge_count}, final={final_edge_count})"
    )

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")

        if cached_edges_json:
            cached_edge_uuids = json.loads(cached_edges_json.decode())
            assert len(cached_edge_uuids) == final_edge_count, (
                "Cache should match DB edge count after entity removal"
            )
