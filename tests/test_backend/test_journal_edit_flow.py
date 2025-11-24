"""Integration tests for full journal edit and re-extraction flow.

Tests verify the complete user flow from creating a journal entry, extracting entities,
editing the entry, and re-extracting - ensuring no duplicate edges and proper cleanup.
"""

import json
import pytest


@pytest.mark.inference
@pytest.mark.asyncio
async def test_full_journal_edit_and_reextraction_flow(isolated_graph, require_llm):
    """Test full journal edit and re-extraction flow.

    Simulates the complete user workflow:
    1. Create journal entry
    2. Extract nodes (via task)
    3. Edit entry content
    4. Re-extract nodes (via task)
    5. Verify database integrity (no duplicate edges, proper cleanup)

    This is the primary integration test for Bug 2 fix.
    """
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import redis_ops, set_episode_status
    from backend.graph.extract_nodes import extract_nodes

    content_v1 = "Today I visited the museum with Rachel and Michael."
    episode_uuid = await add_journal_entry(content_v1)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result1.extracted_count is not None, "Task should return extraction results"
    assert result1.extracted_count > 0, "Should extract entities from initial content"

    if result1.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result1.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query1 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN r.uuid, e.name
    ORDER BY e.name
    """
    mentions_result1 = await driver.execute_query(mentions_query1)
    initial_mentions = mentions_result1[0]
    initial_edge_uuids = [m["r.uuid"] for m in initial_mentions]
    initial_entity_names = [m["e.name"] for m in initial_mentions]

    assert len(initial_mentions) > 0, "Should have MENTIONS edges after initial extraction"

    entity_query1 = """
    MATCH (e:Entity)
    WHERE e.name <> 'I'
    RETURN e.uuid, e.name
    ORDER BY e.name
    """
    entity_result1 = await driver.execute_query(entity_query1)
    initial_entities = entity_result1[0]
    initial_entity_uuids = {e["e.uuid"]: e["e.name"] for e in initial_entities}

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        assert cached_edges_json is not None, "Redis should cache mentions_edges"

        initial_cached_uuids = json.loads(cached_edges_json.decode())
        assert set(initial_cached_uuids) == set(initial_edge_uuids), (
            "Redis cache should match DB edge UUIDs"
        )

    from graphiti_core.nodes import EpisodicNode
    from backend.database.lifecycle import _ensure_graph
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "Today I visited the museum with Rachel and visited the art gallery."

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

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    assert result2.extracted_count is not None, "Task should return extraction results after re-extraction"
    assert result2.extracted_count > 0, "Should extract entities from edited content"

    if result2.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result2.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    mentions_query2 = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN r.uuid, e.name
    ORDER BY e.name
    """
    mentions_result2 = await driver.execute_query(mentions_query2)
    final_mentions = mentions_result2[0]
    final_edge_uuids = [m["r.uuid"] for m in final_mentions]
    final_entity_names = [m["e.name"] for m in final_mentions]

    for old_uuid in initial_edge_uuids:
        assert old_uuid not in final_edge_uuids, (
            f"Old MENTIONS edge {old_uuid} should be deleted during re-extraction"
        )

    mentions_by_entity = {}
    for mention in final_mentions:
        name = mention["e.name"]
        if name not in mentions_by_entity:
            mentions_by_entity[name] = []
        mentions_by_entity[name].append(mention["r.uuid"])

    for entity_name, edge_uuids in mentions_by_entity.items():
        assert len(edge_uuids) == 1, (
            f"Entity '{entity_name}' should have exactly 1 MENTIONS edge, found {len(edge_uuids)}: {edge_uuids}"
        )

    entity_query2 = """
    MATCH (e:Entity)
    WHERE e.name <> 'I'
    RETURN e.uuid, e.name
    ORDER BY e.name
    """
    entity_result2 = await driver.execute_query(entity_query2)
    final_entities = entity_result2[0]

    entity_counts = {}
    for entity in final_entities:
        name = entity["e.name"]
        if name in entity_counts:
            entity_counts[name] += 1
        else:
            entity_counts[name] = 1

    for name, count in entity_counts.items():
        assert count == 1, (
            f"Entity '{name}' should exist exactly once, found {count} times"
        )

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        assert cached_edges_json is not None, "Redis should cache mentions_edges after re-extraction"

        final_cached_uuids = json.loads(cached_edges_json.decode())
        assert set(final_cached_uuids) == set(final_edge_uuids), (
            "Redis cache should match DB edge UUIDs after re-extraction"
        )
        assert len(final_cached_uuids) == len(final_mentions), (
            f"Cache should have {len(final_mentions)} edges, found {len(final_cached_uuids)}"
        )

    cached_nodes_json = r.hget(cache_key, "nodes")
    assert cached_nodes_json is not None, "Redis should cache nodes"

    cached_nodes = json.loads(cached_nodes_json.decode())
    assert len(cached_nodes) > 0, "Should have cached entity nodes"

    for node in cached_nodes:
        assert "uuid" in node
        assert "name" in node
        assert "type" in node


@pytest.mark.inference
@pytest.mark.asyncio
async def test_multiple_reextractions_maintain_consistency(isolated_graph, require_llm):
    """Test multiple re-extractions maintain database consistency.

    Verifies:
    - Multiple edit-extract cycles work correctly
    - No edge duplication across multiple re-extractions
    - Entity deduplication works across multiple cycles
    """
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import set_episode_status
    from backend.graph.extract_nodes import extract_nodes
    from backend.database.lifecycle import _ensure_graph

    content_v1 = "I met Lisa at the coffee shop."
    episode_uuid = await add_journal_entry(content_v1)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result1.extracted_count > 0

    if result1.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result1.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    def _verify_no_duplicate_edges():
        query = f"""
        MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
        RETURN e.name, count(r) as edge_count
        """
        return driver.execute_query(query)

    verify_result1 = await _verify_no_duplicate_edges()
    for row in verify_result1[0]:
        assert row["edge_count"] == 1, f"Entity {row['e.name']} should have exactly 1 edge, found {row['edge_count']}"

    from graphiti_core.nodes import EpisodicNode
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "I met Lisa and Mark at the coffee shop."

    def _update_episode_sync(new_content):
        graph, lock = _ensure_graph(DEFAULT_JOURNAL)
        from backend.database.utils import to_cypher_literal

        query = f"""
        MATCH (ep:Episodic {{uuid: {to_cypher_literal(episode_uuid)}}})
        SET ep.content = {to_cypher_literal(new_content)}
        RETURN ep.uuid
        """
        with lock:
            result = graph.query(query)
        return result

    import asyncio
    await asyncio.to_thread(_update_episode_sync, episode.content)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result2.extracted_count > 0

    if result2.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result2.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    verify_result2 = await _verify_no_duplicate_edges()
    for row in verify_result2[0]:
        assert row["edge_count"] == 1, f"After 2nd extraction: Entity {row['e.name']} should have exactly 1 edge, found {row['edge_count']}"

    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "I met Lisa at the bookstore with Sarah."

    await asyncio.to_thread(_update_episode_sync, episode.content)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    result3 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)
    assert result3.extracted_count > 0

    if result3.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result3.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    verify_result3 = await _verify_no_duplicate_edges()
    for row in verify_result3[0]:
        assert row["edge_count"] == 1, f"After 3rd extraction: Entity {row['e.name']} should have exactly 1 edge, found {row['edge_count']}"

    lisa_query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS 'lisa'
    RETURN count(e) as lisa_count
    """
    lisa_result = await driver.execute_query(lisa_query)
    lisa_count = lisa_result[0][0]["lisa_count"]
    assert lisa_count == 1, f"Lisa should exist exactly once across all extractions, found {lisa_count}"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_reextraction_idempotency(isolated_graph, require_llm):
    """Test that re-extracting same content is idempotent.

    Verifies:
    - Re-extracting unchanged content produces same results
    - No edge duplication when content hasn't changed
    - Cache remains consistent
    """
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver
    from backend.database.redis_ops import redis_ops, set_episode_status
    from backend.graph.extract_nodes import extract_nodes

    content = "I visited the park with Jennifer."
    episode_uuid = await add_journal_entry(content)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    if result1.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result1.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    driver = get_driver(DEFAULT_JOURNAL)

    mentions_query = f"""
    MATCH (ep:Episodic {{uuid: '{episode_uuid}'}})-[r:MENTIONS]->(e:Entity)
    RETURN r.uuid, e.name
    ORDER BY e.name
    """
    mentions_result1 = await driver.execute_query(mentions_query)
    edges_after_first = mentions_result1[0]
    edge_uuids_first = {m["r.uuid"] for m in edges_after_first}
    entity_names_first = {m["e.name"] for m in edges_after_first}

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        cached_uuids_first = set(json.loads(cached_edges_json.decode()))

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    if result2.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result2.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    mentions_result2 = await driver.execute_query(mentions_query)
    edges_after_second = mentions_result2[0]
    edge_uuids_second = {m["r.uuid"] for m in edges_after_second}
    entity_names_second = {m["e.name"] for m in edges_after_second}

    assert len(edges_after_second) == len(edges_after_first), (
        "Re-extracting same content should produce same number of edges"
    )

    assert entity_names_first == entity_names_second, (
        "Re-extracting same content should mention same entities"
    )

    with redis_ops() as r:
        cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
        cached_edges_json = r.hget(cache_key, "mentions_edges")
        cached_uuids_second = set(json.loads(cached_edges_json.decode()))

    assert len(cached_uuids_second) == len(cached_uuids_first), (
        "Cache should have same number of edge UUIDs after re-extraction"
    )


@pytest.mark.inference
@pytest.mark.asyncio
async def test_task_integration_with_status_transitions(isolated_graph, require_llm):
    """Test that extract_nodes_task properly manages episode status during re-extraction.

    Verifies:
    - Status transitions work correctly during re-extraction
    - Task can be called multiple times on same episode
    - Status is properly updated after each extraction
    """
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend.graph.extract_nodes import extract_nodes
    from backend.database import get_driver
    from backend.database.lifecycle import _ensure_graph

    content_v1 = "I met Kevin at the gym."
    episode_uuid = await add_journal_entry(content_v1)

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    status_before = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status_before == "pending_nodes"

    result1 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    if result1.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result1.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    status_after_first = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status_after_first in ["pending_edges", "done"], (
        f"Status should transition after extraction, got {status_after_first}"
    )

    driver = get_driver(DEFAULT_JOURNAL)
    from graphiti_core.nodes import EpisodicNode
    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    episode.content = "I met Kevin and Anna at the gym."

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

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    status_before_reextract = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status_before_reextract == "pending_nodes"

    result2 = await extract_nodes(episode_uuid, DEFAULT_JOURNAL)

    if result2.extracted_count > 0:
        set_episode_status(episode_uuid, "pending_edges", DEFAULT_JOURNAL, uuid_map=result2.uuid_map)
    else:
        set_episode_status(episode_uuid, "done", DEFAULT_JOURNAL)

    status_after_reextract = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status_after_reextract in ["pending_edges", "done"], (
        f"Status should transition after re-extraction, got {status_after_reextract}"
    )

    assert result2.extracted_count is not None
    assert result2.extracted_count > 0
