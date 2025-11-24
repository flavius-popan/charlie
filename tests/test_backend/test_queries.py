"""Tests for database query functions."""

import pytest
from backend.database.queries import delete_entity_mention
from backend import add_journal_entry
from backend.graph.extract_nodes import extract_nodes
from backend.database.redis_ops import redis_ops, get_suppressed_entities
import json


@pytest.fixture(autouse=True)
def clear_test_journal_suppression():
    """Clear suppression list for test_journal before each test."""
    with redis_ops() as r:
        suppression_key = "journal:test_journal:suppressed_entities"
        r.delete(suppression_key)

    yield

    with redis_ops() as r:
        suppression_key = "journal:test_journal:suppressed_entities"
        r.delete(suppression_key)


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_orphaned(isolated_graph, require_llm):
    """Should delete entity when it's only mentioned in this episode."""
    episode_uuid = await add_journal_entry(
        content="I met Sarah at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        assert nodes_json is not None
        entities = json.loads(nodes_json.decode())
        assert len(entities) > 0
        entity_uuid = entities[0]["uuid"]

    entity_name = entities[0]["name"]

    was_deleted = await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    assert was_deleted is True

    suppressed = get_suppressed_entities("test_journal")
    assert entity_name.lower() in suppressed, "Deleted entity should be globally suppressed"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_shared(isolated_graph, require_llm):
    """Should only remove MENTIONS edge when entity referenced elsewhere."""
    ep1_uuid = await add_journal_entry(
        content="I met Sarah today.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep1_uuid, journal="test_journal")

    ep2_uuid = await add_journal_entry(
        content="Sarah came over again.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep2_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{ep1_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        entities = json.loads(nodes_json.decode())
        sarah = next((e for e in entities if "Sarah" in e["name"]), None)
        assert sarah is not None

    was_deleted = await delete_entity_mention(ep1_uuid, sarah["uuid"], "test_journal")

    assert was_deleted is False

    suppressed = get_suppressed_entities("test_journal")
    assert "sarah" in suppressed, "Deleted entity should be globally suppressed even if not orphaned"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_updates_redis_cache(isolated_graph, require_llm):
    """Test that deletion removes entity from Redis cache."""
    episode_uuid = await add_journal_entry(
        content="I met Sarah and John at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        entities = json.loads(nodes_json.decode())
        initial_count = len(entities)
        assert initial_count >= 2

        entity_to_delete = entities[0]
        entity_uuid = entity_to_delete["uuid"]
        entity_name = entity_to_delete["name"]

    await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        updated_entities = json.loads(nodes_json.decode())

        assert len(updated_entities) == initial_count - 1
        assert entity_uuid not in [e["uuid"] for e in updated_entities]
        assert entity_name not in [e["name"] for e in updated_entities]

    suppressed = get_suppressed_entities("test_journal")
    assert entity_name.lower() in suppressed, "Deleted entity should be globally suppressed"


@pytest.mark.asyncio
async def test_delete_entity_mention_sets_done_when_only_self_remains(isolated_graph, episode_uuid):
    """Deleting last visible entity should set status to done, even if 'I' remains."""
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import get_episode_status
    from backend.settings import DEFAULT_JOURNAL

    journal = DEFAULT_JOURNAL

    # Setup: Create episode node
    isolated_graph.query(f"""
        CREATE (:Episodic {{
            uuid: '{episode_uuid}',
            group_id: '{journal}',
            content: 'Test entry',
            name: 'Test',
            source: 'text',
            source_description: 'test',
            entity_edges: [],
            created_at: '2025-01-01T00:00:00Z',
            valid_at: '2025-01-01T00:00:00Z'
        }})
    """)

    # Create "I" entity and visible entity "Alice"
    isolated_graph.query(f"""
        CREATE (:Entity:Person {{uuid: 'self-uuid', group_id: '{journal}', name: 'I'}}),
               (:Entity:Person {{uuid: 'alice-uuid', group_id: '{journal}', name: 'Alice'}})
    """)

    # Create MENTIONS edges
    isolated_graph.query(f"""
        MATCH (ep:Episodic {{uuid: '{episode_uuid}'}}), (i:Entity {{uuid: 'self-uuid'}})
        CREATE (ep)-[:MENTIONS {{uuid: 'm1'}}]->(i)
    """)
    isolated_graph.query(f"""
        MATCH (ep:Episodic {{uuid: '{episode_uuid}'}}), (alice:Entity {{uuid: 'alice-uuid'}})
        CREATE (ep)-[:MENTIONS {{uuid: 'm2'}}]->(alice)
    """)

    # Setup Redis cache with both entities and pending_edges status
    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        r.hset(cache_key, "nodes", json.dumps([
            {"uuid": "self-uuid", "name": "I", "type": "Person"},
            {"uuid": "alice-uuid", "name": "Alice", "type": "Person"},
        ]))
        r.hset(cache_key, "mentions_edges", json.dumps(["m1", "m2"]))
        r.hset(cache_key, "status", "pending_edges")
        r.hset(cache_key, "journal", journal)

    # Act: Delete the visible entity (Alice)
    await delete_entity_mention(episode_uuid, "alice-uuid", journal)

    # Assert: Status should be "done" since only "I" remains
    status = get_episode_status(episode_uuid, journal)
    assert status == "done", f"Expected 'done' but got '{status}'"

    # Verify "I" is still in the nodes cache
    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        nodes = json.loads(nodes_json.decode())
        assert len(nodes) == 1
        assert nodes[0]["name"] == "I"
