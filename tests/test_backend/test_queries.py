"""Tests for database query functions."""

import pytest
from backend.database.queries import delete_entity_mention
from backend import add_journal_entry
from backend.graph.extract_nodes import extract_nodes
from backend.database.redis_ops import redis_ops
import json


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

    was_deleted = await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    assert was_deleted is True


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
