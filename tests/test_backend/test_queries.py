"""Tests for database query functions."""

import pytest
from backend.database.queries import fetch_entities_for_episode, delete_entity_mention
from backend.database.utils import SELF_ENTITY_UUID
from backend import add_journal_entry
from backend.graph.extract_nodes import extract_nodes


@pytest.mark.asyncio
async def test_fetch_entities_for_episode_returns_entities(isolated_graph, require_llm):
    """Should return all entities mentioned in episode, excluding SELF."""
    # Create episode that will extract entities
    episode_uuid = await add_journal_entry(
        content="I met Sarah at Central Park for coffee.",
        journal="test_journal"
    )

    # Extract entities (this creates MENTIONS edges)
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    # Fetch entities
    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")

    # Should have entities but not SELF
    assert len(entities) > 0
    entity_names = [e["name"] for e in entities]
    assert "Sarah" in entity_names or "Central Park" in entity_names

    # SELF should be filtered out
    entity_uuids = [e["uuid"] for e in entities]
    assert str(SELF_ENTITY_UUID) not in entity_uuids


@pytest.mark.asyncio
async def test_fetch_entities_for_episode_empty(isolated_graph):
    """Should return empty list when episode has no entity mentions."""
    episode_uuid = await add_journal_entry(
        content="Just some text.",
        journal="test_journal"
    )

    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")

    assert entities == []


@pytest.mark.asyncio
async def test_delete_entity_mention_orphaned(isolated_graph, require_llm):
    """Should delete entity when it's only mentioned in this episode."""
    # Create episode with entity
    episode_uuid = await add_journal_entry(
        content="I met Sarah at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    # Get entity UUID
    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")
    assert len(entities) > 0
    entity_uuid = entities[0]["uuid"]

    # Delete mention
    was_deleted = await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    # Entity should be fully deleted (was orphaned)
    assert was_deleted is True

    # Verify entity no longer appears
    entities_after = await fetch_entities_for_episode(episode_uuid, "test_journal")
    assert entity_uuid not in [e["uuid"] for e in entities_after]


@pytest.mark.asyncio
async def test_delete_entity_mention_shared(isolated_graph, require_llm):
    """Should only remove MENTIONS edge when entity referenced elsewhere."""
    # Create two episodes mentioning same entity
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

    # Get Sarah's UUID from first episode
    entities_ep1 = await fetch_entities_for_episode(ep1_uuid, "test_journal")
    sarah = next((e for e in entities_ep1 if "Sarah" in e["name"]), None)
    assert sarah is not None
    assert sarah["ref_count"] >= 2  # Mentioned in at least 2 episodes

    # Delete from first episode only
    was_deleted = await delete_entity_mention(ep1_uuid, sarah["uuid"], "test_journal")

    # Entity should NOT be fully deleted (still referenced)
    assert was_deleted is False

    # Should not appear in ep1 anymore
    entities_ep1_after = await fetch_entities_for_episode(ep1_uuid, "test_journal")
    assert sarah["uuid"] not in [e["uuid"] for e in entities_ep1_after]

    # Should still appear in ep2
    entities_ep2 = await fetch_entities_for_episode(ep2_uuid, "test_journal")
    assert sarah["uuid"] in [e["uuid"] for e in entities_ep2]
