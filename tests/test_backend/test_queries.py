"""Tests for database query functions."""

import pytest
from backend.database.queries import fetch_entities_for_episode
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
