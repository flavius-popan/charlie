"""Integration tests for the add_journal orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from graphiti_core.errors import GroupIdValidationError
from pipeline import AddJournalResults, add_journal
from pipeline import db_utils


@pytest.mark.asyncio
async def test_add_journal_runs_end_to_end(isolated_graph) -> None:
    """Exercise the full add_journal orchestrator with real dependencies."""
    group_id = "integration-user"
    result = await add_journal(
        content=(
            "Met with Dr. Sarah Chen and COO Mark Patel at Stanford HQ to discuss "
            "the Q2 roadmap and the Apollo initiative."
        ),
        group_id=group_id,
        reference_time=datetime(2024, 2, 1, 9, 30, tzinfo=timezone.utc),
        name="integration-entry",
        source_description="Integration test journal",
    )

    assert isinstance(result, AddJournalResults)
    assert result.episode.group_id == group_id
    assert result.episode.name == "integration-entry"
    assert all(node.group_id == group_id for node in result.nodes)
    assert result.metadata["resolved_count"] <= result.metadata["extracted_count"]
    assert set(result.uuid_map.values()) >= {node.uuid for node in result.nodes}
    assert len(result.episodic_edges) == len(result.nodes)
    assert result.metadata["persistence"]["episode_uuid"] == result.episode.uuid


@pytest.mark.asyncio
async def test_add_journal_resolves_existing_entities(
    isolated_graph, seed_entity
) -> None:
    """Existing graph entities should be reused instead of duplicated."""
    group_id = "dedupe-user"
    existing_uuid = "00000000-0000-4000-8000-000000000001"
    seed_entity(uuid=existing_uuid, name="Dr. Sarah Chen", group_id=group_id)

    result = await add_journal(
        content=(
            "Quick sync with Dr. Sarah Chen on the Apollo rollout. "
            "Confirmed Sarah Chen will present the roadmap."
        ),
        group_id=group_id,
        reference_time=datetime(2024, 3, 12, 15, 0, tzinfo=timezone.utc),
    )

    assert result.metadata["resolved_count"] >= 1
    assert existing_uuid in result.uuid_map.values()


@pytest.mark.asyncio
async def test_add_journal_defaults_group_id(isolated_graph) -> None:
    """Ensure default group_id flows through to episode and node artifacts."""
    result = await add_journal(content="Documented daily standup notes.")

    assert result.episode.group_id == "\\_"
    assert all(node.group_id == "\\_" for node in result.nodes)


@pytest.mark.asyncio
async def test_add_journal_validates_group_id() -> None:
    """Invalid group identifiers should be rejected before any heavy work."""
    with pytest.raises(GroupIdValidationError):
        await add_journal(
            content="Bad input",
            group_id="invalid group!",
        )


@pytest.mark.asyncio
async def test_add_journal_extracts_edges(isolated_graph) -> None:
    """Edges should be extracted between entities in the journal content."""
    group_id = "edge-extraction"
    result = await add_journal(
        content=(
            "Sarah works at Stanford University. "
            "Mark is the director of the AI lab at Stanford."
        ),
        group_id=group_id,
        reference_time=datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc),
    )

    assert isinstance(result, AddJournalResults)
    assert len(result.edges) >= 1
    assert all(e.group_id == group_id for e in result.edges)
    assert "edges" in result.metadata
    assert result.metadata["edges"]["extracted_count"] >= 1


@pytest.mark.asyncio
async def test_add_journal_persists_entities_and_edges(isolated_graph) -> None:
    """Persisted entities, edges, and episodes should be queryable from FalkorDB."""
    group_id = "persist-user"
    reference_time = datetime(2024, 4, 5, 12, 0, tzinfo=timezone.utc)
    result = await add_journal(
        content="Debriefed with Taylor on the Apollo rollout; Taylor mentors me weekly.",
        group_id=group_id,
        reference_time=reference_time,
    )

    entities = await db_utils.fetch_entities_by_group(group_id)
    assert {node.uuid for node in result.nodes}.issubset(set(entities))

    edges = await db_utils.fetch_entity_edges_by_group(group_id)
    assert {edge.uuid for edge in result.edges}.issubset(set(edges))

    episodes = await db_utils.fetch_recent_episodes(
        group_id,
        reference_time,
        limit=5,
    )
    assert any(ep.uuid == result.episode.uuid for ep in episodes)
    assert result.metadata["persistence"]["status"] == "persisted"


@pytest.mark.asyncio
async def test_add_journal_can_skip_persistence(isolated_graph) -> None:
    """Persistence can be disabled for dry runs."""
    stats_before = await db_utils.get_db_stats()
    assert stats_before == {"episodes": 0, "entities": 0}

    result = await add_journal(
        content="Sketched timeline for next sprint.",
        group_id="skip-user",
        reference_time=datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc),
        persist=False,
    )

    stats_after = await db_utils.get_db_stats()
    assert stats_after == stats_before
    assert result.metadata["persistence"] == {"status": "skipped"}
