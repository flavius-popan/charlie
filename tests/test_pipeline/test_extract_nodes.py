"""Integration tests for the ExtractNodes stage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.extract_nodes import ExtractNodes, ExtractNodesOutput


@pytest.mark.asyncio
async def test_extract_nodes_uses_recent_context(isolated_graph, seed_episode) -> None:
    """Ensure ExtractNodes executes against seeded episode context without errors."""
    group_id = "extract-stage"
    seed_episode(
        uuid="11111111-1111-4111-8111-111111111111",
        name="previous-episode",
        group_id=group_id,
        content="Reviewed Apollo launch tasks with Dana Li and Mark Patel.",
        valid_at=datetime(2024, 1, 20, 13, 0, tzinfo=timezone.utc),
    )

    extractor = ExtractNodes(group_id=group_id)
    result = await extractor(
        content=(
            "Met Dana Li to confirm the Redwood migration timeline. "
            "Dana will sync with COO Mark Patel next week."
        ),
        reference_time=datetime(2024, 2, 5, 9, 15, tzinfo=timezone.utc),
        source_description="Integration stage test",
    )

    assert isinstance(result, ExtractNodesOutput)
    assert result.episode.group_id == group_id
    assert result.episode.source_description == "Integration stage test"
    assert result.metadata["resolved_count"] <= result.metadata["extracted_count"]
    assert all(node.group_id == group_id for node in result.nodes)


@pytest.mark.asyncio
async def test_extract_nodes_resolves_against_existing_entities(isolated_graph, seed_entity) -> None:
    """Prior entities in the graph should be reused when names match."""
    group_id = "extract-dedupe"
    existing_uuid = "22222222-2222-4222-8222-222222222222"
    seed_entity(uuid=existing_uuid, name="Dana Li", group_id=group_id)

    extractor = ExtractNodes(group_id=group_id)
    result = await extractor(
        content="Catch-up with Dana Li about the onboarding kickoff.",
        reference_time=datetime(2024, 3, 10, 16, 45, tzinfo=timezone.utc),
    )

    assert result.metadata["resolved_count"] >= 1
    assert existing_uuid in result.uuid_map.values()
