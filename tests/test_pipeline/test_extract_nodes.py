"""Integration tests for the ExtractNodes stage.

Demonstrates the two-layer architecture:
- EntityExtractor (dspy.Module): Pure LLM extraction, optimizable
- ExtractNodes (orchestrator): Full pipeline with DB, resolution, metadata

Tests use ExtractNodes directly (the orchestrator). To test optimization,
compile EntityExtractor separately and inject into ExtractNodes.__init__().
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.extract_nodes import ExtractNodes, ExtractNodesOutput
from pipeline.self_reference import SELF_ENTITY_NAME


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
async def test_extract_nodes_resolves_against_existing_entities(
    isolated_graph, seed_entity
) -> None:
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


@pytest.mark.asyncio
async def test_extract_nodes_injects_self_on_pronoun(isolated_graph) -> None:
    """First-person journals should always include the deterministic SELF entity."""
    group_id = "self-inject"
    extractor = ExtractNodes(group_id=group_id)
    result = await extractor(
        content="I visited the memorial garden after work.",
        reference_time=datetime(2024, 4, 2, 12, 30, tzinfo=timezone.utc),
    )

    assert result.metadata["first_person_detected"] is True
    assert any(node.name == SELF_ENTITY_NAME for node in result.nodes)


@pytest.mark.asyncio
async def test_extract_nodes_omits_self_when_not_referenced(isolated_graph) -> None:
    """Entries without first-person pronouns should not include the author node."""
    group_id = "self-omit"
    extractor = ExtractNodes(group_id=group_id)
    result = await extractor(
        content="Dana and Lee reviewed the rollout plan together.",
        reference_time=datetime(2024, 4, 3, 9, 0, tzinfo=timezone.utc),
    )

    assert result.metadata["first_person_detected"] is False
    assert all(node.name != SELF_ENTITY_NAME for node in result.nodes)
