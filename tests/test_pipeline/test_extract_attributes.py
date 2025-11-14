"""Tests for the ExtractAttributes stage.

Organized into:
1. Unit tests: Pure functions - no DB/LLM
2. Integration tests: ExtractAttributes orchestrator - with DB/LLM

Tests use real implementations to verify actual behavior, not mock behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import utc_now

from pipeline.extract_attributes import (
    ExtractAttributes,
    ExtractAttributesOutput,
)


# ========== Integration Tests: ExtractAttributes Orchestrator ==========


@pytest.mark.asyncio
async def test_extract_attributes_basic_extraction(isolated_graph) -> None:
    """Extract Person attributes from episode content with real LLM call."""
    group_id = "attr-basic"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Today I had coffee with my friend Sarah.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid",
        name="Sarah",
        labels=["Entity", "Person"],
        attributes={},
        group_id=group_id,
        created_at=utc_now(),
    )

    from pipeline.entity_edge_models import entity_types

    extractor = ExtractAttributes(group_id=group_id)
    result = await extractor(
        nodes=[sarah],
        episode=episode,
        previous_episodes=[],
        entity_types=entity_types,
    )

    assert isinstance(result, ExtractAttributesOutput)
    assert len(result.nodes) == 1
    assert "relationship_type" in result.nodes[0].attributes
    assert result.nodes[0].attributes["relationship_type"] == "friend"
    assert result.metadata["nodes_processed"] == 1
    assert result.metadata["nodes_skipped"] == 0


@pytest.mark.asyncio
async def test_extract_attributes_skips_entity_without_type(isolated_graph) -> None:
    """Entity with only generic 'Entity' label should be skipped."""
    group_id = "attr-no-type"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Some generic content.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    generic_entity = EntityNode(
        uuid="generic-uuid",
        name="Something",
        labels=["Entity"],
        attributes={},
        group_id=group_id,
        created_at=utc_now(),
    )

    from pipeline.entity_edge_models import entity_types

    extractor = ExtractAttributes(group_id=group_id)
    result = await extractor(
        nodes=[generic_entity],
        episode=episode,
        previous_episodes=[],
        entity_types=entity_types,
    )

    assert isinstance(result, ExtractAttributesOutput)
    assert result.metadata["nodes_processed"] == 0
    assert result.metadata["nodes_skipped"] == 1


@pytest.mark.asyncio
async def test_extract_attributes_skips_entity_with_no_fields(isolated_graph) -> None:
    """Custom entity type with no Pydantic fields should be skipped."""
    from pydantic import BaseModel

    class EmptyEntity(BaseModel):
        """Entity with no attributes defined."""

        pass

    custom_entity_types = {"EmptyEntity": EmptyEntity}

    group_id = "attr-no-fields"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Some content about empty entity.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    empty_entity = EntityNode(
        uuid="empty-uuid",
        name="EmptyThing",
        labels=["Entity", "EmptyEntity"],
        attributes={},
        group_id=group_id,
        created_at=utc_now(),
    )

    extractor = ExtractAttributes(group_id=group_id)
    result = await extractor(
        nodes=[empty_entity],
        episode=episode,
        previous_episodes=[],
        entity_types=custom_entity_types,
    )

    assert isinstance(result, ExtractAttributesOutput)
    assert result.metadata["nodes_processed"] == 0
    assert result.metadata["nodes_skipped"] == 1


@pytest.mark.asyncio
async def test_extract_attributes_preserves_existing_attributes(isolated_graph) -> None:
    """Entity with existing attributes should preserve them and add new ones."""
    group_id = "attr-preserve"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Today I had coffee with my colleague Sarah at the office.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid",
        name="Sarah",
        labels=["Entity", "Person"],
        attributes={"relationship_type": "friend"},
        group_id=group_id,
        created_at=utc_now(),
    )

    from pipeline.entity_edge_models import entity_types

    extractor = ExtractAttributes(group_id=group_id)
    result = await extractor(
        nodes=[sarah],
        episode=episode,
        previous_episodes=[],
        entity_types=entity_types,
    )

    assert isinstance(result, ExtractAttributesOutput)
    assert len(result.nodes) == 1
    assert "relationship_type" in result.nodes[0].attributes
    assert result.metadata["nodes_processed"] == 1


@pytest.mark.asyncio
async def test_extract_attributes_metadata_tracking(isolated_graph) -> None:
    """Metadata should correctly track processing counts by entity type."""
    group_id = "attr-metadata"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Had coffee with my friend Sarah at the local cafe. Also met with a client.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid",
        name="Sarah",
        labels=["Entity", "Person"],
        attributes={},
        group_id=group_id,
        created_at=utc_now(),
    )

    generic_entity = EntityNode(
        uuid="client-uuid",
        name="client",
        labels=["Entity"],
        attributes={},
        group_id=group_id,
        created_at=utc_now(),
    )

    from pipeline.entity_edge_models import entity_types

    extractor = ExtractAttributes(group_id=group_id)
    result = await extractor(
        nodes=[sarah, generic_entity],
        episode=episode,
        previous_episodes=[],
        entity_types=entity_types,
    )

    assert isinstance(result, ExtractAttributesOutput)
    assert result.metadata["nodes_processed"] == 1
    assert result.metadata["nodes_skipped"] == 1
    assert "attributes_extracted_by_type" in result.metadata
    assert result.metadata["attributes_extracted_by_type"]["Person"] == 1
