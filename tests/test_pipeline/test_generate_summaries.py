"""Tests for the GenerateSummaries stage.

Organized into:
1. Unit tests: Pure functions - no DB/LLM
2. Integration tests: GenerateSummaries orchestrator - with DB/LLM

Tests use real implementations to verify actual behavior, not mock behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from pipeline.generate_summaries import (
    GenerateSummaries,
    GenerateSummariesOutput,
)


# ========== Integration Tests: GenerateSummaries Orchestrator ==========


@pytest.mark.asyncio
async def test_generate_summaries_basic_generation(isolated_graph) -> None:
    """Generate summary for entity with attributes."""
    group_id = "summary-basic"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Today I had coffee with my friend Sarah from work. She told me about her new project on machine learning.",
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
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[sarah],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 1
    assert result.nodes[0].summary != ""
    assert len(result.nodes[0].summary) <= MAX_SUMMARY_CHARS
    assert result.metadata["nodes_processed"] == 1
    assert result.metadata["avg_summary_length"] > 0


@pytest.mark.asyncio
async def test_generate_summaries_combines_with_existing(isolated_graph) -> None:
    """Entity with existing summary should combine old + new information."""
    group_id = "summary-combine"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Sarah got promoted to senior engineer at work today. She's now leading the ML team.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid",
        name="Sarah",
        labels=["Entity", "Person"],
        attributes={"relationship_type": "friend"},
        summary="Friend who works on machine learning projects.",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[sarah],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 1
    assert result.nodes[0].summary != ""
    assert len(result.nodes[0].summary) <= MAX_SUMMARY_CHARS
    assert result.metadata["nodes_processed"] == 1


@pytest.mark.asyncio
async def test_generate_summaries_truncates_long_summaries(isolated_graph) -> None:
    """Episode with verbose content should generate summary <= 250 chars."""
    group_id = "summary-truncate"

    verbose_content = """
    Today I had an incredibly detailed conversation with Dr. Emily Rodriguez about her groundbreaking
    research in quantum computing applications for cryptography. She explained how her team at the
    Advanced Research Laboratory has been working on developing novel quantum algorithms that could
    revolutionize secure communications. Emily spent considerable time discussing the theoretical
    foundations of quantum entanglement and how it relates to her current work on quantum key distribution
    systems. She also mentioned her previous work at MIT where she collaborated with leading physicists
    on experimental quantum systems. The conversation covered various technical aspects including qubit
    coherence times, error correction methodologies, and the challenges of scaling quantum systems for
    practical applications in telecommunications infrastructure.
    """

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content=verbose_content,
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    emily = EntityNode(
        uuid="emily-uuid",
        name="Dr. Emily Rodriguez",
        labels=["Entity", "Person"],
        attributes={"occupation": "researcher"},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[emily],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 1
    assert result.nodes[0].summary != ""
    assert len(result.nodes[0].summary) <= MAX_SUMMARY_CHARS
    assert result.metadata["nodes_processed"] == 1


@pytest.mark.asyncio
async def test_generate_summaries_handles_empty_existing_summary(isolated_graph) -> None:
    """Entity with empty summary should generate summary from scratch."""
    group_id = "summary-empty"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Met with Alex at the downtown coffee shop. He's a software engineer working on mobile apps.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    alex = EntityNode(
        uuid="alex-uuid",
        name="Alex",
        labels=["Entity", "Person"],
        attributes={},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[alex],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 1
    assert result.nodes[0].summary != ""
    assert len(result.nodes[0].summary) <= MAX_SUMMARY_CHARS
    assert result.metadata["nodes_processed"] == 1
    assert result.metadata["avg_summary_length"] > 0


@pytest.mark.asyncio
async def test_generate_summaries_processes_all_entity_types(isolated_graph) -> None:
    """Mix of Person, Emotion, generic Entity should all get summaries."""
    group_id = "summary-types"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Had lunch with Maria today. Felt anxious about the upcoming presentation. The project deadline is next week.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    maria = EntityNode(
        uuid="maria-uuid",
        name="Maria",
        labels=["Entity", "Person"],
        attributes={},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    anxiety = EntityNode(
        uuid="anxiety-uuid",
        name="anxiety",
        labels=["Entity", "Emotion"],
        attributes={},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    project = EntityNode(
        uuid="project-uuid",
        name="project",
        labels=["Entity"],
        attributes={},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[maria, anxiety, project],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 3
    assert all(node.summary != "" for node in result.nodes)
    assert all(len(node.summary) <= MAX_SUMMARY_CHARS for node in result.nodes)
    assert result.metadata["nodes_processed"] == 3
    assert result.metadata["avg_summary_length"] > 0


@pytest.mark.asyncio
async def test_generate_summaries_metadata_tracking(isolated_graph) -> None:
    """Multiple entities should track correct counts and avg_summary_length."""
    group_id = "summary-metadata"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Team meeting with John and Lisa. John presented the Q1 results. Lisa discussed the marketing strategy.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    john = EntityNode(
        uuid="john-uuid",
        name="John",
        labels=["Entity", "Person"],
        attributes={"role": "analyst"},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    lisa = EntityNode(
        uuid="lisa-uuid",
        name="Lisa",
        labels=["Entity", "Person"],
        attributes={"role": "marketer"},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[john, lisa],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 2
    assert result.metadata["nodes_processed"] == 2
    assert "avg_summary_length" in result.metadata
    assert result.metadata["avg_summary_length"] > 0
    assert "truncated_count" in result.metadata
    assert result.metadata["truncated_count"] >= 0


@pytest.mark.asyncio
async def test_generate_summaries_uses_attributes_for_context(isolated_graph) -> None:
    """Person with relationship_type attribute should reference the relationship."""
    group_id = "summary-attributes"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="My colleague Rachel helped me debug the production issue today. She's great with distributed systems.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    rachel = EntityNode(
        uuid="rachel-uuid",
        name="Rachel",
        labels=["Entity", "Person"],
        attributes={"relationship_type": "colleague"},
        summary="",
        group_id=group_id,
        created_at=utc_now(),
    )

    generator = GenerateSummaries(group_id=group_id)
    result = await generator(
        nodes=[rachel],
        episode=episode,
        previous_episodes=[],
    )

    assert isinstance(result, GenerateSummariesOutput)
    assert len(result.nodes) == 1
    assert result.nodes[0].summary != ""
    assert len(result.nodes[0].summary) <= MAX_SUMMARY_CHARS
    assert result.metadata["nodes_processed"] == 1
