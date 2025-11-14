"""Tests for the ExtractEdges stage.

Organized into:
1. Unit tests: Pure functions (normalize, parse, build) - no DB/LLM
2. Integration tests: ExtractEdges orchestrator - with DB/LLM

Tests use real implementations to verify actual behavior, not mock behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import utc_now

from pipeline.extract_edges import (
    ExtractEdges,
    ExtractEdgesOutput,
    ExtractedRelationship,
    ExtractedRelationships,
    _parse_temporal_field,
    build_entity_edges,
    normalize_edge_name,
)


# ========== Unit Tests: Pure Functions ==========


class TestNormalizeEdgeName:
    """Test edge name normalization to SCREAMING_SNAKE_CASE."""

    def test_lowercase_single_word(self) -> None:
        assert normalize_edge_name("works") == "WORKS"

    def test_lowercase_multiple_words(self) -> None:
        assert normalize_edge_name("works at") == "WORKS_AT"

    def test_mixed_case(self) -> None:
        assert normalize_edge_name("Friend Of") == "FRIEND_OF"

    def test_already_screaming_snake_case(self) -> None:
        assert normalize_edge_name("KNOWS") == "KNOWS"

    def test_extra_whitespace(self) -> None:
        assert normalize_edge_name("  works   at  ") == "WORKS_AT"

    def test_underscores_preserved(self) -> None:
        assert normalize_edge_name("works_at") == "WORKS_AT"


class TestParseTemporalField:
    """Test ISO 8601 temporal field parsing."""

    def test_valid_iso_string_with_z(self) -> None:
        result = _parse_temporal_field("2024-01-15T10:30:00Z", "valid_at")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_valid_iso_string_with_offset(self) -> None:
        result = _parse_temporal_field("2024-01-15T10:30:00+00:00", "valid_at")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_none_input(self) -> None:
        result = _parse_temporal_field(None, "valid_at")
        assert result is None

    def test_invalid_format(self) -> None:
        result = _parse_temporal_field("not-a-date", "valid_at")
        assert result is None

    def test_empty_string(self) -> None:
        result = _parse_temporal_field("", "valid_at")
        assert result is None


class TestBuildEntityEdges:
    """Test EntityEdge construction from extracted relationships."""

    def test_builds_basic_edge(self) -> None:
        sarah = EntityNode(
            uuid="sarah-uuid",
            name="Sarah",
            group_id="test",
            created_at=utc_now(),
        )
        stanford = EntityNode(
            uuid="stanford-uuid",
            name="Stanford",
            group_id="test",
            created_at=utc_now(),
        )
        entity_map = {"sarah": sarah, "stanford": stanford}

        relationships = ExtractedRelationships(
            relationships=[
                ExtractedRelationship(
                    source="Sarah",
                    target="Stanford",
                    relation="works at",
                    fact="Sarah works at Stanford as a researcher.",
                )
            ]
        )

        edges = build_entity_edges(
            relationships,
            entity_map,
            episode_uuid="episode-123",
            group_id="test",
        )

        assert len(edges) == 1
        assert edges[0].source_node_uuid == "sarah-uuid"
        assert edges[0].target_node_uuid == "stanford-uuid"
        assert edges[0].name == "WORKS_AT"
        assert edges[0].fact == "Sarah works at Stanford as a researcher."
        assert edges[0].episodes == ["episode-123"]
        assert edges[0].group_id == "test"

    def test_skips_edge_with_missing_source(self) -> None:
        target = EntityNode(
            uuid="target-uuid",
            name="Target",
            group_id="test",
            created_at=utc_now(),
        )
        entity_map = {"target": target}

        relationships = ExtractedRelationships(
            relationships=[
                ExtractedRelationship(
                    source="MissingEntity",
                    target="Target",
                    relation="RELATES_TO",
                    fact="Some fact",
                )
            ]
        )

        edges = build_entity_edges(
            relationships, entity_map, "episode-123", "test"
        )

        assert len(edges) == 0

    def test_skips_edge_with_missing_target(self) -> None:
        source = EntityNode(
            uuid="source-uuid",
            name="Source",
            group_id="test",
            created_at=utc_now(),
        )
        entity_map = {"source": source}

        relationships = ExtractedRelationships(
            relationships=[
                ExtractedRelationship(
                    source="Source",
                    target="MissingEntity",
                    relation="RELATES_TO",
                    fact="Some fact",
                )
            ]
        )

        edges = build_entity_edges(
            relationships, entity_map, "episode-123", "test"
        )

        assert len(edges) == 0

    def test_parses_temporal_metadata(self) -> None:
        source = EntityNode(
            uuid="source-uuid", name="Source", group_id="test", created_at=utc_now()
        )
        target = EntityNode(
            uuid="target-uuid", name="Target", group_id="test", created_at=utc_now()
        )
        entity_map = {"source": source, "target": target}

        relationships = ExtractedRelationships(
            relationships=[
                ExtractedRelationship(
                    source="Source",
                    target="Target",
                    relation="EMPLOYED_AT",
                    fact="Source worked at Target from 2020 to 2023.",
                    valid_at="2020-01-01T00:00:00Z",
                    invalid_at="2023-12-31T23:59:59Z",
                )
            ]
        )

        edges = build_entity_edges(
            relationships, entity_map, "episode-123", "test"
        )

        assert len(edges) == 1
        assert edges[0].valid_at is not None
        assert edges[0].valid_at.year == 2020
        assert edges[0].invalid_at is not None
        assert edges[0].invalid_at.year == 2023

    def test_skips_edge_with_invalid_temporal_range(self) -> None:
        source = EntityNode(
            uuid="source-uuid", name="Source", group_id="test", created_at=utc_now()
        )
        target = EntityNode(
            uuid="target-uuid", name="Target", group_id="test", created_at=utc_now()
        )
        entity_map = {"source": source, "target": target}

        relationships = ExtractedRelationships(
            relationships=[
                ExtractedRelationship(
                    source="Source",
                    target="Target",
                    relation="EMPLOYED_AT",
                    fact="Invalid temporal range",
                    valid_at="2023-01-01T00:00:00Z",
                    invalid_at="2020-01-01T00:00:00Z",  # invalid_at before valid_at
                )
            ]
        )

        edges = build_entity_edges(
            relationships, entity_map, "episode-123", "test"
        )

        assert len(edges) == 0


# ========== Integration Tests: ExtractEdges Orchestrator ==========


@pytest.fixture
def seed_edge(isolated_graph) -> callable:
    """Helper to insert EntityEdge relationships for deduplication tests."""
    import pipeline.falkordblite_driver as db_utils

    def _seed(
        *,
        uuid: str,
        source_uuid: str,
        target_uuid: str,
        name: str,
        fact: str,
        group_id: str,
        episodes: list[str],
    ) -> None:
        import json

        query = f"""
        MATCH (source:Entity {{uuid: {db_utils.to_cypher_literal(source_uuid)}}})
        MATCH (target:Entity {{uuid: {db_utils.to_cypher_literal(target_uuid)}}})
        CREATE (source)-[:RELATES {{
            uuid: {db_utils.to_cypher_literal(uuid)},
            name: {db_utils.to_cypher_literal(name)},
            fact: {db_utils.to_cypher_literal(fact)},
            group_id: {db_utils.to_cypher_literal(group_id)},
            created_at: {db_utils.to_cypher_literal('2024-01-01T00:00:00Z')},
            episodes: {db_utils.to_cypher_literal(json.dumps(episodes))}
        }}]->(target)
        """
        isolated_graph.query(query)

    return _seed


@pytest.mark.asyncio
async def test_extract_edges_basic_extraction(isolated_graph) -> None:
    """Extract edges from episode content with real LLM call."""
    group_id = "edge-basic"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Sarah works at Stanford. Mark is friends with Sarah.",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid",
        name="Sarah",
        group_id=group_id,
        created_at=utc_now(),
    )
    stanford = EntityNode(
        uuid="stanford-uuid",
        name="Stanford",
        group_id=group_id,
        created_at=utc_now(),
    )
    mark = EntityNode(
        uuid="mark-uuid",
        name="Mark",
        group_id=group_id,
        created_at=utc_now(),
    )

    extracted_nodes = [sarah, stanford, mark]
    resolved_nodes = extracted_nodes
    uuid_map = {node.uuid: node.uuid for node in extracted_nodes}

    extractor = ExtractEdges(group_id=group_id, dedupe_enabled=False)
    result = await extractor(
        episode=episode,
        extracted_nodes=extracted_nodes,
        resolved_nodes=resolved_nodes,
        uuid_map=uuid_map,
        previous_episodes=[],
    )

    assert isinstance(result, ExtractEdgesOutput)
    assert len(result.edges) >= 1
    assert all(isinstance(e, EntityEdge) for e in result.edges)
    assert all(e.group_id == group_id for e in result.edges)
    assert result.metadata["extracted_count"] >= 1


@pytest.mark.asyncio
async def test_extract_edges_with_uuid_remapping(isolated_graph, seed_entity) -> None:
    """Edge pointers should be remapped from provisional to canonical UUIDs."""
    group_id = "edge-remap"

    seed_entity(uuid="canonical-sarah", name="Sarah Chen", group_id=group_id)

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Sarah works at Stanford",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    provisional_sarah = EntityNode(
        uuid="provisional-sarah",
        name="Sarah",
        group_id=group_id,
        created_at=utc_now(),
    )
    stanford = EntityNode(
        uuid="stanford-uuid",
        name="Stanford",
        group_id=group_id,
        created_at=utc_now(),
    )

    extracted_nodes = [provisional_sarah, stanford]
    resolved_nodes = [
        EntityNode(
            uuid="canonical-sarah",
            name="Sarah Chen",
            group_id=group_id,
            created_at=utc_now(),
        ),
        stanford,
    ]

    uuid_map = {
        "provisional-sarah": "canonical-sarah",
        "stanford-uuid": "stanford-uuid",
    }

    extractor = ExtractEdges(group_id=group_id, dedupe_enabled=False)
    result = await extractor(
        episode=episode,
        extracted_nodes=extracted_nodes,
        resolved_nodes=resolved_nodes,
        uuid_map=uuid_map,
        previous_episodes=[],
    )

    for edge in result.edges:
        if edge.target_node_uuid == "stanford-uuid":
            assert edge.source_node_uuid == "canonical-sarah"


@pytest.mark.asyncio
async def test_extract_edges_deduplication(
    isolated_graph, seed_entity, seed_edge
) -> None:
    """Existing edges should be merged with new episode IDs, not duplicated."""
    group_id = "edge-dedupe"

    seed_entity(uuid="sarah-uuid", name="Sarah", group_id=group_id)
    seed_entity(uuid="stanford-uuid", name="Stanford", group_id=group_id)
    seed_edge(
        uuid="existing-edge",
        source_uuid="sarah-uuid",
        target_uuid="stanford-uuid",
        name="WORKS_AT",
        fact="Sarah works at Stanford",
        group_id=group_id,
        episodes=["episode-1"],
    )

    episode = EpisodicNode(
        uuid="episode-2",
        name="test-episode-2",
        labels=[],
        source=EpisodeType.text,
        content="Sarah works at Stanford University as a professor",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    sarah = EntityNode(
        uuid="sarah-uuid", name="Sarah", group_id=group_id, created_at=utc_now()
    )
    stanford = EntityNode(
        uuid="stanford-uuid", name="Stanford", group_id=group_id, created_at=utc_now()
    )

    extracted_nodes = [sarah, stanford]
    resolved_nodes = extracted_nodes
    uuid_map = {node.uuid: node.uuid for node in extracted_nodes}

    extractor = ExtractEdges(group_id=group_id, dedupe_enabled=True)
    result = await extractor(
        episode=episode,
        extracted_nodes=extracted_nodes,
        resolved_nodes=resolved_nodes,
        uuid_map=uuid_map,
        previous_episodes=[],
    )

    works_at_edges = [
        e
        for e in result.edges
        if e.source_node_uuid == "sarah-uuid"
        and e.target_node_uuid == "stanford-uuid"
    ]

    assert len(works_at_edges) >= 1, "Expected at least one edge between Sarah and Stanford"

    if result.metadata["merged_count"] >= 1:
        edge = works_at_edges[0]
        assert "episode-1" in edge.episodes
        assert "episode-2" in edge.episodes


@pytest.mark.asyncio
async def test_extract_edges_with_dedupe_disabled(isolated_graph) -> None:
    """Deduplication can be disabled for scenarios that need raw extraction."""
    group_id = "edge-no-dedupe"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Alice knows Bob",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    alice = EntityNode(
        uuid="alice-uuid", name="Alice", group_id=group_id, created_at=utc_now()
    )
    bob = EntityNode(
        uuid="bob-uuid", name="Bob", group_id=group_id, created_at=utc_now()
    )

    extracted_nodes = [alice, bob]
    resolved_nodes = extracted_nodes
    uuid_map = {node.uuid: node.uuid for node in extracted_nodes}

    extractor = ExtractEdges(group_id=group_id, dedupe_enabled=False)
    result = await extractor(
        episode=episode,
        extracted_nodes=extracted_nodes,
        resolved_nodes=resolved_nodes,
        uuid_map=uuid_map,
        previous_episodes=[],
    )

    assert result.metadata["merged_count"] == 0
    assert result.metadata["new_count"] == result.metadata["resolved_count"]


@pytest.mark.asyncio
async def test_extract_edges_metadata_counts(isolated_graph) -> None:
    """Metadata should track extraction, building, and resolution statistics."""
    group_id = "edge-metadata"

    episode = EpisodicNode(
        uuid=str(uuid4()),
        name="test-episode",
        labels=[],
        source=EpisodeType.text,
        content="Project Redwood involves teams from Engineering and Design",
        source_description="Test",
        group_id=group_id,
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        created_at=utc_now(),
    )

    redwood = EntityNode(
        uuid="redwood-uuid",
        name="Project Redwood",
        group_id=group_id,
        created_at=utc_now(),
    )
    engineering = EntityNode(
        uuid="eng-uuid", name="Engineering", group_id=group_id, created_at=utc_now()
    )
    design = EntityNode(
        uuid="design-uuid", name="Design", group_id=group_id, created_at=utc_now()
    )

    extracted_nodes = [redwood, engineering, design]
    resolved_nodes = extracted_nodes
    uuid_map = {node.uuid: node.uuid for node in extracted_nodes}

    extractor = ExtractEdges(group_id=group_id, dedupe_enabled=False)
    result = await extractor(
        episode=episode,
        extracted_nodes=extracted_nodes,
        resolved_nodes=resolved_nodes,
        uuid_map=uuid_map,
        previous_episodes=[],
    )

    assert "extracted_count" in result.metadata
    assert "built_count" in result.metadata
    assert "resolved_count" in result.metadata
    assert "new_count" in result.metadata
    assert "merged_count" in result.metadata

    assert result.metadata["built_count"] <= result.metadata["extracted_count"]
    assert result.metadata["resolved_count"] <= result.metadata["built_count"]
