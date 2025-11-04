"""Tests for ExtractNodes pipeline stage.

Tests the Stage 1 implementation directly, not the end-to-end pipeline.
For orchestrator coverage, see test_add_journal.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pipeline.extract_nodes import (
    ExtractNodes,
    ExtractedEntity,
    ExtractedEntities,
    ExtractNodesOutput,
)
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now


class StubExtractor:
    """Stub DSPy extractor returning predetermined entities."""

    def __init__(self, entities: list[ExtractedEntity]):
        self.entities = entities
        self.calls: list[dict[str, str]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            extracted_entities=ExtractedEntities(extracted_entities=self.entities)
        )


class StubExtractNodes(ExtractNodes):
    """ExtractNodes with deterministic provisional entities."""

    def __init__(self, group_id: str, provisional_nodes: list[EntityNode]):
        super().__init__(group_id)
        self._provisional_nodes = provisional_nodes

    def _extract_entities(self, *_, **__) -> list[EntityNode]:
        return self._provisional_nodes


@pytest.mark.asyncio
async def test_extract_nodes_deduplicates_resolved_entities(monkeypatch: pytest.MonkeyPatch):
    """Ensure duplicate resolutions return a single canonical node."""

    existing = EntityNode(
        name="Dr. Sarah Chen",
        group_id="test_user",
        labels=["Entity"],
        summary="",
        created_at=utc_now(),
    )

    async def fake_fetch_entities(group_id: str):
        assert group_id == "test_user"
        return {existing.uuid: existing}

    async def fake_fetch_recent(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        "pipeline.extract_nodes.fetch_entities_by_group", fake_fetch_entities
    )
    monkeypatch.setattr(
        "pipeline.extract_nodes.fetch_recent_episodes", fake_fetch_recent
    )

    provisional = [
        EntityNode(
            name="Dr. Sarah Chen",
            group_id="test_user",
            labels=["Entity"],
            summary="",
            created_at=utc_now(),
        ),
        EntityNode(
            name="Dr. Sarah Chen",
            group_id="test_user",
            labels=["Entity"],
            summary="",
            created_at=utc_now(),
        ),
    ]

    extractor = StubExtractNodes(group_id="test_user", provisional_nodes=provisional)

    result = await extractor(content="irrelevant")

    assert isinstance(result, ExtractNodesOutput)
    assert len(result.nodes) == 1
    resolved = result.nodes[0]
    assert resolved.uuid == existing.uuid
    assert set(result.uuid_map.values()) == {existing.uuid}
    assert len(result.uuid_map) == len(provisional)
    assert result.metadata["resolved_count"] == 1
    assert result.metadata["new_entities"] == 0
    assert result.duplicate_pairs  # duplicates recorded


@pytest.mark.asyncio
async def test_extract_nodes_applies_labels_and_timezone(monkeypatch: pytest.MonkeyPatch):
    """Ensure labels avoid duplication and valid_at is normalized to UTC."""

    async def fake_fetch_entities(group_id: str):
        assert group_id == "test_user"
        return {}

    async def fake_fetch_recent(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        "pipeline.extract_nodes.fetch_entities_by_group", fake_fetch_entities
    )
    monkeypatch.setattr(
        "pipeline.extract_nodes.fetch_recent_episodes", fake_fetch_recent
    )

    entities = [
        ExtractedEntity(name="Alice", entity_type_id=0),
        ExtractedEntity(name="Bob", entity_type_id=1),
    ]
    stub = StubExtractor(entities=entities)
    extractor = ExtractNodes(group_id="test_user")
    extractor.extractor = stub

    naive_reference = datetime(2024, 1, 1, 12, 0, 0)
    entity_types = {"Person": object()}
    result = await extractor(
        content="Conversation notes.",
        reference_time=naive_reference,
        entity_types=entity_types,
    )

    assert result.metadata["extracted_count"] == 2
    assert result.metadata["new_entities"] == 2
    assert len(result.nodes) == 2

    labels_by_name = {node.name: node.labels for node in result.nodes}
    assert labels_by_name["Alice"] == ["Entity"]
    assert labels_by_name["Bob"] == ["Entity", "Person"]

    valid_at = result.episode.valid_at
    assert valid_at.tzinfo is not None
    assert valid_at.tzinfo == timezone.utc
