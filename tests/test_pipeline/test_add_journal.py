"""Tests for add_journal orchestrator using factory hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from pydantic import BaseModel

from pipeline import AddJournalResults, ExtractNodesOutput, add_journal
from graphiti_core.errors import GroupIdValidationError
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now


class PersonEntity(BaseModel):
    occupation: str | None = None


@dataclass
class RecordingExtractNodes:
    """Stub extractor that records calls and returns canned results."""

    group_id: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs) -> ExtractNodesOutput:
        self.calls.append(kwargs)
        content = kwargs["content"]
        reference_time = kwargs.get("reference_time")
        name = kwargs.get("name")
        source_description = kwargs.get("source_description")

        episode = EpisodicNode(
            name=name or "stub-episode",
            group_id=self.group_id,
            labels=[],
            source=EpisodeType.text,
            content=content,
            source_description=source_description,
            created_at=utc_now(),
            valid_at=ensure_utc(reference_time or utc_now()),
        )

        node = EntityNode(
            name="Stub Entity",
            group_id=self.group_id,
            labels=["Entity"],
            summary="",
            created_at=utc_now(),
        )

        metadata = {
            "extracted_count": 1,
            "resolved_count": 1,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "new_entities": 1,
        }

        return ExtractNodesOutput(
            episode=episode,
            nodes=[node],
            uuid_map={node.uuid: node.uuid},
            duplicate_pairs=[],
            metadata=metadata,
        )


@pytest.mark.asyncio
async def test_add_journal_uses_factory_and_returns_results():
    journal_text = "Met with Sarah about the roadmap."
    reference_time = datetime(2024, 2, 1, 9, 30)
    extractors: list[RecordingExtractNodes] = []

    def factory(group_id: str) -> RecordingExtractNodes:
        extractor = RecordingExtractNodes(group_id=group_id)
        extractors.append(extractor)
        return extractor

    entity_types = {"Person": PersonEntity}

    result = await add_journal(
        content=journal_text,
        group_id="test_user",
        reference_time=reference_time,
        name="custom-episode",
        source_description="Unit test",
        entity_types=entity_types,
        extract_nodes_factory=factory,
    )

    assert isinstance(result, AddJournalResults)
    assert result.episode.group_id == "test_user"
    assert len(result.nodes) == 1
    assert result.metadata["new_entities"] == 1

    extractor = extractors[0]
    assert extractor.group_id == "test_user"
    assert len(extractor.calls) == 1
    call_kwargs = extractor.calls[0]
    assert call_kwargs["content"] == journal_text
    assert call_kwargs["reference_time"] == reference_time
    assert call_kwargs["name"] == "custom-episode"
    assert call_kwargs["source_description"] == "Unit test"
    assert call_kwargs["entity_types"] == entity_types


@pytest.mark.asyncio
async def test_add_journal_with_default_group_id():
    extractors: list[RecordingExtractNodes] = []

    def factory(group_id: str) -> RecordingExtractNodes:
        extractor = RecordingExtractNodes(group_id=group_id)
        extractors.append(extractor)
        return extractor

    result = await add_journal(
        content="Daily entry text.",
        extract_nodes_factory=factory,
    )

    assert result.episode.group_id == "\\_"
    assert all(node.group_id == "\\_" for node in result.nodes)

    extractor = extractors[0]
    assert extractor.group_id == "\\_"
    assert extractor.calls[0]["content"] == "Daily entry text."


@pytest.mark.asyncio
async def test_add_journal_validates_inputs():
    def factory(_group_id: str):
        raise AssertionError("Factory should not be invoked when validation fails.")

    with pytest.raises(GroupIdValidationError):
        await add_journal(
            content="invalid input",
            group_id="invalid@group!",
            extract_nodes_factory=factory,
        )
