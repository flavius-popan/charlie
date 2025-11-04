"""Entity extraction and resolution module for Graphiti pipeline.

Architecture Decision: Async for database I/O, sync for LLM inference.
DSPy signatures remain synchronous (handled by dspy_outlines adapter),
while database queries are async to prevent blocking.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any

import dspy
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from pydantic import BaseModel, Field

from pipeline.db_utils import fetch_entities_by_group, fetch_recent_episodes


logger = logging.getLogger(__name__)


# Pydantic models for structured output via Outlines
class ExtractedEntity(BaseModel):
    """Entity with type classification."""

    name: str = Field(description="Entity name")
    entity_type_id: int = Field(description="0=Entity, 1+=custom types", ge=0)


class ExtractedEntities(BaseModel):
    """Collection of extracted entities."""

    extracted_entities: list[ExtractedEntity]


# DSPy signature for entity extraction
class EntityExtractionSignature(dspy.Signature):
    """Extract entity nodes from text.

    Extract entities that are explicitly or implicitly mentioned in the provided text.
    """

    episode_content: str = dspy.InputField(desc="Text to extract entities from")
    entity_types: str = dspy.InputField(desc="JSON schema of available entity types with descriptions")

    extracted_entities: ExtractedEntities = dspy.OutputField(
        desc="List of entities extracted from the text with entity_type_id classifications"
    )


@dataclass
class ExtractNodesOutput:
    """Results from node extraction and resolution."""

    episode: EpisodicNode
    nodes: list[EntityNode]
    uuid_map: dict[str, str]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]]
    metadata: dict[str, Any]


class ExtractNodes(dspy.Module):
    """Extract and resolve entity nodes from episodes.

    Two-stage pipeline:
    1. DSPy signature extracts entities from episode content
    2. Deterministic resolution using graphiti-core's fuzzy matching

    Usage:
        extractor = ExtractNodes(group_id="user_123")
        result = await extractor(episode=my_episode)  # Call instance, NOT forward()
    """

    def __init__(self, group_id: str, dedupe_enabled: bool = True):
        super().__init__()
        self.group_id = group_id
        self.dedupe_enabled = dedupe_enabled
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)

    async def forward(
        self,
        content: str,
        reference_time: datetime | None = None,
        name: str | None = None,
        source_description: str = "Journal entry",
        entity_types: dict | None = None,
    ) -> ExtractNodesOutput:
        """Extract and resolve entities from journal entry text.

        Called internally via __call__(). Use: result = await extractor(content="...")

        Args:
            content: Journal entry text
            reference_time: When entry was written (defaults to now)
            name: Episode identifier (defaults to generated name)
            source_description: Description of source
            entity_types: Custom entity type schemas

        Returns:
            ExtractNodesOutput with episode, resolved nodes, and UUID mappings
        """
        # Create episode (follows graphiti-core convention)
        reference_time = ensure_utc(reference_time or utc_now())
        episode = EpisodicNode(
            name=name or f"journal_{reference_time.isoformat()}",
            group_id=self.group_id,
            labels=[],
            source=EpisodeType.text,
            content=content,
            source_description=source_description,
            created_at=utc_now(),
            valid_at=reference_time,
        )

        logger.info("Created episode %s", episode.uuid)

        # Fetch recent episodes for context (async)
        previous_episodes = await fetch_recent_episodes(
            self.group_id, reference_time, limit=5
        )
        logger.info(
            "Retrieved %d previous episodes for context", len(previous_episodes)
        )

        # Stage 1: Extract provisional entities (sync DSPy)
        provisional_nodes = self._extract_entities(
            episode, previous_episodes, entity_types
        )
        logger.info("Extracted %d provisional entities", len(provisional_nodes))

        # Stage 2: Resolve against existing graph (async DB + sync resolution)
        existing_entities = await fetch_entities_by_group(self.group_id)
        logger.info("Resolving against %d existing entities", len(existing_entities))

        nodes, uuid_map, duplicate_pairs = self._resolve_entities(
            provisional_nodes, existing_entities
        )

        new_entities = sum(
            1
            for provisional_uuid, resolved_uuid in uuid_map.items()
            if provisional_uuid == resolved_uuid
        )
        metadata = {
            "extracted_count": len(provisional_nodes),
            "resolved_count": len(nodes),
            "exact_matches": len(
                [p for p in duplicate_pairs if p[0].name.lower() == p[1].name.lower()]
            ),
            "fuzzy_matches": len(
                [p for p in duplicate_pairs if p[0].name.lower() != p[1].name.lower()]
            ),
            "new_entities": new_entities,
        }

        logger.info(
            "Resolution complete: %d nodes (%d exact, %d fuzzy, %d new)",
            metadata["resolved_count"],
            metadata["exact_matches"],
            metadata["fuzzy_matches"],
            metadata["new_entities"],
        )

        return ExtractNodesOutput(
            episode=episode,
            nodes=nodes,
            uuid_map=uuid_map,
            duplicate_pairs=duplicate_pairs,
            metadata=metadata,
        )

    def _extract_entities(
        self,
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
        entity_types: dict | None,
    ) -> list[EntityNode]:
        """Extract entities via DSPy signature with dspy_outlines adapter.

        Note: previous_episodes are fetched for future use (reflexion, classification)
        but NOT used in initial entity extraction, following graphiti-core's approach.
        """
        entity_types_json = self._format_entity_types(entity_types)

        result = self.extractor(
            episode_content=episode.content,
            entity_types=entity_types_json,
        )

        # dspy_outlines adapter returns ExtractedEntities object (not JSON string)
        extracted = result.extracted_entities

        nodes = []
        for entity in extracted.extracted_entities:
            type_name = self._get_type_name(entity.entity_type_id, entity_types)
            labels = ["Entity"]
            if type_name != "Entity":
                labels.append(type_name)
            node = EntityNode(
                name=entity.name,
                group_id=self.group_id,
                labels=labels,
                summary="",
                created_at=utc_now(),
            )
            nodes.append(node)

        return nodes

    def _resolve_entities(
        self,
        provisional_nodes: list[EntityNode],
        existing_nodes: dict[str, EntityNode],
    ) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
        """Resolve provisional nodes using graphiti-core's deterministic matching.

        Two-pass resolution:
        1. Exact match: Normalized string equality
        2. Fuzzy match: MinHash LSH + Jaccard similarity (threshold=0.9)

        LLM disambiguation (Pass 3) is stubbed for future implementation.
        """
        if not self.dedupe_enabled:
            uuid_map = {node.uuid: node.uuid for node in provisional_nodes}
            return provisional_nodes, uuid_map, []

        # Build candidate indexes for fuzzy matching
        indexes: DedupCandidateIndexes = _build_candidate_indexes(
            list(existing_nodes.values())
        )

        # Track resolution state
        state = DedupResolutionState(
            resolved_nodes=[None] * len(provisional_nodes),
            uuid_map={},
            unresolved_indices=[],
            duplicate_pairs=[],
        )

        # Pass 1 & 2: Exact + fuzzy matching (deterministic)
        _resolve_with_similarity(provisional_nodes, indexes, state)

        # Pass 3: LLM disambiguation (stubbed - treat unresolved as new)
        for idx in state.unresolved_indices:
            node = provisional_nodes[idx]
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

        # Filter out None entries and deduplicate by canonical UUID
        unique_nodes: dict[str, EntityNode] = {}
        for node in state.resolved_nodes:
            if node is None:
                continue
            if node.uuid not in unique_nodes:
                unique_nodes[node.uuid] = node

        return list(unique_nodes.values()), state.uuid_map, state.duplicate_pairs

    def _format_entity_types(self, entity_types: dict | None) -> str:
        """Format entity type schemas for LLM prompt."""
        base_type = {
            "entity_type_id": 0,
            "entity_type_name": "Entity",
            "entity_type_description": "Default entity classification",
        }

        if not entity_types:
            return json.dumps([base_type])

        types_list = [base_type] + [
            {
                "entity_type_id": i + 1,
                "entity_type_name": name,
                "entity_type_description": model.__doc__ or "",
            }
            for i, (name, model) in enumerate(entity_types.items())
        ]

        return json.dumps(types_list)

    def _get_type_name(self, type_id: int, entity_types: dict | None) -> str:
        """Map entity_type_id to type name."""
        if type_id == 0 or not entity_types:
            return "Entity"

        types_list = list(entity_types.keys())
        idx = type_id - 1
        return types_list[idx] if 0 <= idx < len(types_list) else "Entity"

    # ========== Future Enhancements (Stubs) ==========

    def _extract_with_reflexion(self, episode, previous_episodes, entity_types):
        """TODO: Add reflexion loop (EntityReflexionSignature, max 3 iterations)."""
        return self._extract_entities(episode, previous_episodes, entity_types)

    async def _collect_candidates_with_embeddings(self, provisional_nodes, group_id):
        """TODO: Hybrid search (embedding + text) when Qwen embedder lands."""
        return await fetch_entities_by_group(group_id)

    async def _disambiguate_with_llm(self, unresolved, candidates, episode):
        """TODO: EntityDisambiguationSignature for ambiguous matches."""
        return {node.uuid: node.uuid for node in unresolved}


__all__ = ["ExtractNodes", "ExtractNodesOutput"]
