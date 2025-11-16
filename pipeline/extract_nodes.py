"""Entity extraction and resolution module for Graphiti pipeline.

Two-Layer Architecture for DSPy Optimization:

1. EntityExtractor (dspy.Module):
   - Pure LLM extraction logic, no DB/async dependencies
   - Optimizable with DSPy (MIPRO, GEPA, etc.)
   - Input: episode_content (str), entity_types (JSON str)
   - Output: ExtractedEntities (Pydantic model)

2. ExtractNodes (orchestrator):
   - Plain Python class that coordinates the full pipeline
   - Handles DB I/O (async), episode creation, entity resolution
   - Accepts pre-compiled EntityExtractor via dependency injection
   - Returns ExtractNodesOutput with metadata

Usage:
    # Standard usage
    extractor = ExtractNodes(group_id="user_123")
    result = await extractor(content="Today I met with Sarah...")

    # With optimization
    entity_extractor = EntityExtractor()
    compiled = optimizer.compile(entity_extractor, trainset=examples)
    extractor = ExtractNodes(group_id="user_123", entity_extractor=compiled)
    result = await extractor(content="...")

This pattern enables fast optimization by isolating LLM calls from I/O.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
import copy

import dspy
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
from pydantic import BaseModel, Field, model_validator

from pipeline.falkordblite_driver import (
    fetch_entities_by_group,
    fetch_recent_episodes,
    fetch_self_entity,
)
from pipeline.entity_edge_models import entity_types
from pipeline.self_reference import (
    build_provisional_self_node,
    contains_first_person_reference,
    is_self_entity_name,
)


logger = logging.getLogger(__name__)


# Pydantic models for structured output via Outlines
class ExtractedEntity(BaseModel):
    """Entity with name and type classification.

    Attributes are extracted separately in Stage 3, following graphiti-core's pattern.
    """

    name: str = Field(description="Entity name")
    entity_type_id: int = Field(description="0=Entity, 1+=custom types", ge=0)


class ExtractedEntities(BaseModel):
    """Collection of extracted entities."""

    extracted_entities: list[ExtractedEntity]

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, value):
        """Allow bare lists to be parsed as extracted_entities."""
        if isinstance(value, list):
            return {"extracted_entities": value}
        return value


# DSPy signature for entity extraction
class EntityExtractionSignature(dspy.Signature):
    """Extract significant entities from personal journal entries: people, places, organizations, concepts, and activities."""

    episode_content: str = dspy.InputField(
        desc="personal journal entry describing experiences, relationships, locations, and reflections"
    )
    entity_types: str = dspy.InputField(
        desc="available entity types: Person, Place, Organization, Concept, Activity, and Entity (fallback for others)"
    )

    extracted_entities: ExtractedEntities = dspy.OutputField(
        desc="entities mentioned: individuals, specific venues/locations, institutions/groups, abstract topics/themes, and activities/events"
    )


@dataclass
class ExtractNodesOutput:
    """Results from node extraction and resolution."""

    episode: EpisodicNode
    extracted_nodes: list[
        EntityNode
    ]  # Nodes with original UUIDs (for Stage 2 edge extraction)
    nodes: list[EntityNode]  # Resolved nodes with canonical UUIDs
    uuid_map: dict[str, str]  # provisional_uuid → canonical_uuid
    duplicate_pairs: list[
        tuple[EntityNode, EntityNode]
    ]  # Consumed in Stage 5 for DUPLICATE_OF edges
    metadata: dict[str, Any]
    previous_episodes: list[EpisodicNode]  # For Stage 2 edge extraction context


class EntityExtractor(dspy.Module):
    """This module ONLY performs LLM-based entity extraction with no dependencies
    on databases, async operations, or resolution logic. This design makes it
    ideal for DSPy optimization (MIPRO, GEPA, etc.).

    Usage:
        extractor = EntityExtractor()
        entities = extractor(episode_content="...", entity_types="{...}")
    """

    def __init__(self):
        super().__init__()
        # ChainOfThought tends to keep the JSON schema instructions in view for smaller local models.
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)

        # Auto-load optimized prompts if they exist
        prompt_path = Path(__file__).parent / "prompts" / "extract_nodes.json"
        if prompt_path.exists():
            self.load(str(prompt_path))
            logger.info("Loaded optimized prompts from %s", prompt_path)

    def forward(
        self,
        episode_content: str,
        entity_types: str,
    ) -> ExtractedEntities:
        """Extract entities from text content.

        Args:
            episode_content: Text to extract entities from
            entity_types: JSON string of available entity type schemas

        Returns:
            ExtractedEntities with list of extracted entities and type IDs
        """
        result = self.extractor(
            episode_content=episode_content,
            entity_types=entity_types,
        )
        return result.extracted_entities


class ExtractNodes:
    """Extract and resolve entity nodes from episodes.

    Pipeline orchestrator that coordinates:
    1. Episode creation
    2. Database context fetching (async)
    3. Entity extraction via EntityExtractor (optimizable dspy.Module)
    4. Entity resolution using graphiti-core's MinHash LSH
    5. Metadata aggregation

    This is a plain Python class (NOT dspy.Module) to separate orchestration
    from the optimizable LLM logic in EntityExtractor.

    Usage:
        # Standard usage with default extractor
        extractor = ExtractNodes(group_id="user_123")
        result = await extractor(content="Today I met with Sarah...")

        # With pre-optimized EntityExtractor
        entity_extractor = EntityExtractor()
        compiled = optimizer.compile(entity_extractor, trainset=examples)
        extractor = ExtractNodes(group_id="user_123", entity_extractor=compiled)
        result = await extractor(content="...")
    """

    def __init__(
        self,
        group_id: str,
        dedupe_enabled: bool = True,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.group_id = group_id
        self.dedupe_enabled = dedupe_enabled
        self.entity_extractor = entity_extractor or EntityExtractor()

    async def __call__(
        self,
        content: str,
        reference_time: datetime | None = None,
        name: str | None = None,
        source_description: str = "Journal entry",
        entity_types: dict | None = entity_types,
    ) -> ExtractNodesOutput:
        """Extract and resolve entities from journal entry text.

        By default, extracts Person, Place, Organization, Concept, and Activity entities.

        Args:
            content: Journal entry text
            reference_time: When entry was written (defaults to now)
            name: Episode identifier (defaults to generated name)
            source_description: Description of source
            entity_types: Custom entity type schemas (defaults to Person, Place, Organization, Concept, Activity)

        Returns:
            ExtractNodesOutput with episode, resolved nodes, and UUID mappings
        """
        # Validate entity types using graphiti-core validator
        validate_entity_types(entity_types)
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

        # Stage 1: Extract provisional entities via pure DSPy module
        pronoun_detected = contains_first_person_reference(episode.content)
        canonical_self = None
        if pronoun_detected:
            canonical_self = await fetch_self_entity(self.group_id)

        provisional_nodes = self._extract_entities(
            episode, previous_episodes, entity_types
        )
        logger.info("Extracted %d provisional entities", len(provisional_nodes))

        first_person_detected, self_injected = self._handle_self_reference(
            provisional_nodes, pronoun_detected, canonical_self
        )
        if self_injected:
            logger.debug("Injected SELF placeholder for group %s", self.group_id)

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
            "first_person_detected": first_person_detected,
            "self_node_injected": self_injected,
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
            extracted_nodes=provisional_nodes,
            nodes=nodes,
            uuid_map=uuid_map,
            duplicate_pairs=duplicate_pairs,
            metadata=metadata,
            previous_episodes=previous_episodes,
        )

    def _extract_entities(
        self,
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
        entity_types: dict | None,
    ) -> list[EntityNode]:
        """Extract entities via EntityExtractor module.

        Note: previous_episodes are fetched for future use (reflexion, classification)
        but NOT used in initial entity extraction, following graphiti-core's approach.
        """
        entity_types_json = self._format_entity_types(entity_types)

        # Call the pure DSPy module (optimizable)
        extracted = self.entity_extractor(
            episode_content=episode.content,
            entity_types=entity_types_json,
        )

        # Convert to EntityNode objects
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

    def _handle_self_reference(
        self,
        nodes: list[EntityNode],
        pronoun_detected: bool,
        canonical_self: EntityNode | None,
    ) -> tuple[bool, bool]:
        """Normalize how SELF is represented based on pronoun usage."""
        has_llm_self = any(is_self_entity_name(node.name) for node in nodes)
        if has_llm_self and canonical_self is not None:
            for idx, node in enumerate(list(nodes)):
                if is_self_entity_name(node.name):
                    nodes[idx] = copy.deepcopy(canonical_self)

        if not pronoun_detected and has_llm_self:
            nodes[:] = [node for node in nodes if not is_self_entity_name(node.name)]
            has_llm_self = False

        injected = False
        if pronoun_detected and not has_llm_self:
            if canonical_self is not None:
                nodes.append(copy.deepcopy(canonical_self))
            else:
                nodes.append(build_provisional_self_node(self.group_id))
            has_llm_self = True
            injected = True

        first_person_detected = pronoun_detected or has_llm_self
        return first_person_detected, injected

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
        """Format entity type schemas for entity_types input field."""
        base_type = {
            "entity_type_id": 0,
            "entity_type_name": "Entity",
            "entity_type_description": "fallback for significant entities that don't fit custom types",
        }

        if not entity_types:
            return json.dumps([base_type])

        types_list = [base_type]
        for i, (name, model) in enumerate(entity_types.items()):
            base_desc = model.__doc__ or ""

            # Enhance with extraction guidance for personal journaling
            if name == "Person":
                desc = f"{base_desc} Extract individuals mentioned: friends, family, colleagues, romantic partners, professionals (therapists/doctors), acquaintances."
            elif name == "Place":
                desc = f"{base_desc} Extract specific places visited or mentioned: coffee shops, parks, restaurants, cities, neighborhoods, venues, landmarks."
            elif name == "Organization":
                desc = f"{base_desc} Extract organizations engaged with: workplaces, schools, clubs, community groups, institutions, companies."
            elif name == "Concept":
                desc = f"{base_desc} Extract life themes and topics reflected on: personal growth, relationships, mental health, career, identity, values, beliefs."
            elif name == "Activity":
                desc = f"{base_desc} Extract specific activities and events: appointments, outings, hobbies, social gatherings, daily routines, significant moments."
            else:
                desc = base_desc

            types_list.append(
                {
                    "entity_type_id": i + 1,
                    "entity_type_name": name,
                    "entity_type_description": desc,
                }
            )

        return json.dumps(types_list)

    def _get_type_name(self, type_id: int, entity_types: dict | None) -> str:
        """Map entity_type_id to type name."""
        if type_id == 0 or not entity_types:
            return "Entity"

        types_list = list(entity_types.keys())
        idx = type_id - 1
        if not (0 <= idx < len(types_list)):
            logger.warning("Invalid entity_type_id %d, falling back to 'Entity'", type_id)
            return "Entity"
        return types_list[idx]

    # ========== Future Enhancements (Stubs) ==========
    # These demonstrate where additional optimizable modules could be added:
    # - EntityReflexionModule for iterative refinement
    # - EntityDisambiguationModule for LLM-based resolution
    # Each would follow the same pattern: pure dspy.Module, injected into orchestrator

    def _extract_with_reflexion(self, episode, previous_episodes, entity_types):
        """TODO: Add reflexion loop (EntityReflexionSignature, max 3 iterations)."""
        return self._extract_entities(episode, previous_episodes, entity_types)

    async def _collect_candidates_with_embeddings(self, provisional_nodes, group_id):
        """TODO: Hybrid search (embedding + text) when Qwen embedder lands."""
        return await fetch_entities_by_group(group_id)

    async def _disambiguate_with_llm(self, unresolved, candidates, episode):
        """TODO: EntityDisambiguationSignature for ambiguous matches."""
        return {node.uuid: node.uuid for node in unresolved}


__all__ = [
    "EntityExtractor",
    "ExtractNodes",
    "ExtractNodesOutput",
]
