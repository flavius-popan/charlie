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

import dspy
from distilbert_ner import predict_entities
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts.prompt_helpers import to_prompt_json
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
from pydantic import BaseModel, Field

from pipeline.falkordblite_driver import fetch_entities_by_group, fetch_recent_episodes
from pipeline.entity_edge_models import entity_types
from pipeline.ner_type_overrides import map_ner_label_to_entity_type


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


class ReflexionEntity(BaseModel):
    """Entity suggestion produced by the reflexion step."""

    name: str = Field(
        ...,
        description="Name of an entity that still needs to be extracted",
        min_length=1,
    )
    entity_type: str | None = Field(
        default=None,
        description="Optional type hint using the journaling entity schemas",
    )


class ReflexionEntities(BaseModel):
    """Structured container for reflexion suggestions."""

    missed_entities: list[ReflexionEntity] = Field(
        default_factory=list,
        description="Entities missed during the first extraction pass",
    )


# DSPy signature for entity extraction
class EntityExtractionSignature(dspy.Signature):
    """Extract entities from journal entries based on the provided schema."""

    episode_content: str = dspy.InputField(desc="journal entry text")
    entity_types: str = dspy.InputField(desc="JSON list of entity type definitions")

    extracted_entities: ExtractedEntities = dspy.OutputField(
        desc="entities found in the text, classified by type"
    )


class EntityReflexionSignature(dspy.Signature):
    """Suggest entities missed by the initial extraction."""

    episode_content: str = dspy.InputField(desc="journal entry text")
    previous_episodes: str = dspy.InputField(desc="previous entries for context")
    entity_types: str = dspy.InputField(desc="available entity types")
    extracted_entities: str = dspy.InputField(desc="already extracted entities")

    missed_entities: ReflexionEntities = dspy.OutputField(
        desc="entities to add"
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
        self.extractor = dspy.Predict(EntityExtractionSignature)

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


class EntityReflexionModule(dspy.Module):
    """Lightweight DSPy module that mirrors graphiti-core's reflexion prompt."""

    def __init__(self):
        super().__init__()
        self.predict_reflexion = dspy.Predict(EntityReflexionSignature)

        prompt_path = Path(__file__).parent / "prompts" / "entity_reflexion.json"
        if prompt_path.exists():
            self.load(str(prompt_path))
            logger.info("Loaded reflexion prompts from %s", prompt_path)

    def forward(
        self,
        episode_content: str,
        previous_episodes: str,
        entity_types: str,
        extracted_entities: str,
    ) -> list[ReflexionEntity]:
        result = self.predict_reflexion(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_types=entity_types,
            extracted_entities=extracted_entities,
        )
        return self._normalize_reflexion_output(result)

    def _normalize_reflexion_output(self, result: Any) -> list[ReflexionEntity]:
        """Gracefully coerce mixed model outputs (dict/list/tuples) into ReflexionEntity objects."""
        candidates: list[Any]

        if isinstance(result, ReflexionEntities):
            candidates = result.missed_entities
        elif hasattr(result, "missed_entities"):
            candidates = getattr(result, "missed_entities")  # type: ignore[assignment]
        elif isinstance(result, dict) and "missed_entities" in result:
            candidates = result["missed_entities"]  # type: ignore[assignment]
        else:
            candidates = result if isinstance(result, list) else []

        normalized: list[ReflexionEntity] = []

        for item in candidates:
            if isinstance(item, ReflexionEntity):
                if item.name.strip():
                    normalized.append(item)
            elif isinstance(item, dict):
                name = _safe_str(item.get("name") or item.get("entity"))
                if not name:
                    logger.debug("Skipping reflexion candidate without name: %s", item)
                    continue
                entity_type = _safe_str(item.get("entity_type") or item.get("type"))
                try:
                    normalized.append(
                        ReflexionEntity(name=name, entity_type=entity_type)
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug("Failed to coerce reflexion candidate %s: %s", item, exc)
            else:
                logger.debug("Ignoring reflexion candidate with unsupported shape: %r", item)

        return normalized


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        entity_reflexion: EntityReflexionModule | None = None,
        use_ner_extractor: bool = False,
        max_reflexion_iterations: int = 1,
    ):
        self.group_id = group_id
        self.dedupe_enabled = dedupe_enabled
        self.use_ner_extractor = use_ner_extractor
        self.max_reflexion_iterations = max(0, max_reflexion_iterations)

        if self.use_ner_extractor:
            self.entity_extractor = entity_extractor  # Optional override; not used by default
            self.entity_reflexion = entity_reflexion or EntityReflexionModule()
        else:
            self.entity_extractor = entity_extractor or EntityExtractor()
            self.entity_reflexion = entity_reflexion

    async def __call__(
        self,
        content: str,
        reference_time: datetime | None = None,
        name: str | None = None,
        source_description: str = "Journal entry",
        entity_types: dict | None = entity_types,
    ) -> ExtractNodesOutput:
        """Extract and resolve entities from journal entry text.

        By default, extracts Person, Place, Organization, and Activity entities.

        Args:
            content: Journal entry text
            reference_time: When entry was written (defaults to now)
            name: Episode identifier (defaults to generated name)
            source_description: Description of source
            entity_types: Custom entity type schemas (defaults to Person, Place, Organization, Activity)

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

        # Stage 1: Extract provisional entities via configured module
        provisional_nodes, extraction_metadata = self._extract_entities(
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
        metadata.update(extraction_metadata)

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
    ) -> tuple[list[EntityNode], dict[str, Any]]:
        """Extract entities via the configured strategy (DSPy or DistilBERT)."""
        if self.use_ner_extractor:
            return self._extract_entities_with_ner(
                episode,
                previous_episodes,
                entity_types,
            )

        return self._extract_entities_with_dspy(
            episode,
            entity_types,
        )

    def _extract_entities_with_dspy(
        self,
        episode: EpisodicNode,
        entity_types: dict | None,
    ) -> tuple[list[EntityNode], dict[str, Any]]:
        """Default DSPy extractor (unchanged from the earlier implementation)."""
        entity_types_json = self._format_entity_types(entity_types)

        extracted = self.entity_extractor(
            episode_content=episode.content,
            entity_types=entity_types_json,
        )

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

        return nodes, {
            "extractor": "dspy",
            "reflexion_added": 0,
            "reflexion_iterations": 0,
        }

    def _extract_entities_with_ner(
        self,
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
        entity_types: dict | None,
    ) -> tuple[list[EntityNode], dict[str, Any]]:
        """Fast DistilBERT extractor backed by a DSPy reflexion pass."""
        predictions = predict_entities(episode.content)
        available_types = set(entity_types.keys()) if entity_types else None

        nodes: list[EntityNode] = []
        seen_names: set[str] = set()
        ner_details: list[dict[str, Any]] = []
        for prediction in predictions:
            name = prediction.get("text", "").strip()
            if not name:
                continue
            normalized = name.lower()
            if normalized in seen_names:
                continue
            seen_names.add(normalized)

            label = self._normalize_ner_label(prediction.get("label", ""))
            type_name = map_ner_label_to_entity_type(label, name, available_types)
            labels = ["Entity"]
            if type_name != "Entity":
                labels.append(type_name)

            node = EntityNode(
                name=name,
                group_id=self.group_id,
                labels=labels,
                summary="",
                created_at=utc_now(),
            )
            nodes.append(node)
            ner_details.append(
                {
                    "name": name,
                    "ner_label": label,
                    "mapped_type": type_name,
                    "confidence": prediction.get("confidence"),
                }
            )

        reflexion_nodes: list[EntityNode] = []
        reflexion_iterations = 0
        if self.max_reflexion_iterations > 0:
            reflexion_nodes, reflexion_iterations = self._run_reflexion(
                episode,
                previous_episodes,
                nodes,
                entity_types,
                seen_names,
            )

        return nodes + reflexion_nodes, {
            "extractor": "distilbert",
            "reflexion_added": len(reflexion_nodes),
            "reflexion_iterations": reflexion_iterations,
            "ner_entities": ner_details,
            "reflexion_entities": [node.name for node in reflexion_nodes],
        }

    def _run_reflexion(
        self,
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
        base_nodes: list[EntityNode],
        entity_types: dict | None,
        seen_names: set[str],
    ) -> tuple[list[EntityNode], int]:
        """Run up to N reflexion passes and return newly created nodes."""
        if self.max_reflexion_iterations <= 0 or not self.entity_reflexion:
            return [], 0

        reflexion_nodes: list[EntityNode] = []
        previous_payload = (
            to_prompt_json([ep.content for ep in previous_episodes]) if previous_episodes else "[]"
        )
        current_nodes = list(base_nodes)
        available_types = self._get_available_type_names(entity_types)
        entity_types_payload = self._format_entity_types(entity_types)

        for iteration in range(self.max_reflexion_iterations):
            extracted_payload = to_prompt_json([node.name for node in current_nodes])
            suggestions = self.entity_reflexion(
                episode_content=episode.content,
                previous_episodes=previous_payload,
                entity_types=entity_types_payload,
                extracted_entities=extracted_payload,
            )

            new_nodes = self._convert_reflexion_suggestions(
                suggestions,
                available_types,
                seen_names,
            )

            if not new_nodes:
                return reflexion_nodes, iteration + 1

            logger.info(
                "Reflexion iteration %d added %d entities",
                iteration + 1,
                len(new_nodes),
            )
            reflexion_nodes.extend(new_nodes)
            current_nodes.extend(new_nodes)

        return reflexion_nodes, self.max_reflexion_iterations

    def _convert_reflexion_suggestions(
        self,
        suggestions: list[ReflexionEntity],
        available_types: set[str] | None,
        seen_names: set[str],
    ) -> list[EntityNode]:
        """Turn reflexion suggestions into EntityNode instances."""
        new_nodes: list[EntityNode] = []

        for suggestion in suggestions:
            name = (suggestion.name or "").strip()
            if not name:
                continue
            normalized = name.lower()
            if normalized in seen_names:
                continue
            seen_names.add(normalized)

            type_name = self._normalize_type_hint(suggestion.entity_type, available_types)
            if not type_name or type_name == "Entity":
                type_name = map_ner_label_to_entity_type("MISC", name, available_types)

            labels = ["Entity"]
            if type_name != "Entity":
                labels.append(type_name)

            new_nodes.append(
                EntityNode(
                    name=name,
                    group_id=self.group_id,
                    labels=labels,
                    summary="",
                    created_at=utc_now(),
                )
            )

        return new_nodes

    def _normalize_type_hint(
        self,
        entity_type_hint: str | None,
        available_types: set[str] | None,
    ) -> str | None:
        if not entity_type_hint:
            return None
        hint = entity_type_hint.strip()
        if not hint:
            return None
        if not available_types:
            return hint

        for candidate in available_types:
            if candidate.lower() == hint.lower():
                return candidate
        return None

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

    @staticmethod
    def _format_entity_types(entity_types: dict | None) -> str:
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
            desc = model.__doc__ or ""

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

    def _get_available_type_names(self, entity_types: dict | None) -> set[str] | None:
        if not entity_types:
            return None
        return set(entity_types.keys())

    def _normalize_ner_label(self, label: str | None) -> str:
        if not label:
            return ""
        return label.split("-")[-1].upper()


__all__ = [
    "EntityExtractor",
    "EntityReflexionModule",
    "ExtractNodes",
    "ExtractNodesOutput",
]
