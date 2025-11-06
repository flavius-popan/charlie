"""Edge extraction and resolution module for Graphiti pipeline.

Two-Layer Architecture for DSPy Optimization:

1. EdgeExtractor (dspy.Module):
   - Pure LLM extraction logic, no DB/async dependencies
   - Optimizable with DSPy (MIPRO, GEPA, etc.)
   - Input: episode_content (str), entities (list[str]), reference_time (str)
   - Output: ExtractedRelationships (Pydantic model)

2. ExtractEdges (orchestrator):
   - Plain Python class that coordinates the full pipeline
   - Handles DB I/O (async), edge building, resolution
   - Accepts pre-compiled EdgeExtractor via dependency injection
   - Returns ExtractEdgesOutput with metadata

This pattern enables fast optimization by isolating LLM calls from I/O.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import dspy
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.utils.bulk_utils import resolve_edge_pointers
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from pydantic import BaseModel, Field

from pipeline.db_utils import fetch_entity_edges_by_group

logger = logging.getLogger(__name__)


class ExtractedRelationship(BaseModel):
    """Relationship between two entities with supporting fact."""

    source: str = Field(
        description="Source entity name (must match extracted entities)"
    )
    target: str = Field(
        description="Target entity name (must match extracted entities)"
    )
    relation: str = Field(
        description="Relationship type in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, KNOWS)"
    )
    fact: str = Field(
        description="Natural language description of the relationship from the text"
    )
    valid_at: str | None = Field(
        None, description="ISO 8601 datetime when relationship became true (optional)"
    )
    invalid_at: str | None = Field(
        None, description="ISO 8601 datetime when relationship ended (optional)"
    )


class ExtractedRelationships(BaseModel):
    """Collection of relationships extracted from episode."""

    relationships: list[ExtractedRelationship]


class RelationshipExtractionSignature(dspy.Signature):
    """Extract relationships between entities from journal text.

    For each relationship, extract:
    - Source and target entities (must be from the provided entity list)
    - Relation type as SCREAMING_SNAKE_CASE (e.g., WORKS_AT, FRIEND_OF, KNOWS)
    - Supporting fact: a natural paraphrasing from the text
    - Optional: valid_at (when relationship started) as ISO 8601 string
    - Optional: invalid_at (when relationship ended) as ISO 8601 string

    Focus on relationships explicitly mentioned in the text.
    Use reference_time for resolving relative dates.
    """

    episode_content: str = dspy.InputField(
        desc="Journal entry text describing relationships between entities"
    )
    entities: list[str] = dspy.InputField(
        desc="List of entity names to consider for relationships"
    )
    reference_time: str = dspy.InputField(
        desc="Current timestamp for resolving relative dates (ISO 8601 format)"
    )

    relationships: ExtractedRelationships = dspy.OutputField(
        desc="Relationships between entities with supporting facts and optional temporal metadata"
    )


class EdgeExtractor(dspy.Module):
    """Pure LLM extraction module for relationships - optimizable with DSPy.

    This module ONLY performs LLM-based relationship extraction with no dependencies
    on databases, async operations, or resolution logic. This design makes it
    ideal for DSPy optimization (MIPRO, GEPA, etc.).

    Usage:
        extractor = EdgeExtractor()
        relationships = extractor(
            episode_content="...",
            entities=["Sarah", "Stanford"],
            reference_time="2025-01-01T00:00:00Z"
        )
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(RelationshipExtractionSignature)

        prompt_path = Path(__file__).parent / "prompts" / "extract_edges.json"
        if prompt_path.exists():
            self.load(str(prompt_path))
            logger.info("Loaded optimized prompts from %s", prompt_path)

    def forward(
        self,
        episode_content: str,
        entities: list[str],
        reference_time: str,
        previous_episodes: list[EpisodicNode] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    ) -> ExtractedRelationships:
        """Extract relationships from text content.

        Args:
            episode_content: Text to extract relationships from
            entities: List of entity names (from Stage 1)
            reference_time: ISO 8601 timestamp for relative date resolution
            previous_episodes: Reserved for future enhancement (Stage 3 context)
            edge_types: Reserved for future enhancement (custom edge type schemas)
            edge_type_map: Reserved for future enhancement (entity pair → allowed edge types)

        Returns:
            ExtractedRelationships with list of relationships
        """
        # TODO: Future enhancement - use previous_episodes for relationship disambiguation
        # Example: LLM sees "met again" and knows from context this is recurring relationship
        # if previous_episodes:
        #     context['previous_messages'] = [
        #         {'episode_uuid': ep.uuid, 'content': ep.content}
        #         for ep in previous_episodes
        #     ]

        # TODO: Future enhancement - use edge_type_map to guide LLM on allowed edge types
        # Example: For (Person, Emotion) only allow EmotionalAssociation, CoOccurrence
        # if edge_type_map:
        #     edge_type_signature_map = {
        #         edge_type_name: edge_types[edge_type_name]
        #         for edge_types_list in edge_type_map.values()
        #         for edge_type_name in edge_types_list
        #     }
        #     # Pass to LLM in signature context

        result = self.extractor(
            episode_content=episode_content,
            entities=entities,
            reference_time=reference_time,
        )
        return result.relationships


@dataclass
class ExtractEdgesOutput:
    """Results from edge extraction and resolution."""

    edges: list[EntityEdge]
    metadata: dict[str, Any]


def normalize_edge_name(name: str) -> str:
    """Normalize relationship name to SCREAMING_SNAKE_CASE.

    Mirrors graphiti-core's edge naming convention.

    Examples:
        "works at" → "WORKS_AT"
        "knows" → "KNOWS"
        "friend of" → "FRIEND_OF"
    """
    name = " ".join(name.split())
    name = name.replace(" ", "_")
    return name.upper()


def _parse_temporal_field(value: str | None, field_name: str) -> datetime | None:
    """Parse ISO 8601 temporal field with error handling.

    Args:
        value: ISO 8601 string (e.g., "2025-01-01T00:00:00Z")
        field_name: Field name for logging ("valid_at" or "invalid_at")

    Returns:
        UTC datetime or None if parsing fails
    """
    if not value:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Error parsing {field_name}: {e}. Input: {value}")
        return None


def build_entity_edges(
    relationships: ExtractedRelationships,
    entity_map: dict[str, EntityNode],
    episode_uuid: str,
    group_id: str,
) -> list[EntityEdge]:
    """Build EntityEdge objects from extracted relationships.

    Args:
        relationships: Relationships from EdgeExtractor
        entity_map: Dict mapping normalized_entity_name → EntityNode (resolved nodes)
        episode_uuid: UUID of the originating episode
        group_id: Graph partition identifier

    Returns:
        List of EntityEdge objects (relationships with missing entities are skipped)
    """
    entity_edges = []

    for rel in relationships.relationships:
        source_node = entity_map.get(_normalize_string_exact(rel.source))
        target_node = entity_map.get(_normalize_string_exact(rel.target))

        if not source_node or not target_node:
            logger.warning(
                f"Skipping relationship {rel.source} → {rel.target} "
                f"(entity not found in resolved nodes)"
            )
            continue

        valid_at_datetime = _parse_temporal_field(rel.valid_at, "valid_at")
        invalid_at_datetime = _parse_temporal_field(rel.invalid_at, "invalid_at")

        if (
            valid_at_datetime
            and invalid_at_datetime
            and invalid_at_datetime <= valid_at_datetime
        ):
            logger.warning(
                f"Skipping edge with invalid temporal range: invalid_at ({invalid_at_datetime}) "
                f"<= valid_at ({valid_at_datetime}) for {rel.source} → {rel.target}"
            )
            continue

        edge = EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name=normalize_edge_name(rel.relation),
            fact=rel.fact,
            group_id=group_id,
            created_at=utc_now(),
            fact_embedding=None,
            episodes=[episode_uuid],
            expired_at=None,
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
        )
        entity_edges.append(edge)

    return entity_edges


class ExtractEdges:
    """Extract and resolve entity relationships from episodes.

    Orchestrator that coordinates:
    1. Edge extraction via EdgeExtractor (LLM)
    2. Entity edge building with temporal parsing
    3. Edge pointer resolution (uuid_map remapping)
    4. Exact match deduplication against existing edges

    Follows graphiti-core's 3-step flow: extract → remap → resolve

    This is a plain Python class (NOT dspy.Module) to separate orchestration
    from the optimizable LLM logic in EdgeExtractor.

    Usage:
        # Standard usage with default extractor
        extractor = ExtractEdges(group_id="user_123")
        result = await extractor(
            episode=episode,
            extracted_nodes=extracted_nodes,
            resolved_nodes=resolved_nodes,
            uuid_map=uuid_map,
            previous_episodes=previous_episodes,
        )

        # With pre-optimized EdgeExtractor
        edge_extractor = EdgeExtractor()
        compiled = optimizer.compile(edge_extractor, trainset=examples)
        extractor = ExtractEdges(group_id="user_123", edge_extractor=compiled)
        result = await extractor(...)
    """

    def __init__(
        self,
        group_id: str,
        dedupe_enabled: bool = True,
        edge_extractor: EdgeExtractor | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    ):
        self.group_id = group_id
        self.dedupe_enabled = dedupe_enabled
        self.edge_extractor = edge_extractor or EdgeExtractor()
        self.edge_types = edge_types
        self.edge_type_map = edge_type_map

    async def __call__(
        self,
        episode: EpisodicNode,
        extracted_nodes: list[EntityNode],
        resolved_nodes: list[EntityNode],
        uuid_map: dict[str, str],
        previous_episodes: list[EpisodicNode],
        entity_types: dict[str, type[BaseModel]] | None = None,
    ) -> ExtractEdgesOutput:
        """Extract and resolve edges from episode.

        Mirrors graphiti-core's _extract_and_resolve_edges flow:
        1. Extract edges using extracted_nodes (original UUIDs)
        2. Remap edge UUIDs using uuid_map
        3. Resolve edges against existing graph (exact match dedup)

        Args:
            episode: The episode being processed
            extracted_nodes: Nodes with original UUIDs (for LLM indexing)
            resolved_nodes: Nodes with canonical UUIDs (after deduplication)
            uuid_map: Mapping from provisional_uuid → canonical_uuid
            previous_episodes: Recent episodes for context (reserved for future Stage 3)
            entity_types: Custom entity type schemas (reserved for future Stage 3)

        Returns:
            ExtractEdgesOutput with resolved edges and metadata
        """
        entity_map = {
            _normalize_string_exact(node.name): node for node in resolved_nodes
        }

        logger.info(
            "Extracting edges for episode %s with %d entities",
            episode.uuid,
            len(extracted_nodes),
        )

        entity_names = [node.name for node in extracted_nodes]
        relationships = self.edge_extractor(
            episode_content=episode.content,
            entities=entity_names,
            reference_time=episode.valid_at.isoformat(),
            previous_episodes=previous_episodes,
            edge_types=self.edge_types,
            edge_type_map=self.edge_type_map,
        )

        logger.info("Extracted %d relationships", len(relationships.relationships))

        entity_edges = build_entity_edges(
            relationships,
            entity_map,
            episode.uuid,
            self.group_id,
        )

        logger.info("Built %d entity edges", len(entity_edges))

        entity_edges = resolve_edge_pointers(entity_edges, uuid_map)

        if self.dedupe_enabled:
            existing_edges = await fetch_entity_edges_by_group(self.group_id)
            logger.info("Resolving against %d existing edges", len(existing_edges))

            resolved_edges, records = self._resolve_edges(
                entity_edges,
                existing_edges,
                episode.uuid,
            )
        else:
            resolved_edges = entity_edges
            records = [{"status": "new", "edge_uuid": e.uuid} for e in entity_edges]

        new_count = sum(1 for r in records if r["status"] == "new")
        merged_count = sum(1 for r in records if r["status"] == "merged")

        metadata = {
            "extracted_count": len(relationships.relationships),
            "built_count": len(entity_edges),
            "resolved_count": len(resolved_edges),
            "new_count": new_count,
            "merged_count": merged_count,
        }

        logger.info(
            "Edge resolution complete: %d edges (%d new, %d merged)",
            metadata["resolved_count"],
            metadata["new_count"],
            metadata["merged_count"],
        )

        return ExtractEdgesOutput(edges=resolved_edges, metadata=metadata)

    def _resolve_edges(
        self,
        candidate_edges: list[EntityEdge],
        existing_edges: dict[str, EntityEdge],
        episode_uuid: str,
    ) -> tuple[list[EntityEdge], list[dict[str, Any]]]:
        """Exact match deduplication only (Option C).

        Key edges by (source_uuid, target_uuid, edge_name).
        If exact match exists: merge episode IDs.
        If new: create new edge.

        DEFERRED FEATURES (see stub methods below):
        - Semantic similarity matching → _semantic_edge_search()
        - LLM-based deduplication → _llm_deduplicate_edges()
        - Contradiction detection → _detect_contradictions()

        Lifted from archive/pipeline_v1/graphiti_pipeline.py lines 650-680.

        Args:
            candidate_edges: Edges to resolve
            existing_edges: Dict of existing edges (uuid → edge)
            episode_uuid: Current episode UUID

        Returns:
            Tuple of (resolved_edges, resolution_records)
        """
        existing_index: dict[tuple[str, str, str], EntityEdge] = {}
        for edge in existing_edges.values():
            key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
            existing_index[key] = edge

        resolved_edges: list[EntityEdge] = []
        records: list[dict[str, Any]] = []

        for edge in candidate_edges:
            key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
            existing = existing_index.get(key)

            if existing:
                updated = existing.model_copy(deep=True)
                existing_episode_ids = set(existing.episodes or [])
                new_episode_ids = set(edge.episodes or [])
                updated.episodes = sorted(existing_episode_ids.union(new_episode_ids))

                resolved_edge = updated
                record = {
                    "status": "merged",
                    "edge_uuid": updated.uuid,
                    "source_uuid": updated.source_node_uuid,
                    "target_uuid": updated.target_node_uuid,
                    "relation": updated.name,
                }
            else:
                edge.episodes = sorted(set((edge.episodes or []) + [episode_uuid]))
                resolved_edge = edge
                record = {
                    "status": "new",
                    "edge_uuid": edge.uuid,
                    "source_uuid": edge.source_node_uuid,
                    "target_uuid": edge.target_node_uuid,
                    "relation": edge.name,
                }

            resolved_edges.append(resolved_edge)
            records.append(record)
            existing_index[key] = resolved_edge

        return resolved_edges, records

    # ========== Deferred Features (Stubs for Future Integration) ==========

    async def _generate_edge_embeddings(
        self,
        edges: list[EntityEdge],
    ) -> list[EntityEdge]:
        """Generate embeddings for edge facts.

        DEFERRED: Requires local embedder (Qwen or similar).

        Integration point for Stage 2b - Semantic Enhancement:
        - Call after edge extraction, before resolution
        - Use graphiti-core's create_entity_edge_embeddings(embedder, edges)
        - Enables semantic search for fuzzy deduplication

        Future implementation:
            from graphiti_core.edges import create_entity_edge_embeddings

            if self.embedder:
                await create_entity_edge_embeddings(self.embedder, edges)
            return edges
        """
        logger.debug("Edge embedding generation deferred (no local embedder)")
        return edges

    async def _semantic_edge_search(
        self,
        candidate_edges: list[EntityEdge],
        existing_edges: dict[str, EntityEdge],
    ) -> dict[str, list[EntityEdge]]:
        """Find semantically similar edges for deduplication.

        DEFERRED: Requires embeddings + hybrid search.

        Integration point for Stage 2b - Semantic Deduplication:
        - Call before _resolve_edges() for fuzzy matching
        - Use graphiti-core's hybrid search + RRF
        - Returns map of edge_uuid → similar_edges

        Future implementation:
            # Hybrid search: embedding similarity + text similarity
            # Combine with reciprocal rank fusion (RRF)
            # Return candidates for LLM deduplication
            pass
        """
        logger.debug("Semantic edge search deferred (no embeddings)")
        return {}

    async def _llm_deduplicate_edges(
        self,
        edge: EntityEdge,
        similar_edges: list[EntityEdge],
    ) -> tuple[EntityEdge, list[EntityEdge]]:
        """Use LLM to deduplicate semantically similar edges.

        DEFERRED: Requires LLM deduplication logic.

        Integration point for Stage 2b - LLM Deduplication:
        - Call for each edge with semantic matches
        - Use graphiti-core's resolve_edge() prompt
        - Returns (resolved_edge, duplicate_edges)

        Future implementation:
            from graphiti_core.prompts.dedupe_edges import resolve_edge

            # LLM determines if edges are duplicates
            # Merges episode IDs if duplicate
            # Returns resolved edge + duplicates to mark
            pass
        """
        logger.debug("LLM edge deduplication deferred")
        return edge, []

    async def _detect_contradictions(
        self,
        resolved_edges: list[EntityEdge],
        existing_edges: dict[str, EntityEdge],
    ) -> list[EntityEdge]:
        """Detect and invalidate contradicting edges using LLM.

        DEFERRED: Requires LLM call (invalidate_edges.v2 prompt).

        Integration point for Stage 2c - Contradiction Detection:
        - Call after edge resolution
        - Uses LLM to determine which edges contradict each other
        - Use graphiti-core's resolve_edge_contradictions()
        - LLM prompt: invalidate_edges.v2() determines contradictions
        - Returns list of edges to mark as expired_at

        Future implementation:
            from graphiti_core.utils.maintenance.edge_operations import (
                resolve_edge_contradictions
            )

            invalidated_edges = []
            edges_by_pair = self._group_edges_by_node_pair(resolved_edges + list(existing_edges.values()))

            for pair_key, edges_in_pair in edges_by_pair.items():
                for edge in edges_in_pair:
                    contradiction_candidates = [
                        e for e in edges_in_pair
                        if e.uuid != edge.uuid and e.name != edge.name
                    ]
                    if contradiction_candidates:
                        # LLM call: Uses invalidate_edges.v2() prompt
                        invalidated = resolve_edge_contradictions(edge, contradiction_candidates)
                        invalidated_edges.extend(invalidated)

            return invalidated_edges
        """
        logger.debug("Contradiction detection deferred (requires LLM call)")
        return []

    def _extract_with_reflexion(
        self,
        episode: EpisodicNode,
        extracted_nodes: list[EntityNode],
        initial_relationships: ExtractedRelationships,
    ) -> ExtractedRelationships:
        """Iteratively improve relationship extraction with reflexion.

        DEFERRED: Reflexion loop requires multiple LLM calls.

        Integration point for Stage 2d - Quality Improvement:
        - Call after initial extraction
        - Use graphiti-core's reflexion pattern (extract_edges.py:137-167)
        - LLM identifies missed relationships, re-extracts
        - Returns enhanced relationships

        Future implementation:
            # Loop up to MAX_REFLEXION_ITERATIONS (e.g., 3)
            # Ask LLM: "What relationships were missed?"
            # Re-extract with additional context
            # Merge with initial_relationships
            pass
        """
        logger.debug("Reflexion loop deferred (single-pass extraction)")
        return initial_relationships


__all__ = [
    "EdgeExtractor",
    "ExtractEdges",
    "ExtractEdgesOutput",
]
