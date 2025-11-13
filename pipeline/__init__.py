"""Modular Graphiti pipeline stages for local MLX inference.

Each module represents a discrete stage in the knowledge graph ingestion pipeline:
- extract_nodes: Entity extraction and resolution
- (Future) extract_edges: Relationship extraction
- (Future) extract_attributes: Entity attribute enrichment
- (Future) generate_summaries: Entity summarization
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import (
    get_default_group_id,
    validate_excluded_entity_types,
    validate_group_id,
)
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges

from .extract_nodes import ExtractNodes, ExtractNodesOutput
from .extract_edges import ExtractEdges, ExtractEdgesOutput
from .extract_attributes import ExtractAttributes, ExtractAttributesOutput
from .generate_summaries import GenerateSummaries, GenerateSummariesOutput
from .entity_edge_models import entity_types, edge_types, edge_type_map
from .db_utils import write_episode_and_nodes

logger = logging.getLogger(__name__)

# ========== Pipeline Orchestration ==========


@dataclass
class AddJournalResults:
    """Results from processing a journal entry through the pipeline."""

    episode: EpisodicNode
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    episodic_edges: list[EpisodicEdge]
    uuid_map: dict[str, str]
    metadata: dict[str, Any]


async def add_journal(
    content: str,
    group_id: str | None = None,
    reference_time: datetime | None = None,
    name: str | None = None,
    source_description: str = "Journal entry",
    entity_types: dict | None = entity_types,
    excluded_entity_types: list[str] | None = None,
    extract_nodes_factory: Callable[[str], ExtractNodes] | None = None,
    edge_types: dict | None = edge_types,
    edge_type_map: dict | None = edge_type_map,
    persist: bool = True,
) -> AddJournalResults:
    """Process a journal entry and extract knowledge graph elements.

    Entry point for the Graphiti pipeline. Accepts plain text journal content
    and orchestrates all pipeline stages to extract entities, relationships,
    and other graph elements.

    Analogous to graphiti-core's add_episode() but optimized for local MLX inference
    with DSPy modules.

    Args:
        content: Journal entry text
        group_id: Graph partition identifier (defaults to FalkorDB default '\\_')
        reference_time: When entry was written (defaults to now)
        name: Episode identifier (defaults to generated name)
        source_description: Description of entry source
        entity_types: Custom entity type schemas
        excluded_entity_types: Entity types to exclude from extraction
        extract_nodes_factory: Optional hook returning an ExtractNodes module for the given group_id
        edge_types: Custom edge type schemas (for future Stage 2 enhancement)
        edge_type_map: Maps (source_label, target_label) to allowed edge types
        persist: Whether to write results to FalkorDB (defaults to True)

    Returns:
        AddJournalResults with episode, extracted nodes, UUID mappings, and metadata

    Example:
        ```python
        import asyncio
        from pipeline import add_journal

        async def main():
            result = await add_journal(
                content="Today I met with Sarah at Stanford...",
                group_id="user_123"
            )
            print(f"Episode: {result.episode.uuid}")
            print(f"Extracted {len(result.nodes)} entities")
            print(f"Extracted {len(result.edges)} relationships")

        asyncio.run(main())
        ```
    """
    # Validate inputs using graphiti-core validators
    validate_entity_types(entity_types)
    validate_excluded_entity_types(excluded_entity_types, entity_types)
    validate_group_id(group_id)

    # Use FalkorDB default group_id if not provided
    group_id = group_id or get_default_group_id(GraphProvider.FALKORDB)

    # Stage 1: Extract and resolve entity nodes
    def _default_extract_nodes_factory(gid: str) -> ExtractNodes:
        return ExtractNodes(group_id=gid, dedupe_enabled=True)

    factory = extract_nodes_factory or _default_extract_nodes_factory
    extractor = factory(group_id)
    extract_result = await extractor(
        content=content,
        reference_time=reference_time,
        name=name,
        source_description=source_description,
        entity_types=entity_types,
    )

    # Stage 2: Extract relationships between entities
    edge_extractor = ExtractEdges(
        group_id=group_id,
        dedupe_enabled=True,
        edge_types=edge_types,
        edge_type_map=edge_type_map,
    )
    extract_edges_result = await edge_extractor(
        episode=extract_result.episode,
        extracted_nodes=extract_result.extracted_nodes,
        resolved_nodes=extract_result.nodes,
        uuid_map=extract_result.uuid_map,
        previous_episodes=extract_result.previous_episodes,
        entity_types=entity_types,
    )

    # Stage 3: Extract entity attributes
    attribute_extractor = ExtractAttributes(group_id=group_id)
    attributes_result = await attribute_extractor(
        nodes=extract_result.nodes,
        episode=extract_result.episode,
        previous_episodes=extract_result.previous_episodes,
        entity_types=entity_types,
    )

    # Stage 4: Generate entity summaries
    summary_generator = GenerateSummaries(group_id=group_id)
    summaries_result = await summary_generator(
        nodes=attributes_result.nodes,
        episode=extract_result.episode,
        previous_episodes=extract_result.previous_episodes,
    )

    # Stage 5: Save to FalkorDB (episode, nodes, edges, episodic edges)
    episodic_edges = build_episodic_edges(
        summaries_result.nodes,
        extract_result.episode.uuid,
        extract_result.episode.created_at,
    )
    extract_result.episode.entity_edges = [
        edge.uuid for edge in extract_edges_result.edges
    ]

    persistence_result: dict[str, Any]
    if persist:
        persistence_result = await write_episode_and_nodes(
            episode=extract_result.episode,
            nodes=summaries_result.nodes,
            edges=extract_edges_result.edges,
            episodic_edges=episodic_edges,
        )

        if "error" in persistence_result:
            error_message = persistence_result.get("error", "Unknown error")
            logger.error("Persistence failed: %s", error_message)
            raise RuntimeError(f"Database persistence failed: {error_message}")

        logger.info(
            "Persisted episode %s (%d nodes, %d edges)",
            extract_result.episode.uuid,
            persistence_result.get("nodes_created", 0),
            persistence_result.get("edges_created", 0),
        )
    else:
        persistence_result = {"status": "skipped"}

    return AddJournalResults(
        episode=extract_result.episode,
        nodes=summaries_result.nodes,
        edges=extract_edges_result.edges,
        episodic_edges=episodic_edges,
        uuid_map=extract_result.uuid_map,
        metadata={
            **extract_result.metadata,
            "edges": extract_edges_result.metadata,
            "attributes": attributes_result.metadata,
            "summaries": summaries_result.metadata,
            "persistence": persistence_result,
        },
    )


# ========== Module Exports ==========

__all__ = [
    "add_journal",
    "AddJournalResults",
    "ExtractNodes",
    "ExtractNodesOutput",
    "ExtractEdges",
    "ExtractEdgesOutput",
    "ExtractAttributes",
    "ExtractAttributesOutput",
    "GenerateSummaries",
    "GenerateSummariesOutput",
]
