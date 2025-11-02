"""Entity processing utilities using Graphiti conventions."""
import logging
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.datetime_utils import utc_now
from datetime import datetime
from models import Relationships
from settings import GROUP_ID

logger = logging.getLogger(__name__)


def normalize_entity_name(name: str) -> str:
    """Use Graphiti's exact normalization: lowercase + collapse whitespace."""
    return _normalize_string_exact(name)


def deduplicate_entities(entity_candidates: list[str]) -> list[str]:
    """
    Deduplicate entity names using Graphiti's normalization.

    Args:
        entity_candidates: List of entity names (may contain duplicates)

    Returns:
        List of unique entity names (preserving original casing)
    """
    unique_entity_names = []
    seen = set()
    for name in entity_candidates:
        key = normalize_entity_name(name)
        if key in seen:
            continue
        seen.add(key)
        unique_entity_names.append(name)  # Keep original casing
    return unique_entity_names


def build_entity_nodes(entity_names: list[str]) -> tuple[list[EntityNode], dict[str, EntityNode]]:
    """
    Build EntityNode objects from entity names.

    Args:
        entity_names: Deduplicated list of entity names

    Returns:
        Tuple of (list of EntityNode objects, dict mapping normalized_name -> EntityNode)
    """
    entity_nodes = []
    entity_map = {}

    for name in entity_names:
        node = EntityNode(
            name=name,
            group_id=GROUP_ID,
            labels=["Entity"],
            name_embedding=[],
            summary="",
            attributes={},
            created_at=utc_now()
        )
        entity_nodes.append(node)
        entity_map[normalize_entity_name(name)] = node

    return entity_nodes, entity_map


def build_entity_edges(
    relationships: Relationships,
    entity_map: dict[str, EntityNode],
    episode_uuid: str
) -> list[EntityEdge]:
    """
    Build EntityEdge objects from relationships.

    Args:
        relationships: Relationships object from DSPy
        entity_map: Dict mapping normalized entity name -> EntityNode
        episode_uuid: UUID of the originating episode (for provenance tracking)

    Returns:
        List of EntityEdge objects (relationships with missing entities are skipped)
    """
    entity_edges = []

    for rel in relationships.items:
        # Look up EntityNodes by normalized name
        source_node = entity_map.get(normalize_entity_name(rel.source))
        target_node = entity_map.get(normalize_entity_name(rel.target))

        if not source_node or not target_node:
            logger.warning(f"Skipping relationship {rel.source} -> {rel.target} (entity not found)")
            continue

        edge = EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name=normalize_edge_name(rel.relation),
            fact=rel.context,
            group_id=GROUP_ID,
            created_at=utc_now(),
            fact_embedding=[],
            episodes=[episode_uuid],
            expired_at=None,
            valid_at=None,
            invalid_at=None,
            attributes={}
        )
        entity_edges.append(edge)

    return entity_edges


def normalize_edge_name(name: str) -> str:
    """
    Normalize relationship name to SCREAMING_SNAKE_CASE.

    Mirrors Graphiti's prompt-based convention (extract_edges.py:26, 120).

    Examples:
        "works at" → "WORKS_AT"
        "knows" → "KNOWS"
        "WorksAt" → "WORKS_AT"
    """
    # Remove extra whitespace
    name = " ".join(name.split())
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Convert to uppercase
    return name.upper()


def build_episodic_node(
    content: str,
    reference_time: datetime,
    group_id: str = "phase1-poc"
) -> EpisodicNode:
    """
    Create EpisodicNode for a journal entry.

    Mirrors graphiti.py:706-720 (EpisodicNode creation).

    Args:
        content: Raw journal entry text
        reference_time: Timestamp when entry was created
        group_id: Graph partition identifier

    Returns:
        EpisodicNode with entity_edges initially empty (populated after edge creation)
    """
    episode = EpisodicNode(
        name=f"Journal Entry {reference_time.isoformat()}",
        group_id=group_id,
        source=EpisodeType.text,  # Journal entries are text type
        source_description="Daily journal entry",
        content=content,
        valid_at=reference_time,
        created_at=utc_now(),
        labels=[],
        entity_edges=[]  # Will be populated after EntityEdge creation
    )
    return episode


def build_episodic_edges(
    episode: EpisodicNode,
    entity_nodes: list
) -> list[EpisodicEdge]:
    """
    Create MENTIONS edges from episode to each entity.

    Mirrors edge_operations.py:51-68 (build_episodic_edges).

    Args:
        episode: The EpisodicNode
        entity_nodes: List of EntityNode objects extracted from episode

    Returns:
        List of EpisodicEdge objects (one per entity)
    """
    episodic_edges = []

    for node in entity_nodes:
        edge = EpisodicEdge(
            source_node_uuid=episode.uuid,  # Episode mentions entity
            target_node_uuid=node.uuid,
            group_id=episode.group_id,
            created_at=utc_now()
        )
        episodic_edges.append(edge)

    return episodic_edges
