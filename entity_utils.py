"""Entity processing utilities using Graphiti conventions."""
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now
from models import Relationships
from settings import GROUP_ID


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
    entity_map: dict[str, EntityNode]
) -> list[EntityEdge]:
    """
    Build EntityEdge objects from relationships.

    Args:
        relationships: Relationships object from DSPy
        entity_map: Dict mapping normalized entity name -> EntityNode

    Returns:
        List of EntityEdge objects (relationships with missing entities are skipped)
    """
    entity_edges = []

    for rel in relationships.items:
        # Look up EntityNodes by normalized name
        source_node = entity_map.get(normalize_entity_name(rel.source))
        target_node = entity_map.get(normalize_entity_name(rel.target))

        if not source_node or not target_node:
            print(f"Warning: Skipping relationship {rel.source} -> {rel.target} (entity not found)")
            continue

        edge = EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name=rel.relation,
            fact=rel.context,
            group_id=GROUP_ID,
            created_at=utc_now(),
            fact_embedding=[],
            episodes=[],
            expired_at=None,
            valid_at=None,
            invalid_at=None,
            attributes={}
        )
        entity_edges.append(edge)

    return entity_edges
