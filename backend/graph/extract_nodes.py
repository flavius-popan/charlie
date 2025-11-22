"""Entity extraction DSPy module and orchestrator."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

import dspy
from pydantic import BaseModel, model_validator
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EpisodicEdge
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import filter_existing_duplicate_of_edges

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Named entity with type classification."""

    name: str
    entity_type_id: int


class ExtractedEntities(BaseModel):
    """Collection of extracted entities."""

    extracted_entities: list[ExtractedEntity]

    @model_validator(mode="before")
    @classmethod
    def coerce_list(cls, data):
        """Allow direct list input for DSPy compatibility."""
        if isinstance(data, list):
            return {"extracted_entities": data}
        return data


class EntityExtractionSignature(dspy.Signature):
    """Extract meaningful entities from a personal journal entry."""

    episode_content: str = dspy.InputField(desc="personal journal entry text")
    entity_types: str = dspy.InputField(desc="available entity type definitions with IDs")
    extracted_entities: ExtractedEntities = dspy.OutputField(
        desc="entities found in the journal entry, each with name and most specific type ID"
    )


class EntityExtractor(dspy.Module):
    """LLM-based entity extraction module.

    Configured via dspy.context(lm=...) by caller.
    Optimizable with DSPy teleprompters.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(self, episode_content: str, entity_types: str) -> ExtractedEntities:
        """Extract entities from text.

        Args:
            episode_content: Journal entry text
            entity_types: JSON string of entity type definitions

        Returns:
            ExtractedEntities containing extracted entities
        """
        result = self.extractor(episode_content=episode_content, entity_types=entity_types)
        return result.extracted_entities


@dataclass
class ExtractNodesResult:
    """Extraction metadata returned by extract_nodes()."""

    episode_uuid: str
    extracted_count: int
    resolved_count: int
    new_entities: int
    exact_matches: int
    fuzzy_matches: int
    entity_uuids: list[str]
    uuid_map: dict[str, str]


async def _fetch_existing_entities(driver, group_id: str) -> dict[str, EntityNode]:
    """Fetch all entities by manually querying and parsing with _decode_json.

    This avoids the graphiti-core bug where FalkorDB attributes come back as lists.
    """
    from backend.database.utils import _decode_value, _decode_json, to_cypher_literal
    from backend.database.lifecycle import _ensure_graph

    def _fetch_sync():
        graph, lock = _ensure_graph(group_id)
        query = f"""
        MATCH (n:Entity)
        WHERE n.group_id = {to_cypher_literal(group_id)}
        RETURN n.uuid, n.name, n.summary, n.labels, n.attributes, n.created_at, n.group_id
        """

        with lock:
            result = graph.query(query)

        rows = getattr(result, "_raw_response", None)
        if not rows or len(rows) < 2:
            return {}

        entities = {}
        for row in rows[1]:  # Skip header row
            uuid = _decode_value(row[0][1]) if len(row) > 0 else None
            name = _decode_value(row[1][1]) if len(row) > 1 else ""
            summary = _decode_value(row[2][1]) if len(row) > 2 else ""
            labels = _decode_json(row[3][1] if len(row) > 3 else None, ["Entity"])
            attributes = _decode_json(row[4][1] if len(row) > 4 else None, {})
            created_at_raw = _decode_value(row[5][1]) if len(row) > 5 else None
            group_id_val = _decode_value(row[6][1]) if len(row) > 6 else group_id

            if isinstance(created_at_raw, str):
                try:
                    created_at = datetime.fromisoformat(created_at_raw)
                except Exception:
                    created_at = utc_now()
            else:
                created_at = created_at_raw or utc_now()

            if uuid:
                entity = EntityNode(
                    uuid=str(uuid),
                    name=str(name),
                    summary=str(summary),
                    labels=labels if isinstance(labels, list) else ["Entity"],
                    attributes=attributes if isinstance(attributes, dict) else {},
                    created_at=created_at,
                    group_id=str(group_id_val),
                    name_embedding=[],
                )
                entities[entity.uuid] = entity

        return entities

    return await asyncio.to_thread(_fetch_sync)


def _resolve_exact_names(
    provisional_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
) -> None:
    """Case-insensitive exact matching (runs after fuzzy)."""
    for idx, node in enumerate(provisional_nodes):
        if state.resolved_nodes[idx] is not None:
            continue
        normalized = _normalize_string_exact(node.name)
        if not normalized:
            continue
        candidates = indexes.normalized_existing.get(normalized)
        if not candidates:
            continue
        canonical = candidates[0]
        logger.debug("Exact name resolved %s -> %s", node.name, canonical.uuid)
        state.resolved_nodes[idx] = canonical
        state.uuid_map[node.uuid] = canonical.uuid
        if canonical.uuid != node.uuid:
            state.duplicate_pairs.append((node, canonical))
        try:
            state.unresolved_indices.remove(idx)
        except ValueError:
            pass


async def _resolve_entities(
    provisional_nodes: list[EntityNode],
    existing_nodes: dict[str, EntityNode],
    driver,
    dedupe_enabled: bool,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Resolve using graphiti-core dedup helpers."""
    if not dedupe_enabled:
        uuid_map = {node.uuid: node.uuid for node in provisional_nodes}
        return provisional_nodes, uuid_map, []

    indexes: DedupCandidateIndexes = _build_candidate_indexes(list(existing_nodes.values()))

    state = DedupResolutionState(
        resolved_nodes=[None] * len(provisional_nodes),
        uuid_map={},
        unresolved_indices=[],
        duplicate_pairs=[],
    )

    _resolve_with_similarity(provisional_nodes, indexes, state)
    _resolve_exact_names(provisional_nodes, indexes, state)

    for idx in state.unresolved_indices:
        node = provisional_nodes[idx]
        state.resolved_nodes[idx] = node
        state.uuid_map[node.uuid] = node.uuid

    unique_nodes: dict[str, EntityNode] = {}
    for node in state.resolved_nodes:
        if node is None:
            continue
        if node.uuid not in unique_nodes:
            unique_nodes[node.uuid] = node

    duplicate_pairs = await filter_existing_duplicate_of_edges(driver, state.duplicate_pairs)

    return list(unique_nodes.values()), state.uuid_map, duplicate_pairs


async def extract_nodes(
    episode_uuid: str,
    journal: str,
    entity_types: dict | None = None,
    dedupe_enabled: bool = True,
) -> ExtractNodesResult:
    """Extract and resolve entities from existing episode.

    Pipeline:
    1. Fetch episode from database
    2. Check if LLM is configured (skip if not)
    3. Extract entities via EntityExtractor
    4. Fetch existing entities for deduplication
    5. Resolve duplicates (MinHash LSH + exact matching)
    6. Create MENTIONS edges (episode -> entities)
    7. Persist entities and edges (using NullEmbedder)
    8. Store uuid_map in episode.attributes
    9. Return metadata

    LLM Configuration:
        Uses dspy.settings.lm for LLM access. If not configured, raises RuntimeError.

    Args:
        episode_uuid: Episode UUID string
        journal: Journal name
        entity_types: Entity type dict (defaults to backend.graph.entities_edges.entity_types)
        dedupe_enabled: Whether to enable deduplication (default True)

    Returns:
        ExtractNodesResult with extraction metadata

    Raises:
        RuntimeError: If LLM is not configured or episode not found
    """
    from backend.database.driver import get_driver
    from backend.database.persistence import persist_entities_and_edges
    from backend.graph.entities_edges import entity_types as default_entity_types, format_entity_types_for_llm, get_type_name_from_id

    if entity_types is None:
        entity_types = default_entity_types

    if dspy.settings.lm is None:
        raise RuntimeError("No LLM configured. Set dspy.settings.lm before calling extract_nodes()")

    driver = get_driver(journal)

    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)

    entity_types_json = format_entity_types_for_llm(entity_types)

    extractor = EntityExtractor()
    extracted = extractor(
        episode_content=episode.content,
        entity_types=entity_types_json,
    )

    logger.info("Extracted %d provisional entities", len(extracted.extracted_entities))

    provisional_nodes = []
    for entity in extracted.extracted_entities:
        type_name = get_type_name_from_id(entity.entity_type_id, entity_types)
        labels = ["Entity"]
        if type_name != "Entity":
            labels.append(type_name)

        node = EntityNode(
            name=entity.name,
            group_id=journal,
            labels=labels,
            summary="",
            created_at=utc_now(),
            name_embedding=[],
        )
        provisional_nodes.append(node)

    existing_nodes = await _fetch_existing_entities(driver, journal)
    existing_uuid_set = set(existing_nodes.keys())

    logger.info("Resolving against %d existing entities", len(existing_nodes))
    if existing_nodes:
        logger.info("Existing entity names: %s", [n.name for n in existing_nodes.values()])

    nodes, uuid_map, duplicate_pairs = await _resolve_entities(
        provisional_nodes,
        existing_nodes,
        driver,
        dedupe_enabled,
    )

    episodic_edges = []
    for node in nodes:
        edge = EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,
            group_id=journal,
            created_at=utc_now(),
            fact_embedding=[],
        )
        episodic_edges.append(edge)

    await persist_entities_and_edges(
        nodes=nodes,
        edges=[],
        episodic_edges=episodic_edges,
        journal=journal,
    )

    from backend.database.redis_ops import redis_ops
    import json

    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        nodes_data = []
        for node in nodes:
            most_specific_label = node.labels[-1] if len(node.labels) > 1 else "Entity"
            nodes_data.append({
                "uuid": node.uuid,
                "name": node.name,
                "type": most_specific_label,
            })
        r.hset(cache_key, "nodes", json.dumps(nodes_data))

    new_entities = sum(1 for node in nodes if node.uuid not in existing_uuid_set)
    exact_matches = len([p for p in duplicate_pairs if p[0].name.lower() == p[1].name.lower()])
    fuzzy_matches = len([p for p in duplicate_pairs if p[0].name.lower() != p[1].name.lower()])

    logger.info(
        "Resolution complete: %d nodes (%d exact, %d fuzzy, %d new)",
        len(nodes),
        exact_matches,
        fuzzy_matches,
        new_entities,
    )

    return ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=len(provisional_nodes),
        resolved_count=len(nodes),
        new_entities=new_entities,
        exact_matches=exact_matches,
        fuzzy_matches=fuzzy_matches,
        entity_uuids=[node.uuid for node in nodes],
        uuid_map=uuid_map,
    )


__all__ = [
    "ExtractedEntity",
    "ExtractedEntities",
    "EntityExtractionSignature",
    "EntityExtractor",
    "ExtractNodesResult",
    "extract_nodes",
]
