"""Entity extraction using DistilBERT NER."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime

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
from graphiti_core.utils.maintenance.edge_operations import (
    filter_existing_duplicate_of_edges,
)

from backend.database.redis_ops import (
    get_suppressed_entities,
    append_unresolved_entities,
    get_unresolved_entities_count,
)
from backend.database.queries import get_entry_suppressed_entities
from backend.ner import predict_entities
from backend.graph.entities_edges import get_type_name_from_ner_label

logger = logging.getLogger(__name__)


def should_use_llm_dedupe(journal: str, existing_entity_count: int) -> bool:
    """Determine whether to use LLM dedupe based on journal state.

    Automatic switching logic:
    - Queue has items -> batch job hasn't run yet -> use queue mode (fast)
    - Queue empty + no entities -> new journal -> use queue mode (accumulate first)
    - Queue empty + has entities -> mature journal -> use LLM per-episode

    This ensures:
    - Bulk imports stay fast (queue mode, defers LLM calls to batch job)
    - After batch job completes, incremental entries get LLM dedupe
    - New journals accumulate entities before expensive LLM calls

    Args:
        journal: Journal name
        existing_entity_count: Number of entities already in graph for this journal

    Returns:
        True if LLM dedupe should be used, False to queue for batch
    """
    queue_count = get_unresolved_entities_count(journal)
    queue_empty = queue_count == 0
    has_entities = existing_entity_count > 0
    return queue_empty and has_entities


@dataclass
class ExtractedEntity:
    """Entity extracted by NER with type classification."""

    name: str
    entity_type: str  # Person, Location, Organization, Miscellaneous


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


def _extract_entities_with_ner(content: str) -> list[ExtractedEntity]:
    """Extract entities from text using DistilBERT NER.

    Args:
        content: Journal entry text

    Returns:
        List of ExtractedEntity with name and type
    """
    raw_entities = predict_entities(content)

    entities = []
    for entity in raw_entities:
        entity_type = get_type_name_from_ner_label(entity["label"])
        entities.append(
            ExtractedEntity(
                name=entity["text"],
                entity_type=entity_type,
            )
        )

    return entities


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

        if not result.result_set:
            return {}

        entities = {}
        for row in result.result_set:
            uuid = _decode_value(row[0]) if len(row) > 0 else None
            name = _decode_value(row[1]) if len(row) > 1 else ""
            summary = _decode_value(row[2]) if len(row) > 2 else ""
            labels = _decode_json(row[3] if len(row) > 3 else None, ["Entity"])
            attributes = _decode_json(row[4] if len(row) > 4 else None, {})
            created_at_raw = _decode_value(row[5]) if len(row) > 5 else None
            group_id_val = _decode_value(row[6]) if len(row) > 6 else group_id

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


async def _collect_candidate_nodes_by_text_search(
    driver, provisional_nodes: list[EntityNode], group_id: str
) -> dict[str, EntityNode]:
    """Collect candidate entities using text search instead of fetching all entities.

    For each provisional entity, queries the database for entities with similar names
    using case-insensitive text matching. This is more efficient than fetching all entities
    when the journal contains many entities.

    Args:
        driver: FalkorDB driver instance
        provisional_nodes: List of newly extracted entities to search for
        group_id: Journal/group ID to search within

    Returns:
        dict[str, EntityNode]: Candidate entities keyed by UUID
    """
    from backend.database.utils import _decode_value, _decode_json, to_cypher_literal
    from backend.database.lifecycle import _ensure_graph

    def _search_sync():
        graph, lock = _ensure_graph(group_id)
        all_candidates = {}

        for node in provisional_nodes:
            search_term = node.name.strip()
            if not search_term:
                continue

            query = f"""
            MATCH (n:Entity)
            WHERE n.group_id = {to_cypher_literal(group_id)}
              AND toLower(n.name) CONTAINS toLower({to_cypher_literal(search_term)})
            RETURN n.uuid, n.name, n.summary, n.labels, n.attributes, n.created_at, n.group_id
            """

            with lock:
                result = graph.query(query)

            if not result.result_set:
                continue

            for row in result.result_set:
                uuid = _decode_value(row[0]) if len(row) > 0 else None
                name = _decode_value(row[1]) if len(row) > 1 else ""
                summary = _decode_value(row[2]) if len(row) > 2 else ""
                labels = _decode_json(row[3] if len(row) > 3 else None, ["Entity"])
                attributes = _decode_json(row[4] if len(row) > 4 else None, {})
                created_at_raw = _decode_value(row[5]) if len(row) > 5 else None
                group_id_val = _decode_value(row[6]) if len(row) > 6 else group_id

                if isinstance(created_at_raw, str):
                    try:
                        created_at = datetime.fromisoformat(created_at_raw)
                    except Exception:
                        created_at = utc_now()
                else:
                    created_at = created_at_raw or utc_now()

                if uuid and uuid not in all_candidates:
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
                    all_candidates[entity.uuid] = entity

        return all_candidates

    return await asyncio.to_thread(_search_sync)


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

    indexes: DedupCandidateIndexes = _build_candidate_indexes(
        list(existing_nodes.values())
    )

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

    duplicate_pairs = await filter_existing_duplicate_of_edges(
        driver, state.duplicate_pairs
    )

    return list(unique_nodes.values()), state.uuid_map, duplicate_pairs


async def extract_nodes(
    episode_uuid: str,
    journal: str,
    entity_types: dict | None = None,
    dedupe_enabled: bool = True,
) -> ExtractNodesResult:
    """Extract and resolve entities from an episode using NER.

    Pipeline:
    1. Fetch episode from database
    2. Extract entities via DistilBERT NER (fast, synchronous)
    3. Filter out suppressed entities (two-tier check)
    4. Clean up entity names
    5. Fetch existing entities for deduplication
    6. Resolve duplicates (MinHash LSH + exact matching)
    7. Queue unresolved entities for batch LLM dedupe
    8. Create MENTIONS edges (episode -> entities)
    9. Persist entities and edges
    10. Return metadata

    Suppression Filtering
    ---------------------
    Extracted entities are filtered against two suppression tiers:
    1. Global (journal-level): Entities suppressed via delete_entity_all_mentions()
       Stored in Redis set: journal:{journal}:suppressed_entities
    2. Entry-level (per-episode): Entities suppressed via delete_entity_mention()
       Stored in Redis hash field: journal:{journal}:{episode_uuid}[suppressed_entities]

    Both tiers use case-insensitive matching (names normalized to lowercase).

    Deduplication Strategy
    ----------------------
    1. DETERMINISTIC PHASE (always runs when dedupe_enabled=True):
       - MinHash/LSH similarity matching (Jaccard >= 0.9)
       - Exact name matching (case-insensitive)
       - High-confidence matches resolved immediately

    2. UNRESOLVED HANDLING:
       - Entities that MinHash can't match are queued to Redis
         (dedup:unresolved:{journal}) for future batch LLM deduplication.

    Args:
        episode_uuid: Episode UUID string
        journal: Journal name
        entity_types: Entity type dict (ignored, NER types are fixed)
        dedupe_enabled: Whether to enable deduplication (default True)

    Returns:
        ExtractNodesResult with extraction metadata

    Raises:
        RuntimeError: If episode not found
    """
    from backend.database.driver import get_driver
    from backend.database.persistence import persist_entities_and_edges
    from backend.utils.node_cleanup import cleanup_entity_name

    driver = get_driver(journal)

    episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    extracted = _extract_entities_with_ner(episode.content)

    logger.info("NER extracted %d entities", len(extracted))

    global_suppressed = get_suppressed_entities(journal)
    try:
        entry_suppressed = await get_entry_suppressed_entities(journal, episode_uuid)
    except Exception as e:
        logger.warning(
            "Failed to get entry-level suppression for %s: %s", episode_uuid, e
        )
        entry_suppressed = set()
    suppressed = global_suppressed | entry_suppressed

    filtered_entities = extracted
    if suppressed:
        original_count = len(extracted)
        filtered_entities = [
            e for e in extracted if e.name.lower() not in suppressed
        ]
        filtered_count = original_count - len(filtered_entities)
        if filtered_count > 0:
            logger.info(
                "Filtered out %d suppressed entities from episode %s",
                filtered_count,
                episode_uuid,
            )

    cleaned_entities = []
    seen_names: set[str] = set()
    for entity in filtered_entities:
        cleaned_name = cleanup_entity_name(entity.name)
        if cleaned_name is None:
            continue
        key = cleaned_name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        cleaned_entities.append(
            ExtractedEntity(
                name=cleaned_name,
                entity_type=entity.entity_type,
            )
        )
    filtered_entities = cleaned_entities

    provisional_nodes = []
    for entity in filtered_entities:
        labels = ["Entity"]
        if entity.entity_type != "Entity":
            labels.append(entity.entity_type)

        node = EntityNode(
            name=entity.name,
            group_id=journal,
            labels=labels,
            summary="",
            created_at=utc_now(),
            name_embedding=[],
        )
        provisional_nodes.append(node)

    existing_nodes = await _collect_candidate_nodes_by_text_search(
        driver, provisional_nodes, journal
    )
    existing_uuid_set = set(existing_nodes.keys())

    logger.info("Resolving against %d existing entities", len(existing_nodes))
    if existing_nodes:
        logger.info(
            "Existing entity names: %s", [n.name for n in existing_nodes.values()]
        )

    nodes, uuid_map, duplicate_pairs = await _resolve_entities(
        provisional_nodes,
        existing_nodes,
        driver,
        dedupe_enabled,
    )

    unresolved_uuids = []
    for prov_node in provisional_nodes:
        canonical_uuid = uuid_map.get(prov_node.uuid, prov_node.uuid)
        if canonical_uuid == prov_node.uuid and canonical_uuid not in existing_uuid_set:
            unresolved_uuids.append(prov_node.uuid)

    if unresolved_uuids:
        use_llm = should_use_llm_dedupe(journal, len(existing_nodes))
        if use_llm:
            # Future: Call LLM dedupe inline
            # For now, still queue - LLM dedupe module not yet built
            append_unresolved_entities(journal, unresolved_uuids)
            logger.info(
                "Queued %d unresolved entities (LLM per-episode mode, pending implementation)",
                len(unresolved_uuids),
            )
        else:
            append_unresolved_entities(journal, unresolved_uuids)
            logger.info(
                "Queued %d unresolved entities for batch dedup", len(unresolved_uuids)
            )

    from backend.database.redis_ops import redis_ops

    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        old_edge_uuids_json = r.hget(cache_key, "mentions_edges")
        if old_edge_uuids_json:
            old_edge_uuids = json.loads(old_edge_uuids_json)
            if old_edge_uuids:
                from backend.database.lifecycle import _ensure_graph
                from backend.database.utils import to_cypher_literal

                def _delete_edges_sync():
                    graph, lock = _ensure_graph(journal)
                    query = f"""
                    MATCH ()-[r:MENTIONS]->()
                    WHERE r.uuid IN {to_cypher_literal(old_edge_uuids)}
                    DELETE r
                    RETURN count(r) as deleted_count
                    """
                    with lock:
                        result = graph.query(query)
                    return result

                await asyncio.to_thread(_delete_edges_sync)
                logger.info(
                    "Deleted %d old MENTIONS edges for episode %s",
                    len(old_edge_uuids),
                    episode_uuid,
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
        episode_uuid=episode_uuid,
    )

    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        nodes_data = []
        for node in nodes:
            most_specific_label = next(
                (l for l in node.labels if l != "Entity"), "Entity"
            )
            nodes_data.append(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "type": most_specific_label,
                }
            )
        r.hset(cache_key, "nodes", json.dumps(nodes_data))

        mentions_edge_uuids = [edge.uuid for edge in episodic_edges]
        r.hset(cache_key, "mentions_edges", json.dumps(mentions_edge_uuids))

    new_entities = sum(1 for node in nodes if node.uuid not in existing_uuid_set)
    exact_matches = len(
        [p for p in duplicate_pairs if p[0].name.lower() == p[1].name.lower()]
    )
    fuzzy_matches = len(
        [p for p in duplicate_pairs if p[0].name.lower() != p[1].name.lower()]
    )

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
    "ExtractNodesResult",
    "extract_nodes",
]
