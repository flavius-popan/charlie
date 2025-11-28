"""Read operations and queries for the knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodicNode

from backend.database.driver import get_driver
from backend.database.redis_ops import redis_ops, add_suppressed_entity, set_episode_status
from backend.settings import DEFAULT_JOURNAL


async def get_episode(episode_uuid: str, journal: str = DEFAULT_JOURNAL) -> dict | None:
    """Retrieve an episode by UUID.

    Args:
        episode_uuid: Episode UUID string
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        Episode data as dict with all fields (uuid, name, content, valid_at, created_at,
        source, source_description, group_id, entity_edges, labels), or None if not found

    Note:
        Uses graphiti-core's EpisodicNode.get_by_uuid for retrieval.
        Returns None for NodeNotFoundError (episode doesn't exist).
        Other exceptions (database errors, etc.) propagate to caller.
    """
    driver = get_driver(journal)
    try:
        episode_node = await EpisodicNode.get_by_uuid(driver, episode_uuid)
        return episode_node.model_dump()
    except NodeNotFoundError:
        return None


async def episode_exists(uuid: str, journal: str = DEFAULT_JOURNAL) -> bool:
    """Check if episode UUID exists in journal."""
    episode = await get_episode(uuid, journal)
    return episode is not None


HOME_PREVIEW_SOURCE_CHARS = 240
HOME_PREVIEW_MAX_LEN = 80


def _parse_valid_at(raw_valid_at):
    """Convert FalkorDB string timestamps back to aware datetimes when possible."""
    if isinstance(raw_valid_at, datetime):
        return raw_valid_at
    if isinstance(raw_valid_at, str):
        try:
            return datetime.fromisoformat(raw_valid_at)
        except ValueError:
            return raw_valid_at
    return raw_valid_at


def _build_preview(snippet: str, max_len: int = HOME_PREVIEW_MAX_LEN) -> str:
    """Create a readable preview from a content snippet.

    Prefers the first Markdown H1 line, otherwise the first sentence. Falls back
    to truncation with an ellipsis to avoid harsh cut-offs.
    """
    if not snippet:
        return ""

    snippet = snippet.replace("\r", "")
    text = snippet.lstrip()
    if not text:
        return ""

    first_newline = text.find("\n")
    first_line = text if first_newline == -1 else text[:first_newline]

    # Prefer a top-level markdown header
    if first_line.startswith("# "):
        candidate = first_line[2:].strip()
        found_sentence = False
    else:
        flat = text.replace("\n", " ").strip()
        sentence_ends = [pos for pos in (flat.find("."), flat.find("!"), flat.find("?")) if pos != -1]
        if sentence_ends:
            end_pos = min(sentence_ends)
            candidate = flat[: end_pos + 1].strip()
            found_sentence = True
        else:
            candidate = flat
            found_sentence = False

    truncated = False
    if len(candidate) > max_len:
        candidate = candidate[:max_len].rstrip()
        truncated = True

    add_ellipsis = False
    if truncated:
        add_ellipsis = True
    elif not found_sentence and not first_line.startswith("# ") and len(snippet.strip()) >= max_len:
        # No sentence end found in the available snippet and it's long enough
        # that we likely chopped the original content.
        add_ellipsis = True

    if add_ellipsis and not candidate.endswith("..."):
        candidate = candidate.rstrip(".!? ") + "..."

    return candidate


async def get_home_screen(journal: str = DEFAULT_JOURNAL) -> list[dict]:
    """Lightweight query for home screen list items.

    Returns uuid, name, valid_at, and a short preview derived from a small
    substring of content to avoid loading full entries.
    """

    driver = get_driver(journal)
    records, _, _ = await driver.execute_query(
        f"""
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.valid_at AS valid_at,
               SUBSTRING(e.content, 0, {HOME_PREVIEW_SOURCE_CHARS}) AS content_preview
        ORDER BY e.valid_at DESC
        """,
        group_id=journal,
    )

    episodes: list[dict] = []
    for record in records:
        valid_at = _parse_valid_at(record.get("valid_at"))
        preview = _build_preview(record.get("content_preview") or "")
        episodes.append(
            {
                "uuid": record.get("uuid"),
                "name": record.get("name"),
                "valid_at": valid_at,
                "preview": preview,
            }
        )

    return episodes


async def delete_entity_mention(
    episode_uuid: str, entity_uuid: str, journal: str = DEFAULT_JOURNAL
) -> bool:
    """Delete MENTIONS edge between episode and entity, and entity if orphaned.

    Args:
        episode_uuid: Episode UUID
        entity_uuid: Entity UUID to remove mention of
        journal: Journal name

    Returns:
        True if entity was fully deleted (orphaned)
        False if only edge removed OR if edge didn't exist

    Note:
        Returns False when the MENTIONS edge doesn't exist. In the UI context
        (deletion triggered from entity list), this edge case shouldn't occur.

    Example:
        # Remove mention and potentially delete entity
        was_deleted = await delete_entity_mention(ep_uuid, entity_uuid)
        if was_deleted:
            # Entity removed from knowledge graph entirely
            pass
        else:
            # Entity still exists in other episodes
            pass
    """
    import asyncio
    from backend.database.driver import get_driver
    from backend.database.utils import to_cypher_literal, _decode_value

    driver = get_driver(journal)
    graph = driver._graph
    lock = driver._lock

    episode_literal = to_cypher_literal(episode_uuid)
    entity_literal = to_cypher_literal(entity_uuid)

    # Query 1: Delete MENTIONS edge and get entity info
    query1 = f"""
    MATCH (ep:Episodic {{uuid: {episode_literal}}})-[r:MENTIONS]->(ent:Entity {{uuid: {entity_literal}}})
    WITH ent.name as entity_name, r.uuid as edge_uuid, r
    DELETE r
    RETURN entity_name, edge_uuid
    """

    def _locked_query1():
        with lock:
            return graph.query(query1)

    result1 = await asyncio.to_thread(_locked_query1)

    # Query 2: Check if entity is orphaned and delete if so
    query2 = f"""
    MATCH (ent:Entity {{uuid: {entity_literal}}})
    OPTIONAL MATCH (ent)<-[remaining:MENTIONS]-()
    WITH ent, count(remaining) as remaining_refs
    WHERE remaining_refs = 0
    DETACH DELETE ent
    RETURN true as was_deleted
    """

    def _locked_query2():
        with lock:
            return graph.query(query2)

    result2 = await asyncio.to_thread(_locked_query2)

    entity_name = None
    edge_uuid = None

    # Parse result from query 1 (MENTIONS edge deletion)
    if result1.result_set and len(result1.result_set) > 0:
        row = result1.result_set[0]
        entity_name = _decode_value(row[0]) if len(row) > 0 else None
        edge_uuid = _decode_value(row[1]) if len(row) > 1 else None

    # Parse result from query 2 (entity deletion if orphaned)
    was_deleted = bool(result2.result_set and len(result2.result_set) > 0)

    def _update_redis_caches():
        """Update Redis caches after entity deletion (synchronous helper for thread offloading)."""
        with redis_ops() as r:
            cache_key = f"journal:{journal}:{episode_uuid}"

            # 1. Update 'nodes' cache (existing logic)
            nodes_json = r.hget(cache_key, "nodes")
            remaining_nodes = []
            if nodes_json:
                nodes = json.loads(nodes_json.decode())
                remaining_nodes = [n for n in nodes if n["uuid"] != entity_uuid]
                r.hset(cache_key, "nodes", json.dumps(remaining_nodes))

            # 2. Update 'mentions_edges' cache
            mentions_json = r.hget(cache_key, "mentions_edges")
            if mentions_json and edge_uuid:
                edge_uuids = json.loads(mentions_json.decode())
                updated_edges = [uuid for uuid in edge_uuids if uuid != edge_uuid]
                r.hset(cache_key, "mentions_edges", json.dumps(updated_edges))

            # 3. Update 'uuid_map' cache
            uuid_map_json = r.hget(cache_key, "uuid_map")
            if uuid_map_json:
                uuid_map = json.loads(uuid_map_json.decode())
                # Remove mappings where canonical UUID = deleted entity
                updated_map = {
                    prov: canon for prov, canon in uuid_map.items()
                    if canon != entity_uuid
                }
                if updated_map != uuid_map:
                    r.hset(cache_key, "uuid_map", json.dumps(updated_map))

            # 4. If no visible entities remain (excluding "I"), set status to "done"
            visible_remaining = [n for n in remaining_nodes if n.get("name") != "I"]
            if not visible_remaining:
                r.hset(cache_key, "status", "done")

            # 5. TODO: Update 'entity_edges' when edge extraction implemented
            # When edge extraction is added, remove RELATES_TO edge UUIDs
            # that involve the deleted entity from the entity_edges list

    # Offload Redis cache updates to background thread
    await asyncio.to_thread(_update_redis_caches)

    # Suppress entity globally in journal
    if entity_name:
        await add_suppressed_entity(journal, entity_name)

    return was_deleted


async def get_entry_entities(
    episode_uuid: str, journal: str = DEFAULT_JOURNAL
) -> list[dict]:
    """Get entities connected to a specific episode from Redis cache.

    Args:
        episode_uuid: Episode UUID to fetch entities for
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        List of entity dicts with 'uuid', 'name', 'type' fields.
        Excludes the "I" (self) entity. Returns empty list if no cache data.
    """
    import asyncio

    cache_key = f"journal:{journal}:{episode_uuid}"

    def _fetch_nodes():
        with redis_ops() as r:
            return r.hget(cache_key, "nodes")

    nodes_json = await asyncio.to_thread(_fetch_nodes)

    if not nodes_json:
        return []

    nodes = json.loads(nodes_json.decode())
    return [n for n in nodes if n.get("name") != "I"]


async def get_period_entities(
    start_date: datetime, end_date: datetime, journal: str = DEFAULT_JOURNAL
) -> dict:
    """Get aggregated entities for a time period.

    Args:
        start_date: Start of period (inclusive)
        end_date: End of period (exclusive)
        journal: Journal name

    Returns:
        Dict with:
        - entry_count: Number of entries in period
        - connection_count: Total entity connections
        - top_entities: List of most frequently mentioned entities
    """
    import asyncio
    from collections import Counter

    driver = get_driver(journal)

    # Query episodes in date range
    records, _, _ = await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.valid_at >= $start_date
          AND e.valid_at < $end_date
        RETURN e.uuid AS uuid
        ORDER BY e.valid_at DESC
        """,
        group_id=journal,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    episode_uuids = [r.get("uuid") for r in records]
    entry_count = len(episode_uuids)

    if entry_count == 0:
        return {"entry_count": 0, "connection_count": 0, "top_entities": []}

    # Fetch entities from Redis cache for each episode
    entity_counts: Counter[str] = Counter()
    entity_info: dict[str, dict] = {}
    total_connections = 0

    def _fetch_all_nodes():
        results = []
        with redis_ops() as r:
            for uuid in episode_uuids:
                cache_key = f"journal:{journal}:{uuid}"
                nodes_json = r.hget(cache_key, "nodes")
                if nodes_json:
                    results.append(json.loads(nodes_json.decode()))
        return results

    all_nodes = await asyncio.to_thread(_fetch_all_nodes)

    for nodes in all_nodes:
        for node in nodes:
            if node.get("name") == "I":
                continue
            total_connections += 1
            name = node.get("name", "")
            entity_counts[name] += 1
            if name not in entity_info:
                entity_info[name] = {
                    "uuid": node.get("uuid"),
                    "name": name,
                    "type": node.get("type", "Entity"),
                }

    # Get top 25 entities by mention frequency
    top_entities = [
        entity_info[name] for name, _ in entity_counts.most_common(25)
    ]

    return {
        "entry_count": entry_count,
        "connection_count": total_connections,
        "top_entities": top_entities,
    }


__all__ = [
    "get_episode",
    "episode_exists",
    "get_home_screen",
    "delete_entity_mention",
    "get_entry_entities",
    "get_period_entities",
]
