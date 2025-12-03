"""Read operations and queries for the knowledge graph."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from datetime import datetime

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodicNode

from backend.database.driver import get_driver
from backend.database.redis_ops import (
    redis_ops,
    add_suppressed_entity,
    set_episode_status,
)
from backend.database.utils import to_cypher_literal
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
        sentence_ends = [
            pos for pos in (flat.find("."), flat.find("!"), flat.find("?")) if pos != -1
        ]
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
    elif (
        not found_sentence
        and not first_line.startswith("# ")
        and len(snippet.strip()) >= max_len
    ):
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


async def get_home_screen_paginated(
    journal: str = DEFAULT_JOURNAL,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], bool]:
    """Paginated version of get_home_screen.

    Returns: (episodes, has_more) where episodes has same shape as get_home_screen:
        - uuid: str
        - name: str
        - valid_at: datetime
        - preview: str (truncated content)

    Uses SKIP/LIMIT for offset-based pagination. ORDER BY valid_at DESC, uuid DESC
    ensures deterministic ordering even when multiple entries have identical timestamps.
    """
    driver = get_driver(journal)
    limit_plus_one = limit + 1  # Fetch one extra to detect has_more

    records, _, _ = await driver.execute_query(
        f"""
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.valid_at AS valid_at,
               SUBSTRING(e.content, 0, {HOME_PREVIEW_SOURCE_CHARS}) AS content_preview
        ORDER BY e.valid_at DESC, e.uuid DESC
        SKIP $offset
        LIMIT $limit_plus_one
        """,
        group_id=journal,
        offset=offset,
        limit_plus_one=limit_plus_one,
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

    has_more = len(episodes) > limit
    return (episodes[:limit], has_more)


async def delete_entity_mention(
    episode_uuid: str,
    entity_uuid: str,
    journal: str = DEFAULT_JOURNAL,
    suppress_reextraction: bool = False,
) -> bool:
    """Delete MENTIONS edge between episode and entity, and entity if orphaned.

    Args:
        episode_uuid: Episode UUID
        entity_uuid: Entity UUID to remove mention of
        journal: Journal name
        suppress_reextraction: If True, suppress entity from re-extraction in this entry

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
                    prov: canon
                    for prov, canon in uuid_map.items()
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

    # Suppress entity based on scope
    if suppress_reextraction and entity_name:
        await add_entry_suppressed_entity(journal, episode_uuid, entity_name)

    return was_deleted


async def delete_entity_all_mentions(
    entity_uuid: str,
    journal: str,
    suppress_reextraction: bool = False,
) -> tuple[str, int]:
    """Delete entity and ALL its MENTIONS edges across the journal.

    Args:
        entity_uuid: Entity UUID to delete
        journal: Journal name
        suppress_reextraction: If True, suppress entity from re-extraction globally

    Returns:
        (entity_name, deleted_edge_count)
    """
    from backend.database.driver import get_driver
    from backend.database.utils import to_cypher_literal, _decode_value

    driver = get_driver(journal)
    graph = driver._graph
    lock = driver._lock

    entity_literal = to_cypher_literal(entity_uuid)

    # Query 1: Get entity name, all affected episode UUIDs, and all MENTIONS edge UUIDs
    query1 = f"""
    MATCH (ent:Entity {{uuid: {entity_literal}}})
    OPTIONAL MATCH (ep:Episodic)-[r:MENTIONS]->(ent)
    WITH ent.name as entity_name, collect(DISTINCT ep.uuid) as episode_uuids, collect(r.uuid) as edge_uuids
    RETURN entity_name, episode_uuids, edge_uuids
    """

    def _locked_query1():
        with lock:
            return graph.query(query1)

    result1 = await asyncio.to_thread(_locked_query1)

    entity_name = None
    episode_uuids = []
    edge_uuids = []

    if result1.result_set and len(result1.result_set) > 0:
        row = result1.result_set[0]
        entity_name = _decode_value(row[0]) if len(row) > 0 else None
        episode_uuids = _decode_value(row[1]) if len(row) > 1 else []
        edge_uuids = _decode_value(row[2]) if len(row) > 2 else []

    # Query 2: DETACH DELETE the entity (removes entity + all MENTIONS + all RELATES_TO edges)
    query2 = f"""
    MATCH (ent:Entity {{uuid: {entity_literal}}})
    DETACH DELETE ent
    RETURN true as deleted
    """

    def _locked_query2():
        with lock:
            return graph.query(query2)

    await asyncio.to_thread(_locked_query2)

    # Clean up Redis caches for each affected episode
    def _update_redis_caches():
        """Update Redis caches for all affected episodes."""
        with redis_ops() as r:
            for episode_uuid in episode_uuids:
                cache_key = f"journal:{journal}:{episode_uuid}"

                # 1. Update 'nodes' cache
                nodes_json = r.hget(cache_key, "nodes")
                remaining_nodes = []
                if nodes_json:
                    nodes = json.loads(nodes_json.decode())
                    remaining_nodes = [n for n in nodes if n["uuid"] != entity_uuid]
                    r.hset(cache_key, "nodes", json.dumps(remaining_nodes))

                # 2. Update 'mentions_edges' cache
                mentions_json = r.hget(cache_key, "mentions_edges")
                if mentions_json and edge_uuids:
                    edge_uuid_list = json.loads(mentions_json.decode())
                    # Remove all edge UUIDs that were deleted
                    updated_edges = [uuid for uuid in edge_uuid_list if uuid not in edge_uuids]
                    r.hset(cache_key, "mentions_edges", json.dumps(updated_edges))

                # 3. Update 'uuid_map' cache
                uuid_map_json = r.hget(cache_key, "uuid_map")
                if uuid_map_json:
                    uuid_map = json.loads(uuid_map_json.decode())
                    updated_map = {
                        prov: canon
                        for prov, canon in uuid_map.items()
                        if canon != entity_uuid
                    }
                    if updated_map != uuid_map:
                        r.hset(cache_key, "uuid_map", json.dumps(updated_map))

                # 4. Update episode status if no visible entities remain
                visible_remaining = [n for n in remaining_nodes if n.get("name") != "I"]
                if not visible_remaining:
                    r.hset(cache_key, "status", "done")

    await asyncio.to_thread(_update_redis_caches)

    # Add to suppression list if requested
    if suppress_reextraction and entity_name:
        await add_suppressed_entity(journal, entity_name)

    deleted_edge_count = len(edge_uuids) if edge_uuids else 0
    return (entity_name or "", deleted_edge_count)


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
    cache_key = f"journal:{journal}:{episode_uuid}"

    def _fetch_nodes():
        with redis_ops() as r:
            return r.hget(cache_key, "nodes")

    nodes_json = await asyncio.to_thread(_fetch_nodes)

    if not nodes_json:
        return []

    nodes = json.loads(nodes_json.decode())
    return [n for n in nodes if n.get("name") != "I"]


async def get_entry_entities_with_counts(
    episode_uuid: str, journal: str = DEFAULT_JOURNAL
) -> list[dict]:
    """Get entities for an episode with their global mention counts.

    Like get_entry_entities but adds 'mention_count' field showing how many
    episodes mention each entity across the journal. Useful for filtering
    low-value entities that only appear once.

    Args:
        episode_uuid: Episode UUID to fetch entities for
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        List of entity dicts with 'uuid', 'name', 'type', 'mention_count' fields.
        Excludes the "I" (self) entity. Returns empty list if no cache data.
    """
    # First get entities from cache
    entities = await get_entry_entities(episode_uuid, journal)
    if not entities:
        return []

    # Get mention counts from graph
    driver = get_driver(journal)
    entity_uuids = [e["uuid"] for e in entities]

    query = f"""
    UNWIND {to_cypher_literal(entity_uuids)} AS uuid
    MATCH (e:Entity {{uuid: uuid, group_id: {to_cypher_literal(journal)}}})
    OPTIONAL MATCH (ep:Episodic)-[:MENTIONS]->(e)
    RETURN e.uuid AS uuid, count(DISTINCT ep) AS mention_count
    """

    records, _, _ = await driver.execute_query(query)

    # Build uuid -> count map
    counts = {r["uuid"]: r["mention_count"] for r in records}

    # Enrich entities with counts
    for entity in entities:
        entity["mention_count"] = counts.get(entity["uuid"], 0)

    return entities


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

    # Get top 100 entities by mention frequency
    top_entities = [entity_info[name] for name, _ in entity_counts.most_common(100)]

    return {
        "entry_count": entry_count,
        "connection_count": total_connections,
        "top_entities": top_entities,
    }


ENTITY_BROWSER_QUOTE_MAX_CHARS = 220


def truncate_quote(
    content: str, max_chars: int = ENTITY_BROWSER_QUOTE_MAX_CHARS
) -> str:
    """Extract first sentence or truncate at word boundary."""
    if not content:
        return ""
    # Strip markdown headers
    content = re.sub(r"^#+\s+.*$", "", content, flags=re.MULTILINE).strip()
    if not content:
        return ""
    # Find first sentence
    match = re.search(r"^[^.!?]*[.!?]", content)
    if match and len(match.group()) <= max_chars:
        return match.group()
    # Truncate at word boundary
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars].rsplit(" ", 1)[0]
    return truncated + "..."


def extract_entity_snippet(
    content: str, entity_name: str, max_chars: int = ENTITY_BROWSER_QUOTE_MAX_CHARS
) -> str:
    """Extract a snippet centered on the entity mention.

    Finds the entity name in the content and extracts surrounding context.
    Handles possessives ('s) and case-insensitive matching.
    """
    if not content or not entity_name:
        return truncate_quote(content, max_chars)

    # Strip markdown headers and normalize whitespace
    clean = re.sub(r"^#+\s+.*$", "", content, flags=re.MULTILINE)
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return ""

    # Build pattern for entity name (case-insensitive, handle possessives)
    escaped_name = re.escape(entity_name)
    pattern = rf"\b{escaped_name}(?:'s|'s)?\b"

    match = re.search(pattern, clean, re.IGNORECASE)
    if not match:
        # Entity not found, fall back to truncate
        return truncate_quote(content, max_chars)

    # Extract context around the match
    start_pos = match.start()
    end_pos = match.end()

    # Calculate how much context we can show on each side
    context_chars = (max_chars - (end_pos - start_pos)) // 2

    # Find snippet boundaries
    snippet_start = max(0, start_pos - context_chars)
    snippet_end = min(len(clean), end_pos + context_chars)

    # Adjust to word boundaries
    if snippet_start > 0:
        # Find next space after snippet_start
        space_pos = clean.find(" ", snippet_start)
        if space_pos != -1 and space_pos < start_pos:
            snippet_start = space_pos + 1

    if snippet_end < len(clean):
        # Find last space before snippet_end
        space_pos = clean.rfind(" ", end_pos, snippet_end)
        if space_pos != -1:
            snippet_end = space_pos

    snippet = clean[snippet_start:snippet_end].strip()

    # Add ellipsis if truncated
    prefix = "..." if snippet_start > 0 else ""
    suffix = "..." if snippet_end < len(clean) else ""

    return f"{prefix}{snippet}{suffix}"


ENTITY_QUOTE_TARGET_LENGTH = 220


def extract_entity_sentences(
    content: str,
    entity_name: str,
    max_sentences: int = 3,
    target_length: int = ENTITY_QUOTE_TARGET_LENGTH,
) -> list[str]:
    """Extract all sentences mentioning the entity from content.

    Returns complete sentences containing the entity name.
    Handles possessives ('s) and case-insensitive matching.
    Pads shorter sentences with trailing context to reach target_length.
    """
    if not content or not entity_name:
        return []

    # Strip markdown headers and normalize whitespace
    clean = re.sub(r"^#+\s+.*$", "", content, flags=re.MULTILINE)
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return []

    # Split into sentences (handle ., !, ? followed by space or end)
    sentences = re.split(r"(?<=[.!?])\s+", clean)

    # Build pattern for entity name (case-insensitive, handle possessives)
    escaped_name = re.escape(entity_name)
    pattern = rf"\b{escaped_name}(?:'s|'s)?\b"

    # Find sentences containing the entity, with their indices
    matching = []
    for i, sentence in enumerate(sentences):
        if re.search(pattern, sentence, re.IGNORECASE):
            sentence = sentence.strip()
            # Strip leading non-alphabetical characters (bullets, dashes, etc.)
            sentence = re.sub(r"^[^a-zA-Z]+", "", sentence)
            if sentence and len(sentence) > 10:  # Skip very short fragments
                # Pad short sentences with trailing context
                if len(sentence) < target_length:
                    padded = sentence
                    j = i + 1
                    while len(padded) < target_length and j < len(sentences):
                        next_sent = sentences[j].strip()
                        # Also strip leading non-alpha from continuation
                        next_sent = re.sub(r"^[^a-zA-Z]+", "", next_sent)
                        if next_sent:
                            padded = padded + " " + next_sent
                        j += 1
                    # Truncate if we went over, add ellipsis
                    if len(padded) > target_length:
                        padded = padded[:target_length].rsplit(" ", 1)[0] + "..."
                    matching.append(padded)
                else:
                    matching.append(sentence)

    return matching[:max_sentences]


async def get_entity_browser_data(
    entity_uuid: str, journal: str = DEFAULT_JOURNAL
) -> dict:
    """Single function returning everything needed for the entity browser.

    Returns:
        Dict with:
        - entity: {uuid, name, first_mention, last_mention, entry_count}
        - entries: [{episode_uuid, content, valid_at}]  # newest first
        - connections: [{uuid, name, count, sample_content}]
    """
    driver = get_driver(journal)

    # Query 1: Get entity details + all episodes mentioning it
    episodes_query = f"""
    MATCH (e:Entity {{uuid: {to_cypher_literal(entity_uuid)}, group_id: {to_cypher_literal(journal)}}})
    MATCH (ep:Episodic)-[:MENTIONS]->(e)
    RETURN e.name as name, ep.uuid as episode_uuid, ep.content as content, ep.valid_at as valid_at
    ORDER BY ep.valid_at DESC
    """

    episodes_records, _, _ = await driver.execute_query(episodes_query)

    # Parse episode results
    entity_name = None
    entries = []
    for record in episodes_records:
        if entity_name is None:
            entity_name = record.get("name")
        valid_at = _parse_valid_at(record.get("valid_at"))
        entries.append(
            {
                "episode_uuid": record.get("episode_uuid"),
                "content": record.get("content"),
                "valid_at": valid_at,
            }
        )

    # Query 2: Get co-occurring entities (connections)
    connections_query = f"""
    MATCH (ep:Episodic)-[:MENTIONS]->(e:Entity {{uuid: {to_cypher_literal(entity_uuid)}}})
    MATCH (ep)-[:MENTIONS]->(other:Entity)
    WHERE other.uuid <> {to_cypher_literal(entity_uuid)}
      AND other.name <> 'I'
    WITH other, count(DISTINCT ep) as co_count, collect(ep.content)[0] as sample
    RETURN other.uuid as uuid, other.name as name, co_count, sample
    ORDER BY co_count DESC
    LIMIT 20
    """

    connections_records, _, _ = await driver.execute_query(connections_query)

    connections = []
    for record in connections_records:
        connections.append(
            {
                "uuid": record.get("uuid"),
                "name": record.get("name"),
                "count": record.get("co_count"),
                "sample_content": record.get("sample"),
            }
        )

    # Build entity summary
    first_mention = entries[-1]["valid_at"] if entries else None
    last_mention = entries[0]["valid_at"] if entries else None

    return {
        "entity": {
            "uuid": entity_uuid,
            "name": entity_name or "",
            "first_mention": first_mention,
            "last_mention": last_mention,
            "entry_count": len(entries),
        },
        "entries": entries,
        "connections": connections,
    }


async def get_n_plus_one_neighbors(
    source_uuids: list[str],
    journal: str = DEFAULT_JOURNAL,
    exclude_uuids: set[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Find entities that co-occur with source entities (1-hop neighbors).

    Used by entity browser to expand exploration beyond direct connections.
    When reading an entry, the entities in that entry become sources, and
    this query finds what those entities connect to - enabling discovery
    of bridging entities that link multiple topics.

    Ranking prioritizes entities that connect to MORE of the sources
    (higher intersection score), with ties broken by oldest co-occurrence
    to surface forgotten connections (inverse recency for rediscovery).

    Args:
        source_uuids: Entity UUIDs to find neighbors for
        journal: Journal name
        exclude_uuids: UUIDs to exclude from results (e.g. visited entities)
        limit: Max results to return

    Returns:
        List of dicts with 'uuid', 'name', 'intersection_count', 'oldest_cooccurrence'
    """
    if not source_uuids:
        return []

    exclude_uuids = exclude_uuids or set()
    driver = get_driver(journal)

    # Build exclusion list including sources themselves
    all_excluded = set(source_uuids) | exclude_uuids

    query = f"""
    UNWIND {to_cypher_literal(source_uuids)} AS source_uuid
    MATCH (source:Entity {{uuid: source_uuid, group_id: {to_cypher_literal(journal)}}})
    MATCH (ep:Episodic)-[:MENTIONS]->(source)
    MATCH (ep)-[:MENTIONS]->(neighbor:Entity)
    WHERE neighbor.uuid <> source_uuid
      AND neighbor.name <> 'I'
      AND NOT neighbor.uuid IN {to_cypher_literal(list(all_excluded))}
    WITH neighbor, source_uuid, min(ep.valid_at) as oldest_with_source
    WITH neighbor, count(DISTINCT source_uuid) as intersection_count, min(oldest_with_source) as oldest_cooccurrence
    RETURN neighbor.uuid as uuid, neighbor.name as name, intersection_count, oldest_cooccurrence
    ORDER BY intersection_count DESC, oldest_cooccurrence ASC
    LIMIT {limit}
    """

    records, _, _ = await driver.execute_query(query)

    neighbors = []
    for record in records:
        neighbors.append(
            {
                "uuid": record.get("uuid"),
                "name": record.get("name"),
                "intersection_count": record.get("intersection_count"),
                "oldest_cooccurrence": _parse_valid_at(record.get("oldest_cooccurrence")),
            }
        )

    return neighbors


async def add_entry_suppressed_entity(
    journal: str, episode_uuid: str, entity_name: str
) -> None:
    """Add entity to entry-level suppression list.

    Args:
        journal: Journal name
        episode_uuid: Episode UUID
        entity_name: Entity name to suppress (normalized to lowercase)

    Note:
        Entry-level suppression is specific to a single entry, unlike
        journal-level suppression which affects all future entries.
        Uses the existing episode hash at journal:{journal}:{uuid} with
        a 'suppressed_entities' field containing a JSON array of names.
    """
    def _sync_add():
        with redis_ops() as r:
            cache_key = f"journal:{journal}:{episode_uuid}"
            normalized_name = entity_name.lower()

            # Get existing suppressed entities or start with empty list
            existing_json = r.hget(cache_key, "suppressed_entities")
            if existing_json:
                suppressed = json.loads(existing_json.decode())
            else:
                suppressed = []

            # Add if not already present
            if normalized_name not in suppressed:
                suppressed.append(normalized_name)
                r.hset(cache_key, "suppressed_entities", json.dumps(suppressed))

    await asyncio.to_thread(_sync_add)


async def get_entry_suppressed_entities(journal: str, episode_uuid: str) -> set[str]:
    """Get all suppressed entity names for an entry.

    Args:
        journal: Journal name
        episode_uuid: Episode UUID

    Returns:
        Set of suppressed entity names (lowercase), empty set if none
    """
    def _sync_get():
        with redis_ops() as r:
            cache_key = f"journal:{journal}:{episode_uuid}"
            suppressed_json = r.hget(cache_key, "suppressed_entities")
            if suppressed_json:
                return set(json.loads(suppressed_json.decode()))
            return set()

    return await asyncio.to_thread(_sync_get)


async def is_entity_suppressed(episode_uuid: str, journal: str, entity_name: str) -> bool:
    """Check if entity is suppressed at entry OR journal level.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name
        entity_name: Entity name to check

    Returns:
        True if entity is suppressed at entry or journal level, False otherwise

    Note:
        Checks entry-level suppression first (more specific), then falls back
        to journal-level check.
    """
    from backend.database.redis_ops import get_suppressed_entities

    normalized_name = entity_name.lower()

    # Check entry-level suppression first
    entry_suppressed = await get_entry_suppressed_entities(journal, episode_uuid)
    if normalized_name in entry_suppressed:
        return True

    # Fall back to journal-level suppression (sync function, run in thread)
    journal_suppressed = await asyncio.to_thread(get_suppressed_entities, journal)
    return normalized_name in journal_suppressed


__all__ = [
    "get_episode",
    "episode_exists",
    "get_home_screen",
    "get_home_screen_paginated",
    "delete_entity_mention",
    "delete_entity_all_mentions",
    "get_entry_entities",
    "get_entry_entities_with_counts",
    "get_period_entities",
    "get_entity_browser_data",
    "get_n_plus_one_neighbors",
    "add_entry_suppressed_entity",
    "get_entry_suppressed_entities",
    "is_entity_suppressed",
    "truncate_quote",
    "extract_entity_snippet",
]
