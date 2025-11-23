"""Read operations and queries for the knowledge graph."""

from __future__ import annotations

import json

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodicNode

from backend.database.driver import get_driver
from backend.database.redis_ops import redis_ops, add_suppressed_entity
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


async def get_all_episodes(journal: str = DEFAULT_JOURNAL) -> list[dict]:
    """Retrieve all episodes for a journal, ordered by valid_at DESC (newest first).

    Args:
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        List of episode dicts with all fields (uuid, name, content, valid_at, created_at,
        source, source_description, group_id, entity_edges, labels), ordered by valid_at DESC

    Note:
        Uses graphiti-core's EpisodicNode.get_by_group_ids for retrieval.
        The graphiti-core method orders by uuid DESC internally, but we sort by valid_at
        for chronological ordering (newest first).
    """
    driver = get_driver(journal)
    episode_nodes = await EpisodicNode.get_by_group_ids(driver, group_ids=[journal])
    episodes = [node.model_dump() for node in episode_nodes]
    return sorted(episodes, key=lambda ep: ep['valid_at'], reverse=True)


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

    # Delete the MENTIONS edge, then delete entity if orphaned
    query = f"""
    MATCH (ep:Episodic {{uuid: {episode_literal}}})-[r:MENTIONS]->(ent:Entity {{uuid: {entity_literal}}})
    WITH ent.name as entity_name, r.uuid as edge_uuid, r, ent
    DELETE r
    WITH ent, entity_name, edge_uuid
    OPTIONAL MATCH (ent)<-[remaining:MENTIONS]-()
    WITH ent, entity_name, edge_uuid, count(remaining) as remaining_refs
    WHERE remaining_refs = 0
    DETACH DELETE ent
    RETURN entity_name, edge_uuid, remaining_refs = 0 as was_deleted
    """

    def _locked_query():
        with lock:
            return graph.query(query)

    result = await asyncio.to_thread(_locked_query)

    was_deleted = False
    entity_name = None
    edge_uuid = None
    raw = getattr(result, "_raw_response", None)
    if raw and len(raw) >= 2:
        rows = raw[1]
        if rows and len(rows) > 0:
            entity_name = _decode_value(rows[0][0])
            edge_uuid = _decode_value(rows[0][1])
            was_deleted = bool(_decode_value(rows[0][2]))

    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"

        # 1. Update 'nodes' cache (existing logic)
        nodes_json = r.hget(cache_key, "nodes")
        if nodes_json:
            nodes = json.loads(nodes_json.decode())
            updated_nodes = [n for n in nodes if n["uuid"] != entity_uuid]
            r.hset(cache_key, "nodes", json.dumps(updated_nodes))

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

        # 4. TODO: Update 'entity_edges' when edge extraction implemented
        # When edge extraction is added, remove RELATES_TO edge UUIDs
        # that involve the deleted entity from the entity_edges list

    # Suppress entity globally in journal
    if entity_name:
        add_suppressed_entity(journal, entity_name)

    return was_deleted


# Future query operations:
# - Time-range queries
# - Entity timelines
# - Search operations

__all__ = ["get_episode", "get_all_episodes", "delete_entity_mention"]
