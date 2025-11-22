"""Read operations and queries for the knowledge graph."""

from __future__ import annotations

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodicNode

from backend.database.driver import get_driver
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
    DELETE r
    WITH ent
    OPTIONAL MATCH (ent)<-[remaining:MENTIONS]-()
    WITH ent, count(remaining) as remaining_refs
    WHERE remaining_refs = 0
    DETACH DELETE ent
    RETURN remaining_refs = 0 as was_deleted
    """

    def _locked_query():
        with lock:
            return graph.query(query)

    result = await asyncio.to_thread(_locked_query)

    raw = getattr(result, "_raw_response", None)
    if raw and len(raw) >= 2:
        rows = raw[1]
        if rows and len(rows) > 0:
            return bool(_decode_value(rows[0][0]))
    return False


# Future query operations:
# - Time-range queries
# - Entity timelines
# - Search operations

__all__ = ["get_episode", "get_all_episodes", "delete_entity_mention"]
