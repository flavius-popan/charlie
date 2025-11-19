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


# Future query operations:
# - Time-range queries
# - Entity timelines
# - Search operations

__all__ = ["get_episode", "get_all_episodes"]
