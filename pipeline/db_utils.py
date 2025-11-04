"""Async database utilities for pipeline modules.

Wraps sync FalkorDB operations to prevent blocking during I/O.
"""

import asyncio
from datetime import datetime

from falkordb_utils import (
    fetch_entities_by_group as _sync_fetch_entities,
    fetch_recent_episodes as _sync_fetch_recent_episodes,
)
from graphiti_core.nodes import EntityNode, EpisodicNode


async def fetch_entities_by_group(group_id: str) -> dict[str, EntityNode]:
    """Fetch all entities in group (async)."""
    return await asyncio.to_thread(_sync_fetch_entities, group_id)


async def fetch_recent_episodes(
    group_id: str,
    reference_time: datetime,
    limit: int,
) -> list[EpisodicNode]:
    """Fetch recent episodes for context (async)."""
    return await asyncio.to_thread(
        _sync_fetch_recent_episodes, group_id, reference_time, limit
    )


__all__ = ["fetch_entities_by_group", "fetch_recent_episodes"]
