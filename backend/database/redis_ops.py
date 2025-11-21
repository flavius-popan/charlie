"""Redis operations layer for FalkorDB Lite.

Provides native Redis API access for global metadata, housekeeping, and stats
storage that coexists with graph operations in the same .db file.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

from backend.settings import DEFAULT_JOURNAL


class RedisOpsProxy:
    """Proxy for Redis operations providing direct access to the Redis client."""

    def __init__(self, redis_client):
        self._client = redis_client

    def __getattr__(self, name):
        """Proxy all Redis client attributes and methods."""
        return getattr(self._client, name)


@contextmanager
def redis_ops() -> Iterator[RedisOpsProxy]:
    """
    Get Redis operations client for global application state.

    This provides direct access to Redis operations for storing metadata,
    stats, and housekeeping data that is shared across the entire application.
    All operations are global - there is no automatic key namespacing.

    Usage:
        # Application metadata
        with redis_ops() as r:
            r.set('app:version', '1.0.0')
            r.set('app:last_startup', datetime.now().isoformat())

            version = r.get('app:version')  # b'1.0.0'

        # Statistics tracking
        with redis_ops() as r:
            r.hincrby('stats:global', 'total_entries', 1)
            r.hincrby('stats:global', 'api_calls', 1)
            r.hset('stats:global', 'last_import', '2024-11-18')

            stats = r.hgetall('stats:global')
            # {b'total_entries': b'42', b'api_calls': b'1337', ...}

        # Active journals tracking
        with redis_ops() as r:
            r.sadd('meta:active_journals', 'personal', 'work')
            r.srem('meta:active_journals', 'old_journal')

            active = r.smembers('meta:active_journals')
            # {b'personal', b'work'}

        # Temporary data with expiration
        with redis_ops() as r:
            r.setex('cache:recent_query', 300, 'cached_result')
            ttl = r.ttl('cache:recent_query')  # ~300 seconds

    Yields:
        RedisOpsProxy: Proxy object providing access to Redis client methods.

    Raises:
        RuntimeError: If database is unavailable or shutdown is in progress.

    Note:
        This uses the same Redis instance as the graph operations, so all
        data is stored in the same .db file.

        Recommended key naming conventions:
        - Use prefixes like "app:", "stats:", "meta:" for application data
        - Avoid single-letter keys or numeric-only keys
        - Document your key schema in your application code
        - Use r.keys('*') in development to inspect all keys
    """
    from . import lifecycle

    if lifecycle.is_shutdown_requested():
        raise RuntimeError("Database shutdown in progress")

    lifecycle._ensure_graph(DEFAULT_JOURNAL)

    if lifecycle._db is None:
        raise RuntimeError("Database unavailable")

    redis_client = getattr(lifecycle._db, "client", None)
    if redis_client is None:
        raise RuntimeError("Redis client unavailable")

    proxy = RedisOpsProxy(redis_client)
    yield proxy


# Episode Status Management


def set_episode_status(
    episode_uuid: str,
    status: str,
    journal: str | None = None,
    uuid_map: dict[str, str] | None = None,
) -> None:
    """Set episode processing status.

    Args:
        episode_uuid: Episode UUID
        status: Processing status (e.g., "pending_nodes")
        journal: Journal name (required for initial status set)
        uuid_map: Optional UUID mapping (provisional -> canonical)

    Note:
        Episodes are removed from queue when processing completes.
        This prevents unbounded growth - Redis contains only active queue items.
    """
    with redis_ops() as r:
        episode_key = f"episode:{episode_uuid}"
        r.hset(episode_key, "status", status)

        if journal is not None:
            r.hset(episode_key, "journal", journal)

        if uuid_map is not None:
            r.hset(episode_key, "uuid_map", json.dumps(uuid_map))


def get_episode_status(episode_uuid: str) -> str | None:
    """Get episode processing status.

    Args:
        episode_uuid: Episode UUID

    Returns:
        Status string or None if episode not found
    """
    with redis_ops() as r:
        status = r.hget(f"episode:{episode_uuid}", "status")
        return status.decode() if status else None


def get_episode_data(episode_uuid: str) -> dict[str, str]:
    """Get all episode metadata.

    Args:
        episode_uuid: Episode UUID

    Returns:
        Dictionary of episode metadata fields (empty dict if episode not found)
    """
    with redis_ops() as r:
        data = r.hgetall(f"episode:{episode_uuid}")
        return {k.decode(): v.decode() for k, v in data.items()}


def get_episode_uuid_map(episode_uuid: str) -> dict[str, str] | None:
    """Get parsed UUID mapping for episode.

    Args:
        episode_uuid: Episode UUID

    Returns:
        UUID mapping dict (provisional -> canonical) or None if not set
    """
    data = get_episode_data(episode_uuid)
    uuid_map_str = data.get("uuid_map")
    return json.loads(uuid_map_str) if uuid_map_str else None


def get_episodes_by_status(status: str) -> list[str]:
    """Get all episodes with given status.

    Args:
        status: Processing status to filter by

    Returns:
        List of episode UUIDs
    """
    with redis_ops() as r:
        episodes = []
        for key in r.scan_iter(match="episode:*"):
            ep_status = r.hget(key, "status")
            if ep_status and ep_status.decode() == status:
                episode_uuid = key.decode().split(":", 1)[1]
                episodes.append(episode_uuid)
        return episodes


def remove_episode_from_queue(episode_uuid: str) -> None:
    """Remove episode from processing queue.

    Args:
        episode_uuid: Episode UUID
    """
    with redis_ops() as r:
        r.delete(f"episode:{episode_uuid}")


def get_inference_enabled() -> bool:
    """Get current inference enabled status.

    Returns:
        True if inference enabled, False otherwise (default: True)

    Note:
        Setting is persisted in Redis and survives app restarts.
        Default is True (inference enabled).
    """
    with redis_ops() as r:
        enabled = r.get("app:inference_enabled")
        return enabled.decode() == "true" if enabled else True


def set_inference_enabled(enabled: bool) -> None:
    """Enable/disable inference globally.

    Args:
        enabled: True to enable inference, False to disable

    Note:
        Setting is persisted in Redis and survives app restarts.
        Controls whether new journal entries trigger background extraction tasks.
    """
    with redis_ops() as r:
        r.set("app:inference_enabled", "true" if enabled else "false")


def enqueue_pending_episodes() -> int:
    """Enqueue all pending episodes for processing.

    Returns:
        Number of episodes enqueued

    Note:
        Only enqueues if inference is enabled.
        Safe to call multiple times - tasks check status for idempotency.
    """
    if not get_inference_enabled():
        return 0

    # Import here to avoid circular dependency (tasks.py imports from redis_ops)
    from backend.services.tasks import extract_nodes_task

    pending = get_episodes_by_status("pending_nodes")
    for episode_uuid in pending:
        data = get_episode_data(episode_uuid)
        journal = data.get("journal", "")
        extract_nodes_task(episode_uuid, journal)

    return len(pending)
