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


def get_journal_cache_key(journal: str, episode_uuid: str) -> str:
    """Get unified cache key for journal episode.

    Args:
        journal: Journal name
        episode_uuid: Episode UUID

    Returns:
        Cache key in format journal:<journal>:<uuid>
    """
    return f"journal:{journal}:{episode_uuid}"


def set_episode_status(
    episode_uuid: str,
    status: str,
    journal: str,
    uuid_map: dict[str, str] | None = None,
) -> None:
    """Set episode processing status.

    Args:
        episode_uuid: Episode UUID
        status: Processing status (e.g., "pending_nodes")
        journal: Journal name (required)
        uuid_map: Optional UUID mapping (provisional -> canonical)

    Note:
        Episode metadata is stored in unified journal cache key.
        Deletion is explicit (when episode is removed from graph).
        This keeps Redis/graph state in sync.
    """
    with redis_ops() as r:
        cache_key = get_journal_cache_key(journal, episode_uuid)
        r.hset(cache_key, "status", status)
        r.hset(cache_key, "journal", journal)

        if uuid_map is not None:
            r.hset(cache_key, "uuid_map", json.dumps(uuid_map))


def get_episode_status(episode_uuid: str, journal: str | None = None) -> str | None:
    """Get episode processing status.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name (optional, scans all journals if not provided)

    Returns:
        Status string or None if episode not found
    """
    if journal is not None:
        with redis_ops() as r:
            cache_key = get_journal_cache_key(journal, episode_uuid)
            status = r.hget(cache_key, "status")
            return status.decode() if status else None

    data = get_episode_data(episode_uuid)
    return data.get("status")


def get_episode_data(episode_uuid: str, journal: str | None = None) -> dict[str, str]:
    """Get all episode metadata.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name (optional, scans all journals if not provided)

    Returns:
        Dictionary of episode metadata fields (empty dict if episode not found)
    """
    with redis_ops() as r:
        if journal is not None:
            cache_key = get_journal_cache_key(journal, episode_uuid)
            data = r.hgetall(cache_key)
            return {k.decode(): v.decode() for k, v in data.items()}

        for key in r.scan_iter(match="journal:*"):
            key_str = key.decode()
            if key_str.endswith(f":{episode_uuid}"):
                data = r.hgetall(key)
                return {k.decode(): v.decode() for k, v in data.items()}

        return {}


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
        for key in r.scan_iter(match="journal:*"):
            ep_status = r.hget(key, "status")
            if ep_status and ep_status.decode() == status:
                episode_uuid = key.decode().split(":")[-1]
                episodes.append(episode_uuid)
        return episodes


def remove_episode_from_queue(episode_uuid: str, journal: str) -> None:
    """Remove episode from processing queue.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name
    """
    with redis_ops() as r:
        cache_key = get_journal_cache_key(journal, episode_uuid)
        r.delete(cache_key)


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
        Background tasks are enqueued with priority=0 (low priority).
    """
    if not get_inference_enabled():
        return 0

    # Import here to avoid circular dependency (tasks.py imports from redis_ops)
    from backend.services.tasks import extract_nodes_task

    pending = get_episodes_by_status("pending_nodes")
    for episode_uuid in pending:
        data = get_episode_data(episode_uuid)
        journal = data.get("journal", "")
        extract_nodes_task(episode_uuid, journal, priority=0)

    return len(pending)


def cleanup_orphaned_episode_keys() -> int:
    """Clean up old episode:* keys from previous architecture.

    Returns:
        Number of keys deleted

    Note:
        This is a migration cleanup for keys from the old dual-key architecture.
        Safe to run multiple times.
    """
    with redis_ops() as r:
        count = 0
        for key in r.scan_iter(match="episode:*"):
            r.delete(key)
            count += 1
        return count


async def cleanup_orphaned_journal_caches(journal: str) -> int:
    """Remove journal cache keys for episodes that don't exist in graph.

    Args:
        journal: Journal name to clean up

    Returns:
        Number of orphaned caches removed

    Note:
        This removes cache entries for deleted episodes.
        Requires graph query, so use sparingly (e.g., on startup or manual trigger).
    """
    from backend.database.queries import get_episode

    with redis_ops() as r:
        count = 0
        pattern = f"journal:{journal}:*"
        for key in r.scan_iter(match=pattern):
            key_str = key.decode()
            episode_uuid = key_str.split(":")[-1]

            episode = await get_episode(episode_uuid, journal)
            if episode is None:
                r.delete(key)
                count += 1

        return count
