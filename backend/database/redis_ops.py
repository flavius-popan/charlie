"""Redis operations layer for FalkorDB Lite.

Provides native Redis API access for global metadata, housekeeping, and stats
storage that coexists with graph operations in the same .db file.
"""

from __future__ import annotations

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
