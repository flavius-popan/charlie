"""Huey task queue configuration using FalkorDB's embedded Redis."""

from __future__ import annotations

import logging

from huey import RedisHuey
from redis import ConnectionPool

from backend.settings import DEFAULT_JOURNAL, HUEY_WORKER_TYPE, HUEY_WORKERS

logger = logging.getLogger(__name__)


def _get_redis_connection() -> ConnectionPool:
    """Extract connection pool from FalkorDB for Huey task queue.

    Returns:
        ConnectionPool from the embedded FalkorDB Redis instance

    Note:
        This reuses the same Redis instance as the graph operations,
        avoiding file locking conflicts and ensuring data consistency.
        The database must be initialized before calling this function.
    """
    from backend.database import lifecycle

    lifecycle._ensure_graph(DEFAULT_JOURNAL)

    if lifecycle._db is None:
        raise RuntimeError("Database not initialized")

    redis_client = getattr(lifecycle._db, "client", None)
    if redis_client is None:
        raise RuntimeError("Redis client unavailable")

    connection_pool = getattr(redis_client, "connection_pool", None)
    if connection_pool is None:
        raise RuntimeError("Redis connection pool unavailable")

    logger.info("Extracted Redis connection pool from FalkorDB")
    return connection_pool


# RedisHuey supports async tasks via asyncio integration
huey = RedisHuey(
    "charlie",
    connection_pool=_get_redis_connection(),
)

logger.info(
    "Huey task queue initialized (worker_type=%s, workers=%d)",
    HUEY_WORKER_TYPE,
    HUEY_WORKERS,
)
