"""Huey task queue configuration using FalkorDB's embedded Redis.

Note: The Huey consumer is intended to run inside the Charlie TUI process. Running
an external ``huey_consumer`` process would spin up a separate FalkorDB/Redis
instance and is not supported; start the consumer via Charlie so it shares the
already-initialized embedded Redis.
"""

from __future__ import annotations

import logging
import time
from threading import Thread
from typing import Optional

from huey import PriorityRedisHuey
from huey.consumer import Consumer
from redis import ConnectionPool

from backend.settings import DEFAULT_JOURNAL, HUEY_WORKER_TYPE, HUEY_WORKERS

logger = logging.getLogger(__name__)
for _name in ("huey", "huey.consumer", "huey.api", "huey.signals", "huey.queue"):
    logging.getLogger(_name).setLevel(logging.WARNING)


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


# PriorityRedisHuey supports priority queueing (higher priority = processed first)
huey = PriorityRedisHuey(
    "charlie",
    connection_pool=_get_redis_connection(),
)

logger.info(
    "Huey task queue initialized (worker_type=%s, workers=%d)",
    HUEY_WORKER_TYPE,
    HUEY_WORKERS,
)

# In-process consumer management
class InProcessConsumer(Consumer):
    """Consumer variant that skips installing signal handlers in threads."""

    def _set_signal_handlers(self):
        # Signals cannot be installed from non-main threads; the TUI owns process signals.
        return


_consumer: Optional[Consumer] = None
_consumer_thread: Optional[Thread] = None


def start_huey_consumer() -> None:
    """Start Huey consumer in a background thread (idempotent)."""
    global _consumer, _consumer_thread

    if _consumer_thread and _consumer_thread.is_alive():
        return

    # Ensure the embedded database/Redis is initialized before connecting.
    from backend.database import lifecycle

    lifecycle._ensure_graph(DEFAULT_JOURNAL)

    redis_client = getattr(lifecycle._db, "client", None)
    if redis_client is None:
        raise RuntimeError("Redis client unavailable for Huey consumer startup")

    # Ensure Huey uses the current connection pool (DB may be reinitialized in tests)
    huey.storage.pool = getattr(redis_client, "connection_pool", None)
    huey.storage.conn = redis_client

    # Ensure embedded Redis is reachable before starting the consumer thread
    deadline = time.monotonic() + 5.0
    while True:
        try:
            redis_client.ping()
            huey.storage.conn.ping()
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise RuntimeError("Embedded Redis not ready for Huey consumer startup")
            time.sleep(0.05)

    _consumer = InProcessConsumer(
        huey,
        workers=HUEY_WORKERS,
        worker_type=HUEY_WORKER_TYPE,
        flush_locks=False,
        check_worker_health=False,
    )

    def _run():
        try:
            _consumer.run()
        except Exception:
            logger.exception("Huey consumer stopped unexpectedly")

    _consumer_thread = Thread(target=_run, name="huey-consumer", daemon=True)
    _consumer_thread.start()
    logger.info(
        "Huey consumer started in-process (worker_type=%s, workers=%d)",
        HUEY_WORKER_TYPE,
        HUEY_WORKERS,
    )

    # Kick off orchestrator loop (self-reschedules every few seconds)
    try:
        from backend.services.tasks import orchestrate_inference_work

        orchestrate_inference_work.schedule(delay=0)
    except Exception:
        logger.warning("Failed to schedule orchestrate_inference_work on startup", exc_info=True)


def stop_huey_consumer(timeout: float = 3.0) -> None:
    """Stop the in-process Huey consumer (best-effort, bounded wait)."""
    global _consumer, _consumer_thread

    consumer = _consumer
    thread = _consumer_thread

    _consumer = None
    _consumer_thread = None

    if consumer is None:
        return

    try:
        consumer.stop(graceful=True)
    except Exception:
        logger.warning("Error while stopping Huey consumer", exc_info=True)

    if thread is None:
        return

    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning(
            "Huey consumer thread did not exit within %.1fs; continuing shutdown",
            timeout,
        )


def is_huey_consumer_running() -> bool:
    """Return True if the Huey consumer thread is alive."""
    return bool(_consumer_thread and _consumer_thread.is_alive())
