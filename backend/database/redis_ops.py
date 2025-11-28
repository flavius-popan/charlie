"""Redis operations layer for FalkorDB Lite.

Provides native Redis API access for global metadata, housekeeping, and stats
storage that coexists with graph operations in the same .db file.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

import logging

logger = logging.getLogger(__name__)

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
            if r.type(cache_key) != b"hash":
                return {}
            data = r.hgetall(cache_key)
            return {k.decode(): v.decode() for k, v in data.items()}

        for key in r.scan_iter(match="journal:*"):
            key_str = key.decode()
            if key_str.endswith(":suppressed_entities"):
                continue
            if r.type(key) != b"hash":
                continue
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
            key_str = key.decode()
            if key_str.endswith(":suppressed_entities"):
                continue
            if r.type(key) != b"hash":
                continue
            ep_status = r.hget(key, "status")
            if ep_status and ep_status.decode() == status:
                episode_uuid = key_str.split(":")[-1]
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
    """Enqueue all pending episodes for processing in reverse chronological order.

    Returns:
        Number of episodes enqueued

    Note:
        Only enqueues if inference is enabled.
        Episodes are processed newest-first (by valid_at) for better UX.
        Huey's unique=True on extract_nodes_task handles deduplication.
        Background tasks are enqueued with priority=0 (low priority).
    """
    if not get_inference_enabled():
        return 0

    # Import here to avoid circular dependency (tasks.py imports from redis_ops)
    from backend.services.tasks import extract_nodes_task

    count = 0
    for journal in get_journals_with_pending_episodes():
        pending = get_pending_episodes(journal)  # Already sorted newest-first
        for episode_uuid in pending:
            extract_nodes_task(episode_uuid, journal, priority=0)
            count += 1

    return count


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
            if key_str.endswith(":suppressed_entities"):
                continue
            episode_uuid = key_str.split(":")[-1]

            episode = await get_episode(episode_uuid, journal)
            if episode is None:
                r.delete(key)
                count += 1

        return count


# Entity Suppression Management


async def add_suppressed_entity(journal: str, entity_name: str) -> None:
    """Add entity to global suppression list for journal.

    Args:
        journal: Journal name
        entity_name: Entity name to suppress (will be normalized to lowercase)

    Note:
        Entity name is normalized to lowercase for case-insensitive matching.
        Suppressed entities will be filtered out during extraction.
        Uses Redis Set for atomic operations and better performance.
        This function runs synchronous Redis I/O on a background thread.
    """
    import asyncio

    def _sync_add():
        with redis_ops() as r:
            key = f"journal:{journal}:suppressed_entities"
            normalized_name = entity_name.lower()
            r.sadd(key, normalized_name)

    await asyncio.to_thread(_sync_add)


def get_suppressed_entities(journal: str) -> set[str]:
    """Get set of suppressed entity names for journal.

    Args:
        journal: Journal name

    Returns:
        Set of suppressed entity names (lowercase)
    """
    with redis_ops() as r:
        key = f"journal:{journal}:suppressed_entities"
        members = r.smembers(key)
        return {m.decode() for m in members} if members else set()


def remove_suppressed_entity(journal: str, entity_name: str) -> bool:
    """Remove entity from global suppression list for journal.

    Args:
        journal: Journal name
        entity_name: Entity name to un-suppress (will be normalized to lowercase)

    Returns:
        True if entity was in suppression list and removed, False otherwise

    Note:
        For future un-suppression feature.
    """
    with redis_ops() as r:
        key = f"journal:{journal}:suppressed_entities"
        normalized_name = entity_name.lower()
        removed_count = r.srem(key, normalized_name)
        return removed_count > 0


# Active Episode Tracking


def set_active_episode(episode_uuid: str, journal: str) -> None:
    """Mark episode as actively being processed by Huey worker.

    Args:
        episode_uuid: Episode UUID currently being processed
        journal: Journal name

    Note:
        Only one episode can be active at a time (single worker).
        Called at start of extract_nodes_task actual processing.
    """
    with redis_ops() as r:
        r.hset("task:active_episode", mapping={"uuid": episode_uuid, "journal": journal})


def clear_active_episode() -> None:
    """Clear the active episode marker.

    Note:
        Called when task completes (success, failure, or cancellation).
        Safe to call even if no episode is active.
    """
    with redis_ops() as r:
        r.delete("task:active_episode")


def is_episode_actively_processing(episode_uuid: str) -> bool:
    """Check if a specific episode is the one currently being processed.

    Args:
        episode_uuid: Episode UUID to check

    Returns:
        True if this episode is the active one, False otherwise
    """
    with redis_ops() as r:
        data = r.hgetall("task:active_episode")
        if not data:
            return False
        try:
            active_uuid = data.get(b"uuid", b"").decode()
            return active_uuid == episode_uuid
        except (UnicodeDecodeError, AttributeError):
            # Corrupted data - clear it
            r.delete("task:active_episode")
            return False


def get_active_episode_uuid() -> str | None:
    """Get UUID of the currently processing episode.

    Returns:
        Episode UUID if one is being processed, None otherwise
    """
    with redis_ops() as r:
        data = r.hgetall("task:active_episode")
        if not data:
            return None
        try:
            uuid = data.get(b"uuid", b"").decode()
            return uuid if uuid else None
        except (UnicodeDecodeError, AttributeError):
            r.delete("task:active_episode")
            return None


def get_processing_status(journal: str) -> dict:
    """Get processing status for home screen polling.

    Returns active episode UUID, queue count, and model state in a single call.

    Args:
        journal: Journal name

    Returns:
        Dict with:
        - 'active_uuid' (str|None): Currently processing episode UUID
        - 'pending_count' (int): Number of episodes in queue
        - 'model_state' (str): "idle", "loading", "inferring", or "unloading"
        - 'inference_enabled' (bool): Whether inference is enabled
    """
    with redis_ops() as r:
        # Get active episode
        active_uuid = None
        data = r.hgetall("task:active_episode")
        if data:
            try:
                active_uuid = data.get(b"uuid", b"").decode() or None
            except (UnicodeDecodeError, AttributeError):
                r.delete("task:active_episode")

        # Get pending count
        pending_count = r.zcard(f"pending:nodes:{journal}")

        # Get model state (uses staleness detection)
        model_state = _get_model_state_internal(r)

        # Get inference enabled flag
        inference_enabled = r.get("app:inference_enabled")
        inference_enabled = inference_enabled != b"false" if inference_enabled else True

        return {
            "active_uuid": active_uuid,
            "pending_count": pending_count,
            "model_state": model_state,
            "inference_enabled": inference_enabled,
        }


# =============================================================================
# Model State Tracking (for UI display)
# =============================================================================
#
# Tracks whether the LLM model is currently loading or running inference.
# This enables the UI to show different states:
# - "Loading model..." during cold start
# - "Extracting: <entry>" during inference
#
# Key: task:model_state (Redis Hash)
# Fields: state ("loading" | "inferring"), started_at (timestamp)
#
# Robustness features:
# - Staleness detection: auto-clear state older than TTL (crash recovery)
# - TTL safety net: key auto-expires even if staleness check not called
# - Episode info from task:active_episode (no duplication)
# =============================================================================

MODEL_STATE_TTL_SECONDS = 300  # 5 minutes max before considered stale


def set_model_state(state: str) -> None:
    """Set current model state for UI display.

    Args:
        state: One of "idle", "loading", "inferring", "unloading"
               "idle" deletes the key; others set state with timestamp
    """
    with redis_ops() as r:
        if state == "idle":
            r.delete("task:model_state")
        else:
            r.hset(
                "task:model_state",
                mapping={"state": state, "started_at": str(time.time())},
            )
            r.expire("task:model_state", MODEL_STATE_TTL_SECONDS)


def _get_model_state_internal(r) -> str:
    """Get model state with staleness detection (internal, takes redis client).

    Returns "idle" if not set or stale (crash recovery).
    """
    data = r.hgetall("task:model_state")
    if not data:
        return "idle"

    state = data.get(b"state", b"idle").decode()

    # Check for stale state (process may have crashed)
    started_at = data.get(b"started_at")
    if started_at:
        try:
            elapsed = time.time() - float(started_at.decode())
            if elapsed > MODEL_STATE_TTL_SECONDS:
                logger.warning("Model state stale (%.0fs), clearing", elapsed)
                r.delete("task:model_state")
                return "idle"
        except (ValueError, UnicodeDecodeError):
            pass

    return state


def get_model_state() -> dict:
    """Get current model state with staleness detection.

    Returns {"state": "idle"} if not set or stale (crash recovery).
    """
    with redis_ops() as r:
        return {"state": _get_model_state_internal(r)}


def clear_model_state() -> None:
    """Clear model state (called on task completion or error)."""
    with redis_ops() as r:
        r.delete("task:model_state")


def clear_transient_state() -> None:
    """Clear all transient processing state on startup (crash recovery).

    Call this on app initialization to ensure clean state after crashes.
    Preserves user data (pending queue) and preferences (inference_enabled).
    """
    with redis_ops() as r:
        r.delete("task:model_state")      # Model loading/inferring state
        r.delete("task:active_episode")   # Currently processing episode
    logger.info("Cleared transient processing state on startup")


# =============================================================================
# Unresolved Entities Queue (for Batch LLM Dedup)
# =============================================================================
#
# Entities that MinHash deduplication couldn't confidently match are queued
# here for future batch LLM deduplication. This enables fast bulk imports
# by deferring expensive LLM calls.
#
# Queue: dedup:unresolved:{journal} (Redis List, per-journal)
# Storage: UUIDs only - entity data fetched from graph when processing
#
# BATCH JOB CONTRACT (future implementation):
# 1. Pop UUIDs via pop_unresolved_entities()
# 2. Fetch entity data from graph (handle missing entities gracefully)
# 3. Sort by name for small-LLM context efficiency
# 4. Run DSPy LLM dedupe to identify duplicate groups
# 5. For duplicates: merge in graph (redirect edges, delete duplicate node)
# 6. For unique entities: no action needed (already in graph, just processed)
#
# Mode switching is automatic via should_use_llm_dedupe() in extract_nodes.py:
# - Queue has items → queue mode (batch pending)
# - Queue empty + has entities → LLM per-episode mode
# =============================================================================


def append_unresolved_entities(journal: str, entity_uuids: list[str]) -> None:
    """Append unresolved entity UUIDs to batch dedup queue.

    Unresolved entities are those that MinHash deduplication couldn't match
    to existing entities. They become new entities in the graph but are also
    queued here for a future batch LLM dedup job to review cross-episode
    matches (e.g., "Sarah" from Episode 1 and "Sarah Chen" from Episode 50).

    Args:
        journal: Journal name
        entity_uuids: List of entity UUIDs (entity data is in the graph)
    """
    if not entity_uuids:
        return
    with redis_ops() as r:
        key = f"dedup:unresolved:{journal}"
        r.rpush(key, *entity_uuids)


def pop_unresolved_entities(journal: str, count: int = 100) -> list[str]:
    """Pop unresolved entity UUIDs from batch dedup queue.

    Args:
        journal: Journal name
        count: Max entities to pop

    Returns:
        List of entity UUIDs
    """
    with redis_ops() as r:
        key = f"dedup:unresolved:{journal}"
        uuids = []
        for _ in range(count):
            item = r.lpop(key)
            if item is None:
                break
            uuids.append(item.decode() if isinstance(item, bytes) else item)
        return uuids


def get_unresolved_entities_count(journal: str) -> int:
    """Get count of unresolved entities in queue.

    Args:
        journal: Journal name

    Returns:
        Number of entities in queue
    """
    with redis_ops() as r:
        return r.llen(f"dedup:unresolved:{journal}")


# Pending Episodes Queue (Chronologically Ordered)


def add_pending_episode(episode_uuid: str, journal: str, valid_at) -> None:
    """Add episode to pending queue sorted by valid_at timestamp.

    Uses a Redis Sorted Set to maintain chronological order (oldest first).
    This ensures entities are extracted progressively, improving dedup quality.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name
        valid_at: Episode reference time (datetime object)
    """
    with redis_ops() as r:
        key = f"pending:nodes:{journal}"
        score = valid_at.timestamp()
        r.zadd(key, {episode_uuid: score})


def get_pending_episodes(journal: str) -> list[str]:
    """Get pending episodes in reverse chronological order (newest first).

    Args:
        journal: Journal name

    Returns:
        List of episode UUIDs sorted by valid_at (newest first)
    """
    with redis_ops() as r:
        key = f"pending:nodes:{journal}"
        return [uuid.decode() for uuid in r.zrevrange(key, 0, -1)]


def remove_pending_episode(episode_uuid: str, journal: str) -> None:
    """Remove episode from pending queue.

    Args:
        episode_uuid: Episode UUID
        journal: Journal name
    """
    with redis_ops() as r:
        key = f"pending:nodes:{journal}"
        r.zrem(key, episode_uuid)


def get_pending_episodes_count(journal: str) -> int:
    """Get count of pending episodes for a journal.

    Args:
        journal: Journal name

    Returns:
        Number of pending episodes
    """
    with redis_ops() as r:
        return r.zcard(f"pending:nodes:{journal}")


def get_journals_with_pending_episodes() -> list[str]:
    """Get list of journals that have pending episodes.

    Returns:
        List of journal names with at least one pending episode
    """
    with redis_ops() as r:
        journals = []
        for key in r.scan_iter(match="pending:nodes:*"):
            if r.zcard(key) > 0:
                journal = key.decode().split(":")[-1]
                journals.append(journal)
        return journals
