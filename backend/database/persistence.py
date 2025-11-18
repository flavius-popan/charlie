"""Episode and entity persistence operations."""

from __future__ import annotations

import asyncio
import json
import logging
from threading import Lock
from typing import Any

from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now

from backend.database.driver import FalkorLiteDriver
from backend.database.lifecycle import _ensure_graph, is_shutdown_requested
from backend.database.utils import (
    SELF_ENTITY_LABELS,
    SELF_ENTITY_NAME,
    SELF_UUID_LITERAL,
    _merge_episode_sync,
    to_cypher_literal,
)
from backend.settings import DEFAULT_JOURNAL

logger = logging.getLogger(__name__)

# Per-journal initialization tracking
_graph_initialized: dict[str, bool] = {}
_graph_init_lock = None
_seeded_self_groups: set[str] = set()
_self_seed_lock = None

# Threading lock to protect asyncio.Lock creation
_asyncio_lock_creation_lock = Lock()


def _ensure_asyncio_lock(lock_var: asyncio.Lock | None) -> asyncio.Lock:
    """Safely create or reuse an asyncio.Lock with event loop binding check.

    Uses double-checked locking to prevent race conditions when multiple tasks
    try to create the lock simultaneously. Also handles event loop rebinding
    if the lock was created on a different event loop.

    Thread safety: Uses threading.Lock to protect asyncio.Lock creation since
    this function may be called from multiple asyncio tasks concurrently.

    Args:
        lock_var: Existing asyncio.Lock or None

    Returns:
        Valid asyncio.Lock bound to current event loop
    """
    current_loop = asyncio.get_running_loop()

    # Fast path: lock exists and is bound to current loop
    if lock_var is not None and getattr(lock_var, '_loop', None) == current_loop:
        return lock_var

    # Slow path: need to create or rebind lock
    with _asyncio_lock_creation_lock:
        # Re-check after acquiring threading lock (double-checked locking)
        if lock_var is None or getattr(lock_var, '_loop', None) != current_loop:
            return asyncio.Lock()
        return lock_var


async def ensure_graph_ready(journal: str = DEFAULT_JOURNAL, *, delete_existing: bool = False) -> None:
    """Build Graphiti indices/constraints for a journal's graph.

    Thread safety: Uses asyncio.Lock to ensure indices are built exactly once
    per journal. Multiple concurrent calls for the same journal will serialize,
    with the first call building indices and subsequent calls skipping work.

    Args:
        journal: Journal name (graph to initialize)
        delete_existing: Whether to delete existing indices first
    """
    global _graph_initialized, _graph_init_lock, _seeded_self_groups

    if delete_existing:
        _graph_initialized[journal] = False
        _seeded_self_groups.discard(journal)

    if _graph_initialized.get(journal, False) and not delete_existing:
        return

    # Ensure we have a valid asyncio.Lock for this event loop
    _graph_init_lock = _ensure_asyncio_lock(_graph_init_lock)

    async with _graph_init_lock:
        if _graph_initialized.get(journal, False) and not delete_existing:
            return

        driver = FalkorLiteDriver(journal=journal)
        try:
            await driver.build_indices_and_constraints(delete_existing=delete_existing)
        except ImportError as exc:
            logger.warning("Skipping index bootstrap: %s", exc)

        _graph_initialized[journal] = True


def _merge_self_entity_sync(graph, journal: str, name: str) -> None:
    """Synchronous SELF entity merge for use in asyncio.to_thread.

    Args:
        graph: FalkorDB graph instance
        journal: Journal name
        name: Name for the SELF entity
    """
    now_literal = to_cypher_literal(utc_now().isoformat())
    summary_literal = to_cypher_literal(
        "Represents the journal author for first-person perspective anchoring."
    )
    labels_literal = to_cypher_literal(json.dumps(SELF_ENTITY_LABELS))
    attributes_literal = to_cypher_literal(json.dumps({}))

    query = f"""
    MERGE (self:Entity:Person {{uuid: {SELF_UUID_LITERAL}}})
    SET self.name = {to_cypher_literal(name)},
        self.group_id = COALESCE(self.group_id, {to_cypher_literal(journal)}),
        self.labels = {labels_literal},
        self.summary = CASE
            WHEN self.summary = '' OR self.summary IS NULL THEN {summary_literal}
            ELSE self.summary
        END,
        self.attributes = CASE
            WHEN self.attributes = '' OR self.attributes IS NULL THEN {attributes_literal}
            ELSE self.attributes
        END,
        self.created_at = COALESCE(self.created_at, {now_literal})
    RETURN self.uuid
    """

    try:
        graph.query(query)
    except Exception as exc:
        logger.exception("Failed to seed SELF entity")
        raise RuntimeError(f"Failed to seed SELF entity for journal '{journal}'") from exc


async def ensure_self_entity(journal: str, name: str = SELF_ENTITY_NAME) -> None:
    """Seed the deterministic SELF entity for this journal if missing.

    Uses double-checked locking pattern for thread safety:
    1. Fast-path check without lock (racy read - optimization)
    2. Acquire asyncio.Lock
    3. Re-check condition inside lock (prevents duplicate work)
    4. Perform state modification inside lock

    The initial fast-path check is a racy read without synchronization.
    This is safe because:
    - False negatives (thinks not seeded when it is): Acquire lock unnecessarily,
      but re-check inside lock prevents duplicate work
    - False positives (thinks seeded when it's not): Cannot occur - once an entry
      is added to the set, it's never removed, and Python set membership tests
      are atomic for existing members

    Thread safety: The final `_seeded_self_groups.add()` happens inside the
    asyncio.Lock, ensuring only one task seeds the entity per journal.

    Args:
        journal: Journal name
        name: Name for the SELF entity (defaults to SELF_ENTITY_NAME)
    """
    if journal in _seeded_self_groups:  # Racy optimization - see docstring
        return

    # Ensure we have a valid asyncio.Lock for this event loop
    global _self_seed_lock
    _self_seed_lock = _ensure_asyncio_lock(_self_seed_lock)

    async with _self_seed_lock:
        if journal in _seeded_self_groups:
            return

        graph, lock = _ensure_graph(journal)

        # Use lock to serialize access to this journal's graph
        def _locked_merge():
            with lock:
                _merge_self_entity_sync(graph, journal, name)

        await asyncio.to_thread(_locked_merge)
        _seeded_self_groups.add(journal)


async def ensure_database_ready(journal: str) -> None:
    """Ensure database is initialized and SELF entity exists for this journal.

    Args:
        journal: Journal name
    """
    await ensure_graph_ready(journal=journal)
    await ensure_self_entity(journal)


async def persist_episode(episode: EpisodicNode, journal: str) -> None:
    """Persist an episode to FalkorDB.

    Thread safety: Uses per-journal locks to serialize writes to the same journal.
    Multiple journals can be written to concurrently. The actual database write
    happens inside asyncio.to_thread() with a threading.Lock to protect the
    FalkorDB graph instance.

    Args:
        episode: The EpisodicNode to persist
        journal: Journal name (graph to persist to)

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If persistence fails or shutdown is in progress
    """
    # Fail-fast if shutdown requested (no waiting - immediate rejection)
    if is_shutdown_requested():
        raise RuntimeError("Database shutdown in progress - cannot persist episode")

    await ensure_database_ready(journal)

    graph, lock = _ensure_graph(journal)

    try:
        # Convert episode to dict for persistence
        episode_dict = {
            'uuid': episode.uuid,
            'name': episode.name,
            'group_id': episode.group_id,
            'content': episode.content,
            'source': episode.source,
            'source_description': episode.source_description,
            'valid_at': episode.valid_at,
            'created_at': episode.created_at,
            'entity_edges': episode.entity_edges,
            'labels': episode.labels,
        }

        # Use lock to serialize access to this journal's graph
        def _locked_merge():
            with lock:
                _merge_episode_sync(graph, episode_dict)

        await asyncio.to_thread(_locked_merge)
        logger.info("Persisted episode %s to journal %s", episode.uuid, journal)

    except Exception as exc:
        logger.exception("Failed to persist episode")
        raise RuntimeError(f"Persistence failed: {exc}") from exc


def reset_persistence_state() -> None:
    """Reset persistence state (for testing)."""
    global _graph_initialized, _seeded_self_groups
    _graph_initialized.clear()
    _seeded_self_groups.clear()


__all__ = [
    "ensure_database_ready",
    "ensure_graph_ready",
    "ensure_self_entity",
    "persist_episode",
    "reset_persistence_state",
]
