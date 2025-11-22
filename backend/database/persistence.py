"""Episode and entity persistence operations."""

from __future__ import annotations

import asyncio
import json
import logging
from threading import Lock
from typing import Any

from graphiti_core.nodes import EpisodicNode, EntityNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient, EMBEDDING_DIM
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


class NullEmbedder(EmbedderClient):
    """Fallback embedder that returns empty vectors when embeddings are unavailable."""

    async def create(self, input_data):  # type: ignore[override]
        return [[0.0] * EMBEDDING_DIM for _ in input_data]


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
    """Synchronous author entity "I" merge for use in asyncio.to_thread.

    Args:
        graph: FalkorDB graph instance
        journal: Journal name
        name: Name for the author entity "I"
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
        logger.exception('Failed to seed author entity "I"')
        raise RuntimeError(f"Failed to seed author entity (\"I\") for journal '{journal}'") from exc


async def ensure_self_entity(journal: str, name: str = SELF_ENTITY_NAME) -> None:
    """Seed the deterministic author entity "I" for this journal if missing.

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
        name: Name for the author entity "I" (defaults to SELF_ENTITY_NAME)
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
    """Ensure database is initialized and author entity "I" exists for this journal.

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


async def update_episode(
    episode_uuid: str,
    journal: str = DEFAULT_JOURNAL,
    *,
    content: str | None = None,
    name: str | None = None,
    valid_at: Any | None = None,
) -> None:
    """Update episode fields (content, name, valid_at only).

    Args:
        episode_uuid: Episode UUID string
        journal: Journal name (defaults to DEFAULT_JOURNAL)
        content: New content (optional)
        name: New title/name (optional)
        valid_at: New reference time as timezone-aware datetime object (optional)
                  Accepts datetime objects or ISO 8601 strings (converted to datetime)

    Raises:
        ValueError: If episode not found, no fields provided, or invalid datetime format

    Note:
        Uses graphiti-core's EpisodicNode for update operations.
        Only updates user-editable fields (content, name, valid_at).
        Immutable fields: uuid, group_id, created_at, source, source_description, entity_edges, labels.

        Datetime Handling:
        - Accepts timezone-aware datetime objects directly
        - Accepts ISO 8601 strings (e.g., "2024-01-15T10:30:00+00:00") - parsed to datetime
        - Naive datetimes are rejected with clear error (prevents timezone bugs)
        - Storage format is ISO 8601 string in database
        - Retrieval returns timezone-aware datetime objects (graphiti-core handles conversion)
    """
    from datetime import datetime
    from graphiti_core.errors import NodeNotFoundError
    from backend.database.driver import get_driver

    driver = get_driver(journal)

    # Retrieve the episode first (fail fast if not found)
    try:
        episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    except NodeNotFoundError as exc:
        raise ValueError(f"Episode {episode_uuid} not found") from exc

    # Validate that at least one field is being updated
    if content is None and name is None and valid_at is None:
        raise ValueError("At least one field must be provided for update")

    # Update only the provided fields
    if content is not None:
        episode.content = content

    if name is not None:
        episode.name = name

    if valid_at is not None:
        # Handle datetime conversion with clear semantics
        if isinstance(valid_at, datetime):
            # Reject naive datetimes to prevent timezone bugs
            if valid_at.tzinfo is None:
                raise ValueError(
                    "Naive datetime not allowed for valid_at. "
                    "Use timezone-aware datetime (e.g., datetime(..., tzinfo=timezone.utc))"
                )
            episode.valid_at = valid_at
        elif isinstance(valid_at, str):
            # Parse ISO 8601 string to datetime
            try:
                parsed_dt = datetime.fromisoformat(valid_at)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ISO 8601 datetime string: {valid_at}. "
                    f"Expected format: 'YYYY-MM-DDTHH:MM:SS+HH:MM'"
                ) from exc

            # Check for timezone after successful parsing
            if parsed_dt.tzinfo is None:
                raise ValueError(
                    f"ISO string '{valid_at}' is timezone-naive. "
                    "Provide timezone info (e.g., '2024-01-15T10:30:00+00:00')"
                )
            episode.valid_at = parsed_dt
        else:
            raise ValueError(
                f"valid_at must be timezone-aware datetime or ISO 8601 string, "
                f"got {type(valid_at).__name__}"
            )

    # Save back to database
    await episode.save(driver)
    logger.info("Updated episode %s in journal %s", episode_uuid, journal)


async def delete_episode(episode_uuid: str, journal: str = DEFAULT_JOURNAL) -> None:
    """Delete an episode from the graph.

    Args:
        episode_uuid: Episode UUID string
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Raises:
        ValueError: If episode not found

    Behavior:
        - Deletes the Episodic node
        - Removes MENTIONS edges to entities
        - Deletes entities that are no longer referenced by any episode
        - Removes the episode hash from Redis queue (idempotent, best-effort)

    Redis cleanup note:
        Redis removal is best-effort; failures are logged but do not block graph deletion.
        Queue operations are idempotent, so leftover hashes will not mutate graph data,
        but they may emit retries/log noise until cleaned up manually.
    """
    from backend.database.redis_ops import remove_episode_from_queue
    from graphiti_core.errors import NodeNotFoundError
    from backend.database.driver import get_driver

    driver = get_driver(journal)

    # Retrieve the episode
    try:
        episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
    except NodeNotFoundError as exc:
        raise ValueError(f"Episode {episode_uuid} not found") from exc

    # Gather deletion stats before mutating the graph
    stats_query = """
    MATCH (ep:Episodic {uuid: $episode_uuid})
    OPTIONAL MATCH (ep)-[:MENTIONS]->(ent:Entity)
    WITH ep, collect(DISTINCT ent) AS ents
    WITH ep, [e IN ents WHERE e IS NOT NULL AND size((e)<-[:MENTIONS]-()) = 1] AS doomed
    OPTIONAL MATCH (d:Entity)-[rel]-() WHERE d IN doomed AND type(rel) <> 'MENTIONS'
    RETURN coalesce(ep.name, '') AS episode_name,
           size(doomed) AS doomed_entities,
           [e IN doomed | coalesce(e.name, e.uuid)] AS doomed_entity_names,
           count(DISTINCT rel) AS non_mention_edges
    """
    stats_records, _, _ = await driver.execute_query(stats_query, episode_uuid=episode_uuid)
    stats = stats_records[0] if stats_records else {}
    episode_name = stats.get("episode_name") or episode.name or ""
    doomed_entities = int(stats.get("doomed_entities") or 0)
    doomed_entity_names = stats.get("doomed_entity_names") or []
    non_mention_edges = int(stats.get("non_mention_edges") or 0)

    total_nodes_to_delete = 1 + doomed_entities  # episode + orphan entities
    logger.info(
        "Deleting episode %s (%s): removing %d nodes (1 episode + %d entities: %s) and %d non-MENTIONS edges",
        episode_uuid,
        episode_name,
        total_nodes_to_delete,
        doomed_entities,
        ", ".join(doomed_entity_names) if doomed_entity_names else "none",
        non_mention_edges,
    )

    # Remove MENTIONS edges to entities, delete episode, and prune entities with no remaining mentions
    prune_query = """
    MATCH (ent_episode:Episodic {uuid: $episode_uuid})
    OPTIONAL MATCH (ent_episode)-[r:MENTIONS]->(entity:Entity)
    WITH ent_episode, collect(distinct entity) AS ents, collect(r) AS rels
    FOREACH (_ IN rels | DELETE _)
    DELETE ent_episode
    WITH ents
    UNWIND ents AS ent
    WITH DISTINCT ent
    WHERE ent IS NOT NULL AND NOT (ent)<-[:MENTIONS]-()
    DETACH DELETE ent
    """
    await driver.execute_query(prune_query, episode_uuid=episode_uuid)

    # Keep Redis in sync (idempotent if already removed)
    try:
        remove_episode_from_queue(episode_uuid)
    except Exception:
        logger.warning(
            "Failed to remove episode %s from Redis queue after deletion", episode_uuid, exc_info=True
        )

    logger.info("Deleted episode %s from journal %s (graph and Redis cleaned)", episode_uuid, journal)


async def persist_entities_and_edges(
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    episodic_edges: list[EpisodicEdge],
    journal: str,
) -> None:
    """Persist entities and relationships using graphiti-core bulk writer.

    Ensures embeddings are set, creates NullEmbedder, calls add_nodes_and_edges_bulk.
    Thread-safe via per-journal locking.

    Args:
        nodes: List of EntityNode objects to persist
        edges: List of EntityEdge objects to persist
        episodic_edges: List of EpisodicEdge objects to persist
        journal: Journal name (graph to persist to)

    Raises:
        RuntimeError: If persistence fails
    """
    from backend.database.driver import get_driver
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    await ensure_database_ready(journal)

    driver = get_driver(journal)
    embedder = NullEmbedder()

    try:
        # Ensure all nodes have embeddings set (even if empty)
        for node in nodes:
            if not node.name_embedding:
                node.name_embedding = []

        # Ensure all entity edges have embeddings set (even if empty)
        for edge in edges:
            if not edge.fact_embedding:
                edge.fact_embedding = []

        # Use graphiti-core bulk writer
        await add_nodes_and_edges_bulk(
            driver=driver,
            episodic_nodes=[],
            episodic_edges=episodic_edges,
            entity_nodes=nodes,
            entity_edges=edges,
            embedder=embedder,
        )

        logger.info(
            "Persisted %d entities, %d entity edges, %d episodic edges to journal %s",
            len(nodes),
            len(edges),
            len(episodic_edges),
            journal,
        )

    except Exception as exc:
        logger.exception("Failed to persist entities and edges")
        raise RuntimeError(f"Entity persistence failed: {exc}") from exc


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
    "update_episode",
    "delete_episode",
    "persist_entities_and_edges",
    "reset_persistence_state",
]
