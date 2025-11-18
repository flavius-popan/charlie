"""Database operations for backend API - simplified from pipeline/falkordblite_driver.py"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import signal
import time
from threading import Lock
from typing import Any
from uuid import UUID

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.embedder.client import EmbedderClient, EMBEDDING_DIM
from graphiti_core.helpers import get_default_group_id
from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings, utc_now

from backend.settings import (
    DB_PATH,
    DEFAULT_JOURNAL,
    ENABLE_TCP_SERVER,
    TCP_HOST,
    TCP_PASSWORD,
    TCP_PORT,
)

try:
    from redislite.falkordb_client import FalkorDB
except Exception:
    FalkorDB = None  # type: ignore[assignment]

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Stopwords for fulltext search indexing
STOPWORDS = [
    'a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but',
    'by', 'for', 'if', 'in', 'into', 'it', 'no', 'not', 'of', 'on',
    'or', 'such', 'that', 'their', 'then', 'there', 'these', 'they',
    'this', 'to', 'was', 'will', 'with',
]

# Journal name validation
JOURNAL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
MAX_JOURNAL_NAME_LENGTH = 64

def validate_journal_name(journal: str) -> None:
    """Validate journal name for use as FalkorDB graph name.

    Args:
        journal: Journal name to validate

    Raises:
        ValueError: If name is invalid (empty, too long, or contains invalid characters)
    """
    if not journal:
        raise ValueError("Journal name cannot be empty")
    if len(journal) > MAX_JOURNAL_NAME_LENGTH:
        raise ValueError(
            f"Journal name too long: {len(journal)} chars (max {MAX_JOURNAL_NAME_LENGTH})"
        )
    if not JOURNAL_NAME_PATTERN.match(journal):
        raise ValueError(
            f"Invalid journal name '{journal}'. "
            "Use only: letters, numbers, underscores, hyphens"
        )

# SELF entity constants (deterministic UUID for journal author)
SELF_ENTITY_UUID = UUID("11111111-1111-1111-1111-111111111111")
SELF_ENTITY_NAME = "Self"
SELF_ENTITY_LABELS = ["Entity", "Person"]

# Database state
_db_lock = Lock()
_db = None
_graphs: dict[str, Any] = {}  # Cache of graph instances per journal
_graph_locks: dict[str, Lock] = {}  # Per-journal locks for thread safety
_db_unavailable = False

_graph_initialized: dict[str, bool] = {}  # Per-journal initialization tracking
_graph_init_lock = None
_seeded_self_groups: set[str] = set()
_self_seed_lock = None

# TCP server configuration
_tcp_server = {
    "enabled": ENABLE_TCP_SERVER and TCP_PORT > 0,
    "host": TCP_HOST,
    "port": TCP_PORT,
    "password": TCP_PASSWORD,
}

# Ensure database directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def to_cypher_literal(value: Any) -> str:
    """Convert Python values into Cypher literals (FalkorDB lacks parameters)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return json.dumps(value)


SELF_UUID_LITERAL = to_cypher_literal(str(SELF_ENTITY_UUID))


def _tcp_server_active() -> bool:
    """Check if TCP server is configured and enabled."""
    port = _tcp_server["port"]
    return bool(_tcp_server["enabled"] and port and port > 0)


def _build_serverconfig() -> dict[str, str] | None:
    """Build FalkorDB server configuration for TCP endpoint."""
    if not _tcp_server_active():
        return None

    config: dict[str, str] = {"port": str(int(_tcp_server["port"]))}
    if _tcp_server["host"]:
        config["bind"] = _tcp_server["host"]
    if _tcp_server["password"]:
        config["requirepass"] = _tcp_server["password"]
    return config


def get_tcp_server_endpoint() -> tuple[str, int] | None:
    """Return (host, port) if TCP server is configured, otherwise None."""
    if not _tcp_server_active():
        return None
    host = _tcp_server["host"] or "127.0.0.1"
    return host, int(_tcp_server["port"])


def get_tcp_server_password() -> str | None:
    """Return the configured password for the TCP server (if any)."""
    if not _tcp_server_active():
        return None
    return _tcp_server["password"]


def enable_tcp_server(
    *,
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
) -> None:
    """Expose FalkorDB Lite on a TCP port in addition to the default unix socket."""
    global _tcp_server
    if port is not None:
        if port <= 0:
            logger.warning("Ignoring TCP enable request with invalid port %s", port)
            return
        _tcp_server["port"] = port
    if host:
        _tcp_server["host"] = host
    if password is not None:
        _tcp_server["password"] = password or None

    _tcp_server["enabled"] = bool(_tcp_server["port"])
    if _db is not None:
        logger.warning(
            "FalkorDB Lite TCP configuration updated after initialization; "
            "restart the process to apply the change."
        )


def disable_tcp_server() -> None:
    """Disable the TCP listener (unix socket access remains available)."""
    _tcp_server["enabled"] = False


def _decode_value(value: Any) -> Any:
    """Decode FalkorDB result cells to Python primitives."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _init_db():
    """Initialize the FalkorDB database connection."""
    global _db, _db_unavailable

    if FalkorDB is None:
        logger.debug("FalkorDB client not available")
        _db_unavailable = True
        raise RuntimeError("FalkorDB Lite is unavailable")

    try:
        serverconfig = _build_serverconfig()
        kwargs: dict[str, Any] = {}
        if serverconfig:
            kwargs["serverconfig"] = serverconfig
        _db = FalkorDB(dbfilename=str(DB_PATH), **kwargs)
        if serverconfig:
            endpoint = get_tcp_server_endpoint()
            if endpoint:
                host, port = endpoint
                logger.info(
                    "FalkorDB Lite TCP debug endpoint exposed on %s:%d",
                    host,
                    port,
                )
    except Exception as exc:
        logger.warning("Failed to initialize FalkorDB Lite: %s", exc)
        _db_unavailable = True
        raise RuntimeError("FalkorDB Lite is unavailable") from exc


def _ensure_graph(journal: str):
    """Get or create a FalkorDB graph for the specified journal.

    Args:
        journal: Journal name (used as graph name)

    Returns:
        Tuple of (graph instance, lock for this journal)

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If FalkorDB is unavailable
    """
    validate_journal_name(journal)

    global _db, _graphs, _graph_locks, _db_unavailable

    if _db_unavailable:
        raise RuntimeError("FalkorDB Lite is unavailable")

    with _db_lock:
        # Return cached graph and lock if exists
        if journal in _graphs:
            return _graphs[journal], _graph_locks[journal]

        # Initialize database if needed
        if _db is None:
            _init_db()

        # Select and cache the graph for this journal
        _graphs[journal] = _db.select_graph(journal)
        _graph_locks[journal] = Lock()  # Create per-journal lock
        logger.debug("Created graph for journal '%s'", journal)
        return _graphs[journal], _graph_locks[journal]


def _pid_exists(pid: int) -> bool:
    """Check if process with given PID exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Return True when the process exits within timeout seconds."""
    if pid <= 0:
        return True

    deadline = time.time() + timeout
    if psutil is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return True
        while time.time() < deadline:
            if not process.is_running():
                return True
            try:
                process.wait(0.1)
                return True
            except psutil.TimeoutExpired:
                continue
        return False

    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)
    return False


def _send_signal(pid: int, sig: signal.Signals) -> None:
    """Send signal to process with given PID."""
    if pid <= 0:
        return
    if psutil is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        try:
            process.send_signal(sig)
        except Exception:
            logger.debug("Failed to send %s to redis pid %s", sig.name, pid, exc_info=True)
        return
    try:
        os.kill(pid, sig.value)
    except OSError:
        logger.debug("os.kill failed for pid %s signal %s", pid, sig, exc_info=True)


def _ensure_redis_stopped(redis_client) -> None:
    """Best-effort shutdown of the embedded redis-server process.

    Uses direct SIGTERM instead of the blocking shutdown() command to avoid
    a known redis-py client timeout issue that causes 5-10 second delays.
    See: https://github.com/redis/redis-py/issues/3704
    """
    if redis_client is None:
        return

    pid = getattr(redis_client, "pid", 0)
    if pid <= 0:
        return

    try:
        redis_client.connection_pool.disconnect()
    except Exception:
        logger.debug("Failed to disconnect redis connection pool cleanly", exc_info=True)

    _send_signal(pid, signal.SIGTERM)
    if _wait_for_exit(pid, timeout=3.0):
        return

    logger.warning("redis-server pid %s ignored SIGTERM; sending SIGKILL", pid)
    _send_signal(pid, signal.SIGKILL)
    _wait_for_exit(pid, timeout=1.0)


def _close_db():
    """Ensure the embedded FalkorDB process shuts down cleanly.

    Skips db.close() because it calls client.shutdown() which has a
    known blocking timeout issue. We handle shutdown directly via SIGTERM.
    """
    global _db, _graphs, _graph_locks
    with _db_lock:
        if _db is None:
            return
        client = getattr(_db, "client", None)
        try:
            _ensure_redis_stopped(client)
        finally:
            _db = None
            _graphs.clear()
            _graph_locks.clear()


atexit.register(_close_db)


def get_falkordb_graph(journal: str = DEFAULT_JOURNAL):
    """Return a FalkorDB graph instance for the specified journal.

    Args:
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        Graph instance for the journal

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If FalkorDB is unavailable
    """
    graph, _lock = _ensure_graph(journal)
    return graph


class NullEmbedder(EmbedderClient):
    """Fallback embedder that returns empty vectors."""

    async def create(self, input_data):  # type: ignore[override]
        return [[0.0] * EMBEDDING_DIM for _ in input_data]


_embedder = NullEmbedder()


class FalkorLiteSession(GraphDriverSession):
    """Simplified FalkorDB session for backend operations."""

    provider = GraphProvider.FALKORDB

    def __init__(self, journal: str = DEFAULT_JOURNAL):
        self._journal = journal
        self._graph, self._lock = _ensure_graph(journal)

    async def run(self, query, **kwargs):
        """Execute bulk operations (episodes only for now)."""
        graph = self._graph
        if graph is None:
            raise RuntimeError("FalkorDB Lite is unavailable")

        if "episodes" in kwargs:
            episodes = kwargs["episodes"]
            for ep in episodes:
                # Use lock to serialize access to this journal's graph
                def _locked_merge():
                    with self._lock:
                        _merge_episode_sync(graph, ep)
                await asyncio.to_thread(_locked_merge)
            return None

        if "nodes" in kwargs:
            nodes = kwargs["nodes"]
            if nodes:  # Only raise if there are actually nodes to persist
                raise NotImplementedError(
                    "Entity node persistence not yet implemented in backend"
                )
            return None

        if "entity_edges" in kwargs:
            edges = kwargs["entity_edges"]
            if edges:  # Only raise if there are actually edges to persist
                raise NotImplementedError(
                    "Entity edge persistence not yet implemented in backend"
                )
            return None

        if "episodic_edges" in kwargs:
            episodic_edges = kwargs["episodic_edges"]
            if episodic_edges:  # Only raise if there are actually edges to persist
                raise NotImplementedError(
                    "Episodic edge persistence not yet implemented in backend"
                )
            return None

        raise NotImplementedError(
            "FalkorLiteSession.run only supports bulk persistence operations"
        )

    async def close(self):
        pass

    async def execute_write(self, tx_func, *args, **kwargs):
        """Execute a write transaction."""
        return await tx_func(self, *args, **kwargs)

    async def __aenter__(self):
        """Async context manager enter."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False


class FalkorLiteDriver(GraphDriver):
    """Simplified FalkorDB driver for backend operations."""

    provider = GraphProvider.FALKORDB

    def __init__(self, journal: str = DEFAULT_JOURNAL):
        super().__init__()
        self._journal = journal
        self._graph, self._lock = _ensure_graph(journal)
        self._database = "falkordb-lite"

    async def execute_query(self, cypher_query_, **kwargs: Any):
        """Execute a Cypher query with parameter substitution."""
        graph = self._graph
        if graph is None:
            raise RuntimeError("FalkorDB Lite is unavailable")

        params = dict(kwargs)
        params.pop("routing_", None)
        params = convert_datetimes_to_strings(params)

        def _inject(query: str, substitutions: dict[str, Any]) -> str:
            for key, value in substitutions.items():
                placeholder = f"${key}"
                literal = to_cypher_literal(value)
                query = query.replace(placeholder, literal)
            return query

        formatted_query = _inject(cypher_query_, params)

        # Use lock to serialize access to this journal's graph
        def _locked_query():
            with self._lock:
                return graph.query(formatted_query, None)

        result = await asyncio.to_thread(_locked_query)

        raw = getattr(result, "_raw_response", None)
        records: list[dict[str, Any]] = []
        header: list[str] = []

        if isinstance(raw, list) and len(raw) >= 2:
            header = [_decode_value(col[1]) for col in raw[0]]
            rows = raw[1]

            for row in rows:
                record: dict[str, Any] = {}
                for idx, field_name in enumerate(header):
                    value = row[idx][1] if idx < len(row) else None
                    record[str(field_name)] = _decode_value(value)
                records.append(record)

        return records, header, None

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """Build FalkorDB range and fulltext indices."""
        from graphiti_core.graph_queries import get_range_indices

        if delete_existing:
            await self.delete_all_indexes()

        # Range indices from graphiti-core
        index_queries = get_range_indices(self.provider)

        # Fulltext indices for BM25 search on episodes and entities
        fulltext_queries = [
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Episodic', stopwords: {STOPWORDS}}},
                'content', 'source', 'source_description', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Entity', stopwords: {STOPWORDS}}},
                'name', 'summary', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Community', stopwords: {STOPWORDS}}},
                'name', 'summary'
            )""",
        ]

        index_queries.extend(fulltext_queries)

        for query in index_queries:
            try:
                await self.execute_query(query)
            except Exception as exc:
                if "already indexed" in str(exc).lower():
                    logger.debug("Index already exists, skipping: %s", exc)
                else:
                    raise

    def session(self, database=None) -> GraphDriverSession:
        return FalkorLiteSession(journal=self._journal)

    async def close(self):
        """Close the driver (no-op for FalkorDB Lite)."""
        pass

    async def delete_all_indexes(self):
        """Delete all indexes (simplified implementation)."""
        pass

    @staticmethod
    def _sanitize_fulltext(query: str) -> str:
        """Remove special characters that interfere with fulltext search."""
        separator_map = str.maketrans(
            {
                ',': ' ', '.': ' ', '<': ' ', '>': ' ', '{': ' ', '}': ' ',
                '[': ' ', ']': ' ', '"': ' ', "'": ' ', ':': ' ', ';': ' ',
                '!': ' ', '@': ' ', '#': ' ', '$': ' ', '%': ' ', '^': ' ',
                '&': ' ', '*': ' ', '(': ' ', ')': ' ', '-': ' ', '+': ' ',
                '=': ' ', '~': ' ', '?': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        return " ".join(sanitized.split())

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """Build a fulltext search query for FalkorDB using RedisSearch syntax."""
        group_filter = (
            f"(@group_id:{'|'.join(group_ids)})" if group_ids and len(group_ids) > 0 else ""
        )
        sanitized_query = self._sanitize_fulltext(query)
        filtered_words = [word for word in sanitized_query.split() if word.lower() not in STOPWORDS]
        sanitized_query = " | ".join(filtered_words)

        if len(sanitized_query.split(" ")) + len(group_filter.split(" ")) >= max_query_length:
            return ""

        if not sanitized_query and not group_filter:
            return ""

        if not sanitized_query:
            return group_filter

        if group_filter:
            return f"{group_filter} ({sanitized_query})"
        return f"({sanitized_query})"


def _get_driver(journal: str = DEFAULT_JOURNAL) -> FalkorLiteDriver:
    """Create a driver instance for the specified journal.

    Args:
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        New FalkorLiteDriver instance for the journal

    Note: This is not a singleton - each call creates a new driver instance.
    """
    return FalkorLiteDriver(journal=journal)


def _merge_episode_sync(graph, episode: dict[str, Any]) -> None:
    """
    Synchronous episode merge for use in asyncio.to_thread.

    Args:
        graph: FalkorDB graph instance
        episode: Episode dict (graphiti-core converts EpisodicNode to dict before calling)
    """

    # graphiti-core converts EpisodicNode to dict and removes 'labels' field (bulk_utils.py line 160-162)
    # Use setdefault to ensure these fields exist with empty defaults
    episode.setdefault('entity_edges', [])
    episode.setdefault('labels', [])

    entity_edges_json = json.dumps(episode['entity_edges'])
    labels_json = json.dumps(episode['labels'])

    # Handle source - could be EpisodeType enum or string
    source_val = episode['source']
    if hasattr(source_val, 'value'):
        source_val = source_val.value
    elif isinstance(source_val, str):
        pass  # Already a string
    else:
        source_val = str(source_val)

    # Handle datetime fields - could be datetime objects or ISO strings
    valid_at = episode['valid_at']
    if hasattr(valid_at, 'isoformat'):
        valid_at = valid_at.isoformat()

    created_at = episode['created_at']
    if hasattr(created_at, 'isoformat'):
        created_at = created_at.isoformat()

    query = f"""
    MERGE (e:Episodic {{uuid: {to_cypher_literal(episode['uuid'])}}})
    SET e.name = {to_cypher_literal(episode['name'])},
        e.group_id = {to_cypher_literal(episode['group_id'])},
        e.content = {to_cypher_literal(episode['content'])},
        e.source = {to_cypher_literal(source_val)},
        e.source_description = {to_cypher_literal(episode['source_description'])},
        e.valid_at = {to_cypher_literal(valid_at)},
        e.created_at = {to_cypher_literal(created_at)},
        e.entity_edges = {to_cypher_literal(entity_edges_json)},
        e.labels = {to_cypher_literal(labels_json)}
    RETURN e.uuid
    """

    try:
        graph.query(query)
    except Exception as exc:
        logger.exception("Failed to merge episode")
        raise RuntimeError(f"Failed to persist episode {episode['uuid']}") from exc


async def ensure_graph_ready(journal: str = DEFAULT_JOURNAL, *, delete_existing: bool = False) -> None:
    """Build Graphiti indices/constraints for a journal's graph.

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

    if _graph_init_lock is None:
        _graph_init_lock = asyncio.Lock()

    async with _graph_init_lock:
        if _graph_initialized.get(journal, False) and not delete_existing:
            return

        # Create a journal-specific driver for index building
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

    Args:
        journal: Journal name
        name: Name for the SELF entity (defaults to SELF_ENTITY_NAME)
    """
    if journal in _seeded_self_groups:
        return

    global _self_seed_lock
    if _self_seed_lock is None:
        _self_seed_lock = asyncio.Lock()

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
    """
    Persist an episode to FalkorDB.

    Args:
        episode: The EpisodicNode to persist
        journal: Journal name (graph to persist to)

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If persistence fails
    """
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


def shutdown_database():
    """Manual database shutdown for testing.

    Resets all global state to allow clean database reinitialization.
    """
    global _db_unavailable, _graph_initialized
    _close_db()
    _db_unavailable = False
    _graph_initialized.clear()
    _seeded_self_groups.clear()


__all__ = [
    "ensure_database_ready",
    "persist_episode",
    "shutdown_database",
    "SELF_ENTITY_UUID",
    "SELF_ENTITY_NAME",
    "SELF_ENTITY_LABELS",
]
