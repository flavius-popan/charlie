"""FalkorDB Lite driver plus consolidated persistence and query utilities."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import signal
import time
from datetime import datetime
from threading import Lock
from typing import Any, Iterable

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient, EMBEDDING_DIM
from graphiti_core.helpers import get_default_group_id
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings, utc_now

from settings import (
    DB_PATH,
    GRAPH_NAME,
    FALKORLITE_TCP_ENABLED_BY_DEFAULT,
    FALKORLITE_TCP_HOST,
    FALKORLITE_TCP_PASSWORD,
    FALKORLITE_TCP_PORT,
    GROUP_ID,
)
from pipeline.self_reference import (
    SELF_ENTITY_LABELS,
    SELF_ENTITY_NAME,
    SELF_ENTITY_UUID,
)

try:  # Optional dependency in CI / unit tests
    from redislite.falkordb_client import FalkorDB
except Exception:  # noqa: BLE001
    FalkorDB = None  # type: ignore[assignment]

try:  # Optional, used for graceful shutdown observation
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

STOPWORDS = [
    'a',
    'is',
    'the',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'for',
    'if',
    'in',
    'into',
    'it',
    'no',
    'not',
    'of',
    'on',
    'or',
    'such',
    'that',
    'their',
    'then',
    'there',
    'these',
    'they',
    'this',
    'to',
    'was',
    'will',
    'with',
]

logger = logging.getLogger(__name__)

_db_lock = Lock()
_db = None
_graph = None
_db_unavailable = False

# Ensure the embedded database directory exists before first use.
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_tcp_server = {
    "enabled": FALKORLITE_TCP_ENABLED_BY_DEFAULT and FALKORLITE_TCP_PORT > 0,
    "host": FALKORLITE_TCP_HOST,
    "port": FALKORLITE_TCP_PORT,
    "password": FALKORLITE_TCP_PASSWORD,
}

_graph_initialized = False
_graph_init_lock: asyncio.Lock | None = None
_self_seed_lock: asyncio.Lock | None = None
_seeded_self_groups: set[str] = set()


def _tcp_server_active() -> bool:
    port = _tcp_server["port"]
    return bool(_tcp_server["enabled"] and port and port > 0)


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
            "FalkorDB Lite TCP configuration updated after initialisation; "
            "restart the process to apply the change."
        )


def disable_tcp_server() -> None:
    """Disable the TCP listener (unix socket access remains available)."""
    _tcp_server["enabled"] = False


def get_tcp_server_endpoint() -> tuple[str, int] | None:
    """Return (host, port) if the TCP server is configured, otherwise None."""
    if not _tcp_server_active():
        return None
    host = _tcp_server["host"] or "127.0.0.1"
    return host, int(_tcp_server["port"])


def get_tcp_server_password() -> str | None:
    """Return the configured password for the TCP server (if any)."""
    if not _tcp_server_active():
        return None
    return _tcp_server["password"]


async def ensure_graph_ready(*, delete_existing: bool = False) -> None:
    """Build Graphiti indices/constraints once per process."""
    global _graph_initialized, _graph_init_lock, _seeded_self_groups
    if delete_existing:
        _graph_initialized = False
        _seeded_self_groups.clear()

    if _graph_initialized and not delete_existing:
        return

    if _graph_init_lock is None:
        _graph_init_lock = asyncio.Lock()

    async with _graph_init_lock:
        if _graph_initialized and not delete_existing:
            return
        driver = _get_driver()
        try:
            await driver.build_indices_and_constraints(
                delete_existing=delete_existing,
            )
        except ImportError as exc:  # pragma: no cover - falkordb optional
            logger.warning(
                "Skipping Falkor index bootstrap because dependencies are missing: %s",
                exc,
            )
        _graph_initialized = True


async def ensure_self_entity(group_id: str, name: str = SELF_ENTITY_NAME) -> None:
    """Seed the deterministic SELF entity for journaling if it's missing."""
    normalized_group = group_id or DEFAULT_SELF_GROUP
    if normalized_group in _seeded_self_groups:
        return

    global _self_seed_lock
    if _self_seed_lock is None:
        _self_seed_lock = asyncio.Lock()

    async with _self_seed_lock:
        if normalized_group in _seeded_self_groups:
            return
        await asyncio.to_thread(_merge_self_entity_sync, normalized_group, name)
        _seeded_self_groups.add(normalized_group)


def _build_serverconfig() -> dict[str, str] | None:
    if not _tcp_server_active():
        return None

    config: dict[str, str] = {"port": str(int(_tcp_server["port"]))}
    if _tcp_server["host"]:
        config["bind"] = _tcp_server["host"]
    if _tcp_server["password"]:
        config["requirepass"] = _tcp_server["password"]
    return config


def _ensure_graph():
    """Return a FalkorDB graph instance if available, otherwise None."""
    global _db, _graph, _db_unavailable
    if _graph is not None:
        return _graph
    if _db_unavailable:
        return None
    if FalkorDB is None:
        logger.debug("FalkorDB client not available; returning empty query results.")
        _db_unavailable = True
        return None
    with _db_lock:
        if _graph is not None:
            return _graph
        if _db_unavailable:
            return None
        try:
            serverconfig = _build_serverconfig()
            kwargs: dict[str, Any] = {}
            if serverconfig:
                kwargs["serverconfig"] = serverconfig
            _db = FalkorDB(dbfilename=str(DB_PATH), **kwargs)
            _graph = _db.select_graph(GRAPH_NAME)
            if serverconfig:
                endpoint = get_tcp_server_endpoint()
                if endpoint:
                    host, port = endpoint
                    logger.info(
                        "FalkorDB Lite TCP debug endpoint exposed on %s:%d",
                        host,
                        port,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise FalkorDBLite: %s", exc)
            _db_unavailable = True
            return None

    return _graph


def _merge_self_entity_sync(group_id: str, name: str) -> None:
    graph = _ensure_graph()
    if graph is None:
        return

    now_literal = to_cypher_literal(utc_now().isoformat())
    summary_literal = to_cypher_literal(
        "Represents the journal author for first-person perspective anchoring."
    )
    labels_literal = to_cypher_literal(json.dumps(SELF_ENTITY_LABELS))
    attributes_literal = to_cypher_literal(json.dumps({}))

    query = f"""
    MERGE (self:Entity:Person {{uuid: {SELF_UUID_LITERAL}}})
    SET self.name = {to_cypher_literal(name)},
        self.group_id = COALESCE(self.group_id, {to_cypher_literal(group_id)}),
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to seed SELF entity: %s", exc)


def _pid_exists(pid: int) -> bool:
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
    if pid <= 0:
        return
    if psutil is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        try:
            process.send_signal(sig)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to send %s to redis pid %s", sig.name, pid, exc_info=True)
        return
    try:
        os.kill(pid, sig.value)
    except OSError:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    global _db, _graph
    with _db_lock:
        if _db is None:
            return
        client = getattr(_db, "client", None)
        try:
            _ensure_redis_stopped(client)
        finally:
            _db = None
            _graph = None


def shutdown_falkordb():
    """Public helper for tests to forcefully stop the embedded database."""
    _close_db()


atexit.register(_close_db)


def get_falkordb_graph():
    """Return the shared FalkorDB graph instance (None if unavailable)."""
    return _ensure_graph()


def get_driver() -> GraphDriver:
    """Expose the shared Falkor Lite driver."""
    return _get_driver()


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
DEFAULT_SELF_GROUP = GROUP_ID or get_default_group_id(GraphProvider.FALKORDB)


def _decode_value(value: Any) -> Any:
    """Decode FalkorDB result cells to Python primitives."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:  # noqa: BLE001
            return value
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _decode_json(value: Any, default: Any) -> Any:
    """Decode JSON stored in FalkorDB properties."""
    decoded = _decode_value(value)
    if decoded in ("", None):
        return default
    if isinstance(decoded, (list, dict)):
        return decoded
    try:
        return json.loads(decoded)
    except Exception:  # noqa: BLE001
        return default


def _normalize_string_list(value: Any) -> list[str]:
    """Ensure labels/edges are represented as a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return []


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO8601 datetimes returned from FalkorDB."""
    decoded = _decode_value(value)
    if not decoded:
        return None
    if isinstance(decoded, datetime):
        return decoded
    try:
        return datetime.fromisoformat(decoded)
    except Exception:  # noqa: BLE001
        return None


def _iter_statistics_rows(result) -> Iterable[list[Any]]:
    """Yield decoded rows from a FalkorDB query result."""
    rows = getattr(result, "statistics", None) or []
    for row in rows:
        yield [
            _decode_value(col[1])
            if isinstance(col, (list, tuple)) and len(col) > 1
            else None
            for col in row
        ]


def _fetch_recent_episodes_sync(
    group_id: str,
    reference_time: datetime,
    limit: int,
) -> list[EpisodicNode]:
    graph = _ensure_graph()
    if graph is None:
        return []

    reference_literal = to_cypher_literal(reference_time.isoformat())
    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (e:Episodic)
    WHERE e.group_id = {group_literal}
      AND e.valid_at <= {reference_literal}
    RETURN e.uuid, e.name, e.content, e.source_description,
           e.valid_at, e.created_at, e.entity_edges, e.labels,
           e.source
    ORDER BY e.valid_at DESC
    LIMIT {limit}
    """

    try:
        result = graph.query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FalkorDB query failed (fetch_recent_episodes): %s", exc)
        return []

    episodes: list[EpisodicNode] = []
    for (
        uuid,
        name,
        content,
        source_description,
        valid_at,
        created_at,
        entity_edges,
        labels,
        source_value,
    ) in _iter_statistics_rows(result):
        source_literal = _decode_value(source_value) or EpisodeType.text.value
        try:
            source = EpisodeType(source_literal)
        except Exception:  # noqa: BLE001
            source = EpisodeType.text

        episode_kwargs: dict[str, Any] = {
            "name": str(name or ""),
            "content": str(content or ""),
            "source_description": str(source_description or ""),
            "valid_at": _parse_datetime(valid_at) or reference_time,
            "created_at": _parse_datetime(created_at) or reference_time,
            "entity_edges": _normalize_string_list(_decode_json(entity_edges, [])),
            "labels": _normalize_string_list(_decode_json(labels, [])),
            "group_id": group_id,
            "source": source,
        }
        if uuid:
            episode_kwargs["uuid"] = str(uuid)

        episodes.append(EpisodicNode(**episode_kwargs))

    return episodes


async def fetch_recent_episodes(
    group_id: str,
    reference_time: datetime,
    limit: int,
) -> list[EpisodicNode]:
    """Fetch the most recent episodes for a group."""
    await ensure_graph_ready()
    return await asyncio.to_thread(
        _fetch_recent_episodes_sync,
        group_id,
        reference_time,
        limit,
    )


def _fetch_entities_by_group_sync(group_id: str) -> dict[str, EntityNode]:
    graph = _ensure_graph()
    if graph is None:
        return {}

    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (n:Entity)
    WHERE n.group_id = {group_literal} OR n.uuid = {SELF_UUID_LITERAL}
    RETURN n.uuid, n.name, n.summary, n.labels, n.attributes, n.created_at
    """

    try:
        result = graph.query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FalkorDB query failed (fetch_entities_by_group): %s", exc)
        return {}

    entities: dict[str, EntityNode] = {}
    for uuid, name, summary, labels, attributes, created_at in _iter_statistics_rows(
        result
    ):
        node_kwargs: dict[str, Any] = {
            "name": str(name or ""),
            "summary": str(summary or ""),
            "labels": _normalize_string_list(_decode_json(labels, ["Entity"])),
            "attributes": _decode_json(attributes, {}),
            "created_at": _parse_datetime(created_at) or utc_now(),
            "group_id": group_id,
        }
        if uuid:
            node_kwargs["uuid"] = str(uuid)

        node = EntityNode(**node_kwargs)
        entities[node.uuid] = node

    return entities


async def fetch_entities_by_group(group_id: str) -> dict[str, EntityNode]:
    """Fetch all entities in the given group keyed by UUID."""
    await ensure_graph_ready()
    await ensure_self_entity(group_id)
    return await asyncio.to_thread(_fetch_entities_by_group_sync, group_id)


def _fetch_self_entity_sync() -> EntityNode | None:
    graph = _ensure_graph()
    if graph is None:
        return None

    query = f"""
    MATCH (n:Entity {{uuid: {SELF_UUID_LITERAL}}})
    RETURN n.uuid, n.name, n.summary, n.labels, n.attributes, n.created_at, n.group_id
    LIMIT 1
    """

    try:
        result = graph.query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FalkorDB query failed (fetch_self_entity): %s", exc)
        return None

    rows = list(_iter_statistics_rows(result))
    if not rows:
        return None

    uuid, name, summary, labels, attributes, created_at, group_id = rows[0]
    return EntityNode(
        uuid=str(uuid or ""),
        name=str(name or ""),
        summary=str(summary or ""),
        labels=_normalize_string_list(_decode_json(labels, ["Entity"])),
        attributes=_decode_json(attributes, {}),
        created_at=_parse_datetime(created_at) or utc_now(),
        group_id=str(group_id or ""),
    )


async def fetch_self_entity(group_id: str | None = None) -> EntityNode | None:
    """Return the canonical SELF entity for cloning in extraction stages."""
    await ensure_graph_ready()
    await ensure_self_entity(group_id or DEFAULT_SELF_GROUP)
    return await asyncio.to_thread(_fetch_self_entity_sync)


def _fetch_entity_edges_by_group_sync(group_id: str) -> dict[str, EntityEdge]:
    graph = _ensure_graph()
    if graph is None:
        return {}

    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
    WHERE edge.group_id = {group_literal}
    RETURN edge.uuid, edge.name, edge.fact, edge.fact_embedding,
           edge.episodes, edge.created_at, edge.expired_at,
           edge.valid_at, edge.invalid_at, edge.attributes,
           source.uuid, target.uuid
    """

    try:
        result = graph.query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FalkorDB query failed (fetch_entity_edges_by_group): %s", exc)
        return {}

    edges: dict[str, EntityEdge] = {}
    for (
        uuid,
        name,
        fact,
        fact_embedding,
        episodes,
        created_at,
        expired_at,
        valid_at,
        invalid_at,
        attributes,
        source_uuid,
        target_uuid,
    ) in _iter_statistics_rows(result):
        edge_kwargs: dict[str, Any] = {
            "name": str(name or ""),
            "fact": str(fact or ""),
            "fact_embedding": _decode_json(fact_embedding, None),
            "episodes": _normalize_string_list(_decode_json(episodes, [])),
            "created_at": _parse_datetime(created_at) or utc_now(),
            "expired_at": _parse_datetime(expired_at),
            "valid_at": _parse_datetime(valid_at),
            "invalid_at": _parse_datetime(invalid_at),
            "attributes": _decode_json(attributes, {}),
            "source_node_uuid": str(source_uuid or ""),
            "target_node_uuid": str(target_uuid or ""),
            "group_id": group_id,
        }
        if uuid:
            edge_kwargs["uuid"] = str(uuid)

        edge = EntityEdge(**edge_kwargs)
        edges[edge.uuid] = edge

    return edges


async def fetch_entity_edges_by_group(group_id: str) -> dict[str, EntityEdge]:
    """Fetch all entity edges for a group from FalkorDB.

    Args:
        group_id: Graph partition identifier

    Returns:
        Dict mapping edge_uuid -> EntityEdge
    """
    await ensure_graph_ready()
    return await asyncio.to_thread(_fetch_entity_edges_by_group_sync, group_id)


def _get_db_stats_sync() -> dict[str, int]:
    """Query database statistics (sync)."""
    graph = _ensure_graph()
    if graph is None:
        return {"episodes": 0, "entities": 0}

    try:
        episodic_result = graph.query("MATCH (e:Episodic) RETURN count(e)")
        episodic_count = 0
        if episodic_result.statistics and episodic_result.statistics[0]:
            col = episodic_result.statistics[0][0]
            val = col[1] if len(col) > 1 else col[0]
            episodic_count = int(_decode_value(val) or 0)

        entity_result = graph.query("MATCH (n:Entity) RETURN count(n)")
        entity_count = 0
        if entity_result.statistics and entity_result.statistics[0]:
            col = entity_result.statistics[0][0]
            val = col[1] if len(col) > 1 else col[0]
            entity_count = int(_decode_value(val) or 0)

        return {"episodes": episodic_count, "entities": entity_count}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get database stats: %s", exc)
        return {"episodes": 0, "entities": 0}


async def get_db_stats() -> dict[str, int]:
    """Get database statistics (episode and entity counts)."""
    await ensure_graph_ready()
    return await asyncio.to_thread(_get_db_stats_sync)


def _reset_database_sync() -> str:
    """Clear all graph data (DESTRUCTIVE, sync)."""
    graph = _ensure_graph()
    if graph is None:
        return "Database unavailable"

    try:
        graph.query("MATCH (n) DETACH DELETE n")
        return "Database cleared successfully"
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to reset database: %s", exc)
        return f"Database reset failed: {exc}"


async def reset_database() -> str:
    """Clear all graph data (DESTRUCTIVE)."""
    result = await asyncio.to_thread(_reset_database_sync)
    await ensure_graph_ready(delete_existing=True)
    await ensure_self_entity(GROUP_ID)
    return result


def _prepare_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for item in items:
        prepared.append(convert_datetimes_to_strings(dict(item)))
    return prepared


def _merge_episodes(graph, episodes: list[dict[str, Any]]) -> None:
    for episode in _prepare_dicts(episodes):
        episode.setdefault("entity_edges", [])
        episode.setdefault("labels", [])
        props = {
            "uuid": episode["uuid"],
            "name": episode.get("name", ""),
            "group_id": episode.get("group_id", ""),
            "source": episode.get("source", ""),
            "source_description": episode.get("source_description", ""),
            "content": episode.get("content", ""),
            "valid_at": episode.get("valid_at"),
            "created_at": episode.get("created_at"),
            "entity_edges": json.dumps(episode["entity_edges"]),
            "labels": json.dumps(episode["labels"]),
        }
        set_clause = ", ".join(
            [f"e.{key} = {to_cypher_literal(value)}" for key, value in props.items()]
        )
        query = f"""
        MERGE (e:Episodic {{uuid: {to_cypher_literal(episode["uuid"])}}})
        SET {set_clause}
        RETURN e.uuid AS uuid
        """
        graph.query(query)


def _merge_entity_nodes(graph, nodes: list[dict[str, Any]]) -> None:
    for node in _prepare_dicts(nodes):
        labels = node.get("labels") or ["Entity"]
        props = {
            "uuid": node["uuid"],
            "name": node.get("name", ""),
            "group_id": node.get("group_id", ""),
            "created_at": node.get("created_at"),
            "labels": json.dumps(labels),
            "summary": node.get("summary", ""),
            "attributes": json.dumps(node.get("attributes", {})),
        }
        set_clause = ", ".join(
            [f"n.{key} = {to_cypher_literal(value)}" for key, value in props.items()]
        )
        embedding_literal = json.dumps(node.get("name_embedding") or [])
        query = f"""
        MERGE (n:Entity {{uuid: {to_cypher_literal(node["uuid"])}}})
        SET {set_clause}
        SET n.name_embedding = vecf32({embedding_literal})
        RETURN n.uuid AS uuid
        """
        graph.query(query)


def _merge_entity_edges(graph, edges: list[dict[str, Any]]) -> None:
    for edge in _prepare_dicts(edges):
        props = {
            "uuid": edge["uuid"],
            "name": edge.get("name", ""),
            "fact": edge.get("fact", ""),
            "group_id": edge.get("group_id", ""),
            "created_at": edge.get("created_at"),
            "episodes": edge.get("episodes", []),
            "expired_at": edge.get("expired_at"),
            "valid_at": edge.get("valid_at"),
            "invalid_at": edge.get("invalid_at"),
            "attributes": json.dumps(edge.get("attributes", {})),
        }
        set_clause = ", ".join(
            [f"r.{key} = {to_cypher_literal(value)}" for key, value in props.items()]
        )
        embedding_literal = json.dumps(edge.get("fact_embedding") or [])
        query = f"""
        MATCH (source:Entity {{uuid: {to_cypher_literal(edge["source_node_uuid"])}}})
        MATCH (target:Entity {{uuid: {to_cypher_literal(edge["target_node_uuid"])}}})
        MERGE (source)-[r:RELATES_TO {{uuid: {to_cypher_literal(edge["uuid"])}}}]->(target)
        SET {set_clause}
        SET r.fact_embedding = vecf32({embedding_literal})
        RETURN r.uuid AS uuid
        """
        graph.query(query)


def _merge_episodic_edges(graph, edges: list[dict[str, Any]]) -> None:
    for edge in _prepare_dicts(edges):
        props = {
            "uuid": edge["uuid"],
            "group_id": edge.get("group_id", ""),
            "created_at": edge.get("created_at"),
        }
        set_clause = ", ".join(
            [f"r.{key} = {to_cypher_literal(value)}" for key, value in props.items()]
        )
        query = f"""
        MATCH (episode:Episodic {{uuid: {to_cypher_literal(edge["source_node_uuid"])}}})
        MATCH (entity:Entity {{uuid: {to_cypher_literal(edge["target_node_uuid"])}}})
        MERGE (episode)-[r:MENTIONS {{uuid: {to_cypher_literal(edge["uuid"])}}}]->(entity)
        SET {set_clause}
        RETURN r.uuid AS uuid
        """
        graph.query(query)


class FalkorLiteSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self):
        self.graph = self._ensure_graph_or_raise()

    @staticmethod
    def _ensure_graph_or_raise():
        graph = get_falkordb_graph()
        if graph is None:
            raise RuntimeError("FalkorDB Lite is unavailable")
        return graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def close(self):
        return None

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if "episodes" in kwargs:
            await asyncio.to_thread(_merge_episodes, self.graph, kwargs["episodes"])
            return None
        if "nodes" in kwargs:
            await asyncio.to_thread(_merge_entity_nodes, self.graph, kwargs["nodes"])
            return None
        if "entity_edges" in kwargs:
            await asyncio.to_thread(
                _merge_entity_edges, self.graph, kwargs["entity_edges"]
            )
            return None
        if "episodic_edges" in kwargs:
            await asyncio.to_thread(
                _merge_episodic_edges, self.graph, kwargs["episodic_edges"]
            )
            return None

        raise NotImplementedError(
            "FalkorLiteSession.run only supports bulk persistence operations"
        )


class FalkorLiteDriver(GraphDriver):
    provider = GraphProvider.FALKORDB

    def __init__(self):
        super().__init__()
        if get_falkordb_graph() is None:
            raise RuntimeError("FalkorDB Lite is unavailable")
        self._database = "falkordb-lite"

    async def execute_query(self, cypher_query_, **kwargs: Any):
        graph = _ensure_graph()
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
        result = await asyncio.to_thread(graph.query, formatted_query, None)

        raw = getattr(result, "_raw_response", None)
        records: list[dict[str, Any]] = []
        header: list[str] = []
        rows = []
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

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorLiteSession()

    async def close(self):
        return None

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        from graphiti_core.graph_queries import get_range_indices

        if delete_existing:
            await self.delete_all_indexes()

        # Get range indices from graphiti-core
        index_queries = get_range_indices(self.provider)

        # Add fulltext indices - FalkorDB Lite supports them natively
        # Using standard English stopwords (same as graphiti-core's falkordb_driver)
        stopwords = ['a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but',
                     'by', 'for', 'if', 'in', 'into', 'it', 'no', 'not', 'of', 'on',
                     'or', 'such', 'that', 'their', 'then', 'there', 'these', 'they',
                     'this', 'to', 'was', 'will', 'with']

        fulltext_queries = [
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Episodic', stopwords: {stopwords}}},
                'content', 'source', 'source_description', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Entity', stopwords: {stopwords}}},
                'name', 'summary', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Community', stopwords: {stopwords}}},
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

    async def delete_all_indexes(self):
        logger.debug("FalkorDB Lite does not expose index management APIs")

    @staticmethod
    def _sanitize_fulltext(query: str) -> str:
        separator_map = str.maketrans(
            {
                ',': ' ',
                '.': ' ',
                '<': ' ',
                '>': ' ',
                '{': ' ',
                '}': ' ',
                '[': ' ',
                ']': ' ',
                '"': ' ',
                "'": ' ',
                ':': ' ',
                ';': ' ',
                '!': ' ',
                '@': ' ',
                '#': ' ',
                '$': ' ',
                '%': ' ',
                '^': ' ',
                '&': ' ',
                '*': ' ',
                '(': ' ',
                ')': ' ',
                '-': ' ',
                '+': ' ',
                '=': ' ',
                '~': ' ',
                '?': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        return " ".join(sanitized.split())

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """Simplified fulltext query builder for local Falkor Lite."""
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


_driver: FalkorLiteDriver | None = None


class NullEmbedder(EmbedderClient):
    """Fallback embedder that returns empty vectors when embeddings are unavailable."""

    async def create(self, input_data):  # type: ignore[override]
        return [[0.0] * EMBEDDING_DIM for _ in input_data]


_embedder = NullEmbedder()


def _get_driver() -> FalkorLiteDriver:
    global _driver
    if _driver is None:
        _driver = FalkorLiteDriver()
    return _driver


def _ensure_embeddings(nodes: list[EntityNode], edges: list[EntityEdge]) -> None:
    for node in nodes:
        if node.name_embedding is None:
            node.name_embedding = []
    for edge in edges:
        if edge.fact_embedding is None:
            edge.fact_embedding = []


async def persist_episode_and_nodes(
    episode: EpisodicNode,
    nodes: list[EntityNode],
    edges: list[EntityEdge] | None = None,
    episodic_edges: list[EpisodicEdge] | None = None,
) -> dict[str, Any]:
    """Persist pipeline outputs using graphiti-core's bulk writer."""
    await ensure_graph_ready()
    await ensure_self_entity(episode.group_id or GROUP_ID)
    edges = edges or []
    episodic_edges = episodic_edges or []

    driver = _get_driver()
    _ensure_embeddings(nodes, edges)

    try:
        await add_nodes_and_edges_bulk(
            driver,
            [episode],
            episodic_edges,
            nodes,
            edges,
            _embedder,
        )

        logger.info(
            "Persisted episode %s (%d nodes, %d edges, %d episodic edges)",
            episode.uuid,
            len(nodes),
            len(edges),
            len(episodic_edges),
        )

        return {
            "status": "persisted",
            "episode_uuid": episode.uuid,
            "nodes_written": len(nodes),
            "edges_written": len(edges),
            "episodic_edges_written": len(episodic_edges),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Persistence failed")
        return {"error": str(exc)}


__all__ = [
    "FalkorLiteDriver",
    "FalkorLiteSession",
    "NullEmbedder",
    "disable_tcp_server",
    "ensure_graph_ready",
    "ensure_self_entity",
    "fetch_self_entity",
    "enable_tcp_server",
    "fetch_entities_by_group",
    "fetch_entity_edges_by_group",
    "fetch_recent_episodes",
    "get_tcp_server_endpoint",
    "get_tcp_server_password",
    "get_db_stats",
    "get_falkordb_graph",
    "get_driver",
    "persist_episode_and_nodes",
    "reset_database",
    "to_cypher_literal",
    "shutdown_falkordb",
]
