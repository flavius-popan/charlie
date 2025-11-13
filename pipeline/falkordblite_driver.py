"""FalkorDB Lite driver plus consolidated persistence and query utilities."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
from datetime import datetime
from threading import Lock
from typing import Any, Iterable

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings, utc_now

from settings import DB_PATH, GRAPH_NAME

try:  # Optional dependency in CI / unit tests
    from redislite.falkordb_client import FalkorDB
except Exception:  # noqa: BLE001
    FalkorDB = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_db_lock = Lock()
_db = None
_graph = None
_db_unavailable = False

# Ensure the embedded database directory exists before first use.
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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
            _db = FalkorDB(dbfilename=str(DB_PATH))
            _graph = _db.select_graph(GRAPH_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise FalkorDBLite: %s", exc)
            _db_unavailable = True
            return None

    return _graph


def _close_db():
    """Ensure the embedded FalkorDB process shuts down cleanly."""
    global _db, _graph
    with _db_lock:
        if _db is None:
            return
        try:
            _db.close()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close FalkorDBLite cleanly", exc_info=True)
        finally:
            _db = None
            _graph = None


atexit.register(_close_db)


def get_falkordb_graph():
    """Return the shared FalkorDB graph instance (None if unavailable)."""
    return _ensure_graph()


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
    WHERE n.group_id = {group_literal}
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
    return await asyncio.to_thread(_fetch_entities_by_group_sync, group_id)


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
    return await asyncio.to_thread(_reset_database_sync)


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
        raise NotImplementedError(
            "Direct query execution is not implemented for FalkorLiteDriver"
        )

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorLiteSession()

    async def close(self):
        return None

    async def delete_all_indexes(self):
        logger.debug("FalkorDB Lite does not expose index management APIs")


_driver: FalkorLiteDriver | None = None


class NullEmbedder(EmbedderClient):
    """Fallback embedder that returns empty vectors when embeddings are unavailable."""

    async def create(self, input_data):  # type: ignore[override]
        return []


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
    "fetch_entities_by_group",
    "fetch_entity_edges_by_group",
    "fetch_recent_episodes",
    "get_db_stats",
    "get_falkordb_graph",
    "persist_episode_and_nodes",
    "reset_database",
    "to_cypher_literal",
]
