"""Async database utilities for pipeline modules."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
from datetime import datetime
from threading import Lock
from typing import Any, Iterable

from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import utc_now

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


async def fetch_entities_by_group(group_id: str) -> dict[str, EntityNode]:
    """Fetch all entities in the given group keyed by UUID."""
    return await asyncio.to_thread(_fetch_entities_by_group_sync, group_id)


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


def _write_episode_and_nodes_sync(
    episode: EpisodicNode,
    nodes: list[EntityNode],
) -> dict[str, Any]:
    """Write episode and entity nodes to database (sync)."""
    graph = _ensure_graph()
    if graph is None:
        return {"error": "Database unavailable"}

    try:
        from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

        episode_dict = convert_datetimes_to_strings(
            {
                "uuid": episode.uuid,
                "name": episode.name,
                "group_id": episode.group_id,
                "source": episode.source.value,
                "source_description": episode.source_description,
                "content": episode.content,
                "valid_at": episode.valid_at,
                "created_at": episode.created_at,
                "entity_edges": episode.entity_edges or [],
                "labels": episode.labels or [],
            }
        )

        episode_props = {
            "uuid": episode_dict["uuid"],
            "name": episode_dict["name"],
            "group_id": episode_dict["group_id"],
            "source": episode_dict["source"],
            "source_description": episode_dict["source_description"],
            "content": episode_dict["content"],
            "valid_at": episode_dict["valid_at"],
            "created_at": episode_dict["created_at"],
            "entity_edges": json.dumps(episode_dict["entity_edges"]),
            "labels": json.dumps(episode_dict["labels"]),
        }

        episode_set_clause = ", ".join(
            [f"e.{key} = {to_cypher_literal(value)}" for key, value in episode_props.items()]
        )

        episode_query = f"""
        MERGE (e:Episodic {{uuid: {to_cypher_literal(episode_dict['uuid'])}}})
        SET {episode_set_clause}
        RETURN e.uuid AS uuid
        """

        graph.query(episode_query)
        logger.info("Wrote episode node: %s", episode.uuid)

        nodes_created = 0
        node_uuids = []

        for node in nodes:
            node_dict = convert_datetimes_to_strings(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "group_id": node.group_id,
                    "created_at": node.created_at,
                    "labels": node.labels or ["Entity"],
                    "summary": node.summary or "",
                    "attributes": node.attributes or {},
                    "name_embedding": node.name_embedding or [],
                }
            )

            props = {
                "uuid": node_dict["uuid"],
                "name": node_dict["name"],
                "group_id": node_dict["group_id"],
                "created_at": node_dict["created_at"],
                "labels": json.dumps(node_dict["labels"]),
                "summary": node_dict["summary"],
                "attributes": json.dumps(node_dict["attributes"]),
            }

            set_clause = ", ".join([f"n.{key} = {to_cypher_literal(value)}" for key, value in props.items()])

            embedding_literal = json.dumps(node_dict["name_embedding"])

            node_query = f"""
            MERGE (n:Entity {{uuid: {to_cypher_literal(node_dict['uuid'])}}})
            SET {set_clause}
            SET n.name_embedding = vecf32({embedding_literal})
            RETURN n.uuid AS uuid
            """

            result = graph.query(node_query)
            if result.result_set:
                nodes_created += 1
                node_uuids.append(node_dict["uuid"])

        logger.info("Wrote %d entity nodes", nodes_created)

        return {
            "episode_uuid": episode.uuid,
            "nodes_created": nodes_created,
            "node_uuids": node_uuids,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write episode and nodes")
        return {"error": str(exc)}


async def write_episode_and_nodes(
    episode: EpisodicNode,
    nodes: list[EntityNode],
) -> dict[str, Any]:
    """Write episode and entity nodes to database."""
    return await asyncio.to_thread(_write_episode_and_nodes_sync, episode, nodes)


__all__ = [
    "fetch_entities_by_group",
    "fetch_recent_episodes",
    "get_db_stats",
    "reset_database",
    "write_episode_and_nodes",
]
