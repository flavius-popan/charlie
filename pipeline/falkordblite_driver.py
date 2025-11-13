"""GraphDriver implementation backed by the embedded FalkorDB Lite instance.

This adapter allows us to call graphiti-core's bulk persistence utilities
(`add_nodes_and_edges_bulk`) even though FalkorDB Lite lacks parameterized
queries and asynchronous APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

from pipeline.db_utils import get_falkordb_graph, to_cypher_literal

logger = logging.getLogger(__name__)


def _ensure_graph_or_raise():
    graph = get_falkordb_graph()
    if graph is None:
        raise RuntimeError("FalkorDB Lite is unavailable")
    return graph


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
        self.graph = _ensure_graph_or_raise()

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
            await asyncio.to_thread(_merge_entity_edges, self.graph, kwargs["entity_edges"])
            return None
        if "episodic_edges" in kwargs:
            await asyncio.to_thread(
                _merge_episodic_edges, self.graph, kwargs["episodic_edges"]
            )
            return None

        raise NotImplementedError("FalkorLiteSession.run only supports bulk persistence operations")


class FalkorLiteDriver(GraphDriver):
    provider = GraphProvider.FALKORDB

    def __init__(self):
        super().__init__()
        _ensure_graph_or_raise()
        self._database = "falkordb-lite"

    async def execute_query(self, cypher_query_, **kwargs: Any):
        raise NotImplementedError("Direct query execution is not implemented for FalkorLiteDriver")

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorLiteSession()

    async def close(self):
        return None

    async def delete_all_indexes(self):
        logger.debug("FalkorDB Lite does not expose index management APIs")
