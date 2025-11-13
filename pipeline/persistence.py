"""Persistence helpers for writing pipeline results via graphiti-core utilities."""

from __future__ import annotations

import logging
from typing import Any

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

from .falkordblite_driver import FalkorLiteDriver

logger = logging.getLogger(__name__)

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


def _ensure_embeddings(
    nodes: list[EntityNode], edges: list[EntityEdge]
) -> None:
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
