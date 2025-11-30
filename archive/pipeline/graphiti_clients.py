"""Helpers for constructing Graphiti client bundles with local backends."""

from __future__ import annotations

import logging
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.tracer import NoOpTracer

from pipeline.falkordblite_driver import NullEmbedder, get_driver
from pipeline.local_llm_client import LocalMLXLLMClient

logger = logging.getLogger(__name__)


class NullCrossEncoder(CrossEncoderClient):
    """No-op cross encoder that returns empty rankings."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return []


_graphiti_clients: GraphitiClients | None = None


def get_graphiti_clients() -> GraphitiClients:
    """Return a lazily constructed GraphitiClients bundle."""
    global _graphiti_clients
    if _graphiti_clients is not None:
        return _graphiti_clients

    driver = get_driver()
    llm_client = LocalMLXLLMClient()
    embedder = NullEmbedder()
    cross_encoder = NullCrossEncoder()
    tracer = NoOpTracer()

    _graphiti_clients = GraphitiClients(
        driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        tracer=tracer,
    )

    logger.info(
        "Initialized Graphiti clients with local MLX LLM (%s)",
        llm_client.model_path,
    )

    return _graphiti_clients


__all__ = ["get_graphiti_clients", "NullCrossEncoder"]
