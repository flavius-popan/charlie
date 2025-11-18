"""Shared fixtures for pipeline integration tests."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Iterator

import pytest
from pipeline import _dspy_setup  # noqa: F401
import dspy

from mlx_runtime import MLXDspyLM

import pipeline.falkordblite_driver as db_utils
import settings


@pytest.fixture(scope="session", autouse=True)
def configure_dspy_for_pipeline(request: pytest.FixtureRequest) -> Iterator[None]:
    """
    Configure DSPy once per test session with the MLX backend.

    Uses deterministic sampling so integration assertions stay stable.
    """
    model_path = request.config.getoption("--model")
    adapter = dspy.ChatAdapter()
    lm = MLXDspyLM(model_path=model_path, generation_config={"temp": 0.0})
    dspy.configure(lm=lm, adapter=adapter)
    yield


@pytest.fixture(scope="session")
def falkordb_test_context(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Start a single FalkorDB Lite instance for the entire test session."""
    db_dir = tmp_path_factory.mktemp("falkordb-lite")
    db_path = db_dir / "pipeline-integration.db"

    settings.DB_PATH = db_path
    db_utils.DB_PATH = db_path  # type: ignore[assignment]

    # Reset cached connections before opening a new graph.
    db_utils.shutdown_falkordb()
    db_utils._db = None  # type: ignore[assignment]
    db_utils._graph = None  # type: ignore[assignment]
    db_utils._db_unavailable = False  # type: ignore[assignment]

    graph = db_utils._ensure_graph()
    if graph is None:
        pytest.skip("FalkorDB Lite is unavailable; integration tests require it.")

    graph.query("MATCH (n) DETACH DELETE n")
    try:
        yield
    finally:
        graph.query("MATCH (n) DETACH DELETE n")
        db_utils.shutdown_falkordb()


@pytest.fixture
def isolated_graph(falkordb_test_context) -> Iterator[object]:
    """
    Provide a clean FalkorDB graph for each test without restarting the server.

    Data is truncated before and after each test to guarantee isolation.
    """
    graph = db_utils.get_falkordb_graph() or db_utils._ensure_graph()
    if graph is None:
        pytest.skip("FalkorDB Lite is unavailable; integration tests require it.")

    graph.query("MATCH (n) DETACH DELETE n")

    try:
        yield graph
    finally:
        graph.query("MATCH (n) DETACH DELETE n")


@pytest.fixture
def seed_entity(isolated_graph) -> Callable[..., None]:
    """Return a helper that inserts Entity nodes into the current test graph."""

    def _seed(*, uuid: str, name: str, group_id: str, labels: list[str] | None = None) -> None:
        entity_labels = labels or ["Entity"]
        labels_literal = ", ".join(f"'{label}'" for label in entity_labels)
        query = f"""
        CREATE (:Entity {{
            uuid: {db_utils.to_cypher_literal(uuid)},
            name: {db_utils.to_cypher_literal(name)},
            summary: {db_utils.to_cypher_literal('')},
            labels: [{labels_literal}],
            attributes: {db_utils.to_cypher_literal(json.dumps({}))},
            group_id: {db_utils.to_cypher_literal(group_id)},
            created_at: {db_utils.to_cypher_literal('2024-01-01T00:00:00Z')}
        }})
        """
        isolated_graph.query(query)

    return _seed


@pytest.fixture
def seed_episode(isolated_graph) -> Callable[..., None]:
    """Return a helper that inserts Episodic nodes for context-driven tests."""

    def _seed(
        *,
        uuid: str,
        name: str,
        group_id: str,
        content: str,
        valid_at: datetime,
    ) -> None:
        query = f"""
        CREATE (:Episodic {{
            uuid: {db_utils.to_cypher_literal(uuid)},
            name: {db_utils.to_cypher_literal(name)},
            content: {db_utils.to_cypher_literal(content)},
            source_description: {db_utils.to_cypher_literal('Seeded context')},
            valid_at: {db_utils.to_cypher_literal(valid_at.isoformat())},
            created_at: {db_utils.to_cypher_literal(valid_at.isoformat())},
            entity_edges: {db_utils.to_cypher_literal([])},
            labels: {db_utils.to_cypher_literal([])},
            group_id: {db_utils.to_cypher_literal(group_id)},
            source: {db_utils.to_cypher_literal('text')}
        }})
        """
        isolated_graph.query(query)

    return _seed
