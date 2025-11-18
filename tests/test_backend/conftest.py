"""Shared fixtures for backend API tests (no DSPy, database only)."""

from __future__ import annotations

from typing import Iterator
from uuid import uuid4

import pytest

import backend.database as db_utils
from backend import settings as backend_settings


@pytest.fixture(scope="session")
def falkordb_test_context(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Start a single FalkorDB Lite instance for the entire test session."""
    db_dir = tmp_path_factory.mktemp("falkordb-lite")
    db_path = db_dir / "backend-tests.db"

    backend_settings.DB_PATH = db_path
    # Update DB_PATH in lifecycle module
    import backend.database.lifecycle as lifecycle
    lifecycle.DB_PATH = db_path  # type: ignore[misc]

    # Reset cached connections before opening a new graph
    db_utils.shutdown_database()

    try:
        yield
    finally:
        db_utils.shutdown_database()


@pytest.fixture
def isolated_graph(falkordb_test_context) -> Iterator[object]:
    """
    Provide a clean FalkorDB graph for each test.

    Uses DEFAULT_JOURNAL and clears all data before/after each test.
    """
    from backend.settings import DEFAULT_JOURNAL

    graph = db_utils.get_falkordb_graph(DEFAULT_JOURNAL)

    # Clear all data
    graph.query("MATCH (n) DETACH DELETE n")

    # Reset graph initialization state for clean tests
    import backend.database.persistence as persistence
    persistence._graph_initialized.clear()
    persistence._seeded_self_groups.clear()

    try:
        yield graph
    finally:
        # Clear all data after test
        graph.query("MATCH (n) DETACH DELETE n")
        persistence._graph_initialized.clear()
        persistence._seeded_self_groups.clear()
