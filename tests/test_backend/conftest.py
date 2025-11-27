"""Shared fixtures for backend API tests."""

from __future__ import annotations

from typing import Iterator
from uuid import uuid4

import pytest

import backend.dspy_cache  # noqa: F401  # sets DSPY cache env vars before dspy import
import dspy

import backend.database as db_utils
from backend import settings as backend_settings
from backend.settings import MODEL_REPO_ID


@pytest.fixture(scope="session")
def falkordb_test_context(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Start a single FalkorDB Lite instance for the entire test session."""
    db_dir = tmp_path_factory.mktemp("falkordb-lite")
    db_path = db_dir / "backend-tests.db"

    # Disable TCP listener in tests to avoid clashing with a running app instance.
    backend_settings.REDIS_TCP_ENABLED = False
    import backend.database.lifecycle as lifecycle
    lifecycle.REDIS_TCP_ENABLED = False  # type: ignore[misc]

    backend_settings.DB_PATH = db_path
    # Update DB_PATH in lifecycle module
    lifecycle.DB_PATH = db_path  # type: ignore[misc]

    # Reset cached connections before opening a new graph
    db_utils.shutdown_database()
    lifecycle.reset_lifecycle_state()

    try:
        yield
    finally:
        db_utils.shutdown_database()
        lifecycle.reset_lifecycle_state()


@pytest.fixture
def isolated_graph(falkordb_test_context) -> Iterator[object]:
    """
    Provide a clean FalkorDB graph for each test.

    Uses DEFAULT_JOURNAL and clears all data before/after each test.
    Also resets lifecycle flags to ensure test isolation.
    """
    from backend.settings import DEFAULT_JOURNAL
    import backend.database.lifecycle as lifecycle

    # Reset lifecycle state BEFORE test (in case previous test left dirty state)
    lifecycle.reset_lifecycle_state()

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
        # Reset lifecycle state AFTER test
        lifecycle.reset_lifecycle_state()
        # Clear all data after test
        graph.query("MATCH (n) DETACH DELETE n")
        persistence._graph_initialized.clear()
        persistence._seeded_self_groups.clear()


@pytest.fixture(scope="session", autouse=True)
def configure_dspy_for_backend(request: pytest.FixtureRequest) -> Iterator[None]:
    """Configure DSPy once per test session with llama.cpp backend.

    Uses deterministic sampling so integration assertions stay stable.
    Only runs when inference tests are selected (e.g., `-m inference` or
    `-m ""`).
    """
    marker_expr = (request.config.getoption("-m") or "").strip()
    wants_inference = (
        marker_expr == ""  # default when caller overrides addopts with -m ""
        or marker_expr == "inference"
        or ("inference" in marker_expr and "not inference" not in marker_expr)
    )

    if wants_inference:
        from backend.inference import DspyLM
        from backend.settings import MODEL_CONFIG

        adapter = dspy.ChatAdapter()
        lm = DspyLM(
            repo_id=MODEL_REPO_ID,
            generation_config={**MODEL_CONFIG, "temp": 0.0},
        )
        dspy.configure(lm=lm, adapter=adapter)

    yield


@pytest.fixture(scope="session", autouse=True)
def reuse_session_lm(configure_dspy_for_backend) -> Iterator[None]:
    """Share the session-loaded LLM with the inference manager cache.

    Ensures inference tests reuse a single loaded model unless a test
    intentionally exercises load/unload behavior (those use reset_model_manager).
    """
    from backend.inference import manager

    if dspy.settings.lm is not None:
        manager.MODELS["llm"] = dspy.settings.lm  # warm cache with shared model

    yield


@pytest.fixture
def require_llm():
    """Skip test if no LLM is configured in dspy.settings.lm."""
    if dspy.settings.lm is None:
        pytest.skip("No LLM configured")


@pytest.fixture
def episode_uuid():
    """Generate valid test episode UUID."""
    return str(uuid4())


@pytest.fixture
def cleanup_test_episodes(episode_uuid):
    """Clean up test episode data from Redis after each test."""
    from backend.database.redis_ops import remove_episode_from_queue
    from backend.settings import DEFAULT_JOURNAL

    yield
    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)


@pytest.fixture
def reset_model_manager():
    """Reset model manager state before and after each test."""
    from backend.inference import manager

    manager.MODELS["llm"] = None
    yield
    manager.MODELS["llm"] = None


def _reset_huey_state():
    """Reset Huey storage state after consumer tests."""
    from backend.services.queue import huey, SafeTaskSet
    from backend.inference import manager
    from backend.database.redis_ops import redis_ops

    try:
        huey.storage.close()
    except Exception:
        pass
    huey.storage.pool = None
    huey.storage.conn = None

    # Reset _tasks_in_flight to regular set if it was wrapped
    if isinstance(huey._tasks_in_flight, SafeTaskSet):
        huey._tasks_in_flight = set(huey._tasks_in_flight)

    # Clear editing state (mirrors consumer startup behavior)
    try:
        with redis_ops() as r:
            r.delete("editing:active")
    except Exception:
        pass

    # Reset startup grace period to avoid blocking model loading in subsequent tests
    manager._app_startup_time = None


@pytest.fixture
def huey_consumer(isolated_graph):
    """Start Huey consumer for test, with proper cleanup.

    Mocks orchestrator scheduling to prevent inference tasks from running.
    Resets Huey storage state after test to avoid polluting subsequent tests.
    """
    from unittest.mock import patch
    from backend.services import queue

    with patch("backend.services.tasks.orchestrate_inference_work.schedule"):
        queue.start_huey_consumer()
        try:
            yield queue
        finally:
            queue.stop_huey_consumer()
            _reset_huey_state()


@pytest.fixture
def huey_consumer_with_orchestrator(isolated_graph):
    """Start Huey consumer with orchestrator enabled (for inference tests).

    Unlike huey_consumer, this does NOT mock the orchestrator - tasks will run.
    Use for integration tests that need real task processing.
    """
    from backend.services import queue

    queue.start_huey_consumer()
    try:
        yield queue
    finally:
        queue.stop_huey_consumer()
        _reset_huey_state()
