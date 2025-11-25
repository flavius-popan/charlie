"""Tests for Huey queue configuration."""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock

import pytest


def test_get_redis_connection_extracts_pool(falkordb_test_context):
    """_get_redis_connection extracts connection pool from FalkorDB."""
    from backend.services.queue import _get_redis_connection

    pool = _get_redis_connection()

    assert pool is not None
    assert hasattr(pool, "connection_class")


def test_get_redis_connection_raises_if_db_not_initialized():
    """_get_redis_connection raises RuntimeError if database not initialized."""
    from backend.services.queue import _get_redis_connection

    with patch("backend.database.lifecycle._db", None):
        with pytest.raises(RuntimeError, match="Database not initialized"):
            _get_redis_connection()


def test_get_redis_connection_raises_if_client_unavailable(falkordb_test_context):
    """_get_redis_connection raises RuntimeError if Redis client unavailable."""
    from backend.services.queue import _get_redis_connection

    with patch("backend.database.lifecycle._db") as mock_db:
        mock_db.client = None
        with pytest.raises(RuntimeError, match="Redis client unavailable"):
            _get_redis_connection()


def test_get_redis_connection_raises_if_pool_unavailable(falkordb_test_context):
    """_get_redis_connection raises RuntimeError if connection pool unavailable."""
    from backend.services.queue import _get_redis_connection

    with patch("backend.database.lifecycle._db") as mock_db:
        mock_client = Mock()
        mock_client.connection_pool = None
        mock_db.client = mock_client
        with pytest.raises(RuntimeError, match="Redis connection pool unavailable"):
            _get_redis_connection()


def test_huey_instance_created(falkordb_test_context):
    """Huey instance is created with correct configuration."""
    from backend.services.queue import huey

    assert huey is not None
    assert huey.name == "charlie"
    assert hasattr(huey, "task")


def test_safe_task_set_remove_is_idempotent(falkordb_test_context):
    """SafeTaskSet.remove() should not raise KeyError (uses discard internally).

    This tests that start_huey_consumer() replaces _tasks_in_flight with
    SafeTaskSet, making remove() safe for concurrent access.
    """
    from backend.services.queue import huey, start_huey_consumer, stop_huey_consumer, SafeTaskSet

    # Start consumer to apply SafeTaskSet replacement
    start_huey_consumer()

    try:
        # Verify _tasks_in_flight is a SafeTaskSet
        assert isinstance(huey._tasks_in_flight, SafeTaskSet)

        # Simulate a task in _tasks_in_flight
        mock_task = Mock()
        mock_task.__str__ = Mock(return_value="test_task:123")
        huey._tasks_in_flight.add(mock_task)

        # Remove task twice - second remove should NOT raise KeyError
        huey._tasks_in_flight.remove(mock_task)
        huey._tasks_in_flight.remove(mock_task)  # Would raise KeyError with regular set

        assert mock_task not in huey._tasks_in_flight
    finally:
        stop_huey_consumer()


def test_start_huey_consumer_is_idempotent(falkordb_test_context):
    """Calling start_huey_consumer() multiple times should only start one consumer."""
    from backend.services import queue

    queue.start_huey_consumer()
    first_thread = queue._consumer_thread

    queue.start_huey_consumer()  # Should be a no-op
    second_thread = queue._consumer_thread

    try:
        assert first_thread is second_thread
        assert first_thread.is_alive()
    finally:
        queue.stop_huey_consumer()
