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


def test_notify_interrupted_tasks_handles_concurrent_removal(falkordb_test_context):
    """notify_interrupted_tasks() should handle KeyError when task already removed.

    This tests the monkey-patch applied in start_huey_consumer() that makes
    notify_interrupted_tasks() thread-safe for concurrent task removal.
    """
    from backend.services.queue import huey, start_huey_consumer, stop_huey_consumer

    # Start consumer to apply the monkey-patch
    start_huey_consumer()

    try:
        # Simulate a task in _tasks_in_flight
        mock_task = Mock()
        mock_task.__str__ = Mock(return_value="test_task:123")
        huey._tasks_in_flight.add(mock_task)

        # Remove task (simulates worker's finally block completing)
        huey._tasks_in_flight.remove(mock_task)

        # Add another task that will be in the set during iteration
        mock_task2 = Mock()
        mock_task2.__str__ = Mock(return_value="test_task:456")
        huey._tasks_in_flight.add(mock_task2)

        # Now call notify_interrupted_tasks - should NOT raise KeyError
        # even though mock_task was already removed
        huey.notify_interrupted_tasks()

        # Set should be empty after notification
        assert len(huey._tasks_in_flight) == 0
    finally:
        stop_huey_consumer()
