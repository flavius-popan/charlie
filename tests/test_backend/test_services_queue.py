"""Tests for Huey queue configuration."""

from __future__ import annotations

from unittest.mock import Mock, patch

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
