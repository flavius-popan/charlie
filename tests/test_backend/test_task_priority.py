"""Tests for task priority queueing."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from backend.settings import DEFAULT_JOURNAL


@pytest.fixture(autouse=True)
def clear_huey_queue(isolated_graph):
    """Clear Huey queue keys before each test.

    Note: Requires isolated_graph to ensure real Redis is available
    (PriorityRedisHuey needs zadd which FakeClient doesn't provide).
    """
    from backend.database.redis_ops import redis_ops
    from backend.services import queue
    from backend.database import lifecycle

    # Stop any running consumer from previous tests to prevent tasks from being processed
    queue.stop_huey_consumer(timeout=1.0)

    # Update Huey's connection pool to use the current test database
    # (isolated_graph creates a new temp DB with a different socket path)
    redis_client = getattr(lifecycle._db, "client", None)
    if redis_client:
        queue.huey.storage.pool = getattr(redis_client, "connection_pool", None)
        queue.huey.storage.conn = redis_client

    with redis_ops() as r:
        # Clear all Huey-related keys
        for key in r.scan_iter(match="huey*"):
            r.delete(key)
    yield
    with redis_ops() as r:
        for key in r.scan_iter(match="huey*"):
            r.delete(key)


def test_extract_nodes_task_accepts_priority_parameter(episode_uuid):
    """Task accepts priority parameter and passes it to Huey."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work"):
            mock_status.return_value = "pending_edges"

            # Call with priority parameter - this enqueues the task
            result = extract_nodes_task(episode_uuid, DEFAULT_JOURNAL, priority=1)

            # Result is a TaskResultWrapper, get the underlying task
            # The task instance should have priority set
            assert result.task.priority == 1


def test_extract_nodes_task_priority_defaults_to_none(episode_uuid):
    """Task defaults to priority=None when not specified."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work"):
            mock_status.return_value = "pending_edges"

            # Call without priority parameter
            result = extract_nodes_task(episode_uuid, DEFAULT_JOURNAL)

            # Task should default to priority=None (which Huey treats as lowest priority)
            assert result.task.priority is None


def test_edit_screen_enqueues_with_high_priority():
    """EditScreen._enqueue_extraction_task should pass priority=1."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.redis_ops import redis_ops

    episode_uuid = "test-uuid"
    journal = DEFAULT_JOURNAL

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work"):
            mock_status.return_value = "pending_edges"

            # Enqueue with priority=1 (simulating EditScreen behavior)
            result = extract_nodes_task(episode_uuid, journal, priority=1)

            # Verify task was enqueued with correct priority
            assert result.task.priority == 1

            # Verify in Redis - PriorityRedisHuey uses sorted sets with negative scores
            # (higher priority = lower score, so they sort first)
            with redis_ops() as r:
                queue_key = "huey.redis.charlie"
                tasks = r.zrange(queue_key, 0, -1, withscores=True)
                assert len(tasks) > 0
                # Score should be -1 (negative of priority)
                _, score = tasks[0]
                assert score == -1.0


def test_orchestrator_enqueues_with_low_priority():
    """enqueue_pending_episodes should use priority=0 for background tasks."""
    from backend.database.redis_ops import redis_ops

    with patch("backend.database.redis_ops.get_episode_status", return_value="pending_edges"):
        with patch("backend.inference.manager.cleanup_if_no_work"):
            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                with patch("backend.database.redis_ops.get_episodes_by_status", return_value=["ep1"]):
                    with patch("backend.database.redis_ops.get_episode_data", return_value={"journal": DEFAULT_JOURNAL}):
                        from backend.database.redis_ops import enqueue_pending_episodes

                        count = enqueue_pending_episodes()

                        assert count == 1

                        # Verify in Redis - PriorityRedisHuey uses sorted sets with negative scores
                        with redis_ops() as r:
                            queue_key = "huey.redis.charlie"
                            tasks = r.zrange(queue_key, 0, -1, withscores=True)
                            assert len(tasks) > 0
                            # Score should be 0 (negative of priority 0)
                            _, score = tasks[0]
                            assert score == 0.0


def test_high_priority_tasks_jump_ahead_of_low_priority():
    """High priority tasks should be dequeued before low priority tasks."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.redis_ops import redis_ops

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work"):
            mock_status.return_value = "pending_edges"

            # Enqueue 3 low-priority background tasks first
            bg1 = extract_nodes_task("bg-task-1", DEFAULT_JOURNAL, priority=0)
            bg2 = extract_nodes_task("bg-task-2", DEFAULT_JOURNAL, priority=0)
            bg3 = extract_nodes_task("bg-task-3", DEFAULT_JOURNAL, priority=0)

            # Now enqueue 1 high-priority user task
            user_task = extract_nodes_task("user-task-1", DEFAULT_JOURNAL, priority=1)

            # Verify tasks are in the queue with correct priorities
            with redis_ops() as r:
                queue_key = "huey.redis.charlie"
                # Get tasks in order they will be processed (ascending by score)
                tasks = r.zrange(queue_key, 0, -1, withscores=True)
                assert len(tasks) == 4

                # First task should be the high-priority user task (score -1)
                _, first_score = tasks[0]
                assert first_score == -1.0

                # Remaining tasks should be low-priority background tasks (score 0)
                for i in range(1, 4):
                    _, score = tasks[i]
                    assert score == 0.0
