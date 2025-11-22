"""Integration tests for running the Huey consumer in-process."""

from __future__ import annotations

import time
import asyncio
from unittest.mock import patch

import pytest

from backend.settings import DEFAULT_JOURNAL


def _wait_for_result(result, timeout: float = 5.0, poll: float = 0.05):
    """Poll a Huey result until it is ready or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = result(blocking=False)
        if value is not None:
            return value
        time.sleep(poll)
    raise AssertionError("Timed out waiting for Huey task result")


def test_inprocess_consumer_uses_same_redis_instance(falkordb_test_context):
    """Consumer thread should reuse the existing embedded Redis (no new pid)."""
    from backend.database import lifecycle
    from backend.services import queue

    lifecycle._ensure_graph(DEFAULT_JOURNAL)
    initial_pid = getattr(lifecycle._db.client, "pid", None)

    @queue.huey.task()
    def ping():
        return "pong"

    try:
        queue.start_huey_consumer()
        result = ping()

        assert _wait_for_result(result, timeout=3.0) == "pong"
        assert getattr(lifecycle._db.client, "pid", None) == initial_pid
    finally:
        queue.stop_huey_consumer()


def test_start_consumer_is_idempotent(falkordb_test_context):
    """Starting the consumer twice should not spawn new Redis or a second worker."""
    from backend.database import lifecycle
    from backend.services import queue

    lifecycle._ensure_graph(DEFAULT_JOURNAL)
    initial_pid = getattr(lifecycle._db.client, "pid", None)

    try:
        queue.start_huey_consumer()
        queue.start_huey_consumer()  # second call should be a no-op

        assert getattr(lifecycle._db.client, "pid", None) == initial_pid
        assert queue.is_huey_consumer_running()
    finally:
        queue.stop_huey_consumer()


def test_delete_episode_cleans_redis_and_skips_reenqueue(falkordb_test_context):
    """Deleting an episode should remove its Redis hash and prevent re-enqueue."""
    from backend import add_journal_entry
    from backend.database.persistence import delete_episode
    from backend.database.redis_ops import (
        enqueue_pending_episodes,
        get_episode_status,
        set_episode_status,
        redis_ops,
    )

    # Isolate Redis journal keys for this test
    with redis_ops() as r:
        keys = list(r.scan_iter(match="journal:*"))
        if keys:
            r.delete(*keys)

    content = "Short entry to test deletion cleanup."
    episode_uuid = asyncio.run(add_journal_entry(content))

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    asyncio.run(delete_episode(episode_uuid, DEFAULT_JOURNAL))

    assert get_episode_status(episode_uuid, DEFAULT_JOURNAL) is None
    enqueued = enqueue_pending_episodes()
    assert enqueued == 0


def test_inference_toggle_defers_work_to_orchestrator(falkordb_test_context):
    """Toggle only flips the flag; unload happens via cleanup, enqueue via orchestrator."""
    from backend.database.redis_ops import (
        get_episode_status,
        set_episode_status,
        set_inference_enabled,
        redis_ops,
    )
    from backend.inference import manager
    from backend.services.tasks import orchestrate_inference_work

    # Isolate Redis journal keys for this test
    with redis_ops() as r:
        keys = list(r.scan_iter(match="journal:*"))
        if keys:
            r.delete(*keys)

    episode_uuid = "episode-toggle-test"

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    set_inference_enabled(False)

    with patch("backend.inference.manager.unload_all_models") as mock_unload:
        manager.cleanup_if_no_work()
        mock_unload.assert_called_once()

    set_inference_enabled(True)

    with patch("backend.services.tasks.extract_nodes_task") as mock_task:
        orchestrate_inference_work.call_local(reschedule=False)

        mock_task.assert_called_once_with(episode_uuid, DEFAULT_JOURNAL, priority=0)

    # Cleanup
    set_inference_enabled(True)
    from backend.database.redis_ops import remove_episode_from_queue

    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)


def test_start_huey_consumer_schedules_orchestrator_once(monkeypatch):
    """start_huey_consumer should kick off orchestrator scheduling without duplicate threads."""
    from backend.services import queue

    # Fake Redis client with required attributes
    class FakeClient:
        connection_pool = object()

        def ping(self):
            return True

        def zadd(self, key, mapping):
            # Stub for PriorityRedisHuey support
            return len(mapping)

    class FakeDB:
        client = FakeClient()

    # Stub lifecycle to provide fake DB/graph
    import backend.database.lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, "_db", FakeDB())
    monkeypatch.setattr(lifecycle, "_ensure_graph", lambda journal: (None, None))

    # Avoid starting real consumer threads
    class DummyConsumer:
        def __init__(*args, **kwargs):
            pass

        def run(self):
            return

        def stop(self, graceful=True):
            return

    calls = {}

    def dummy_thread(target, name=None, daemon=None):
        calls["thread_started"] = True

        class DummyThread:
            def __init__(self, target):
                self._target = target

            def start(self):
                # Don't actually spin a thread; just note it.
                calls["thread_run"] = True

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return

        return DummyThread(target)

    monkeypatch.setattr(queue, "InProcessConsumer", DummyConsumer)
    monkeypatch.setattr(queue, "Thread", dummy_thread)

    with patch("backend.services.tasks.orchestrate_inference_work.schedule") as mock_schedule:
        # Reset globals
        queue._consumer = None
        queue._consumer_thread = None

        queue.start_huey_consumer()

        mock_schedule.assert_called_once_with(delay=0)
        assert calls.get("thread_started")
