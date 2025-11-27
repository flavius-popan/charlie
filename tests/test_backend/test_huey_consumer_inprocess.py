"""Integration tests for running the Huey consumer in-process.

Fast mocked tests run by default. Real inference tests are marked with
@pytest.mark.inference and require `-m inference` to run.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from backend.settings import DEFAULT_JOURNAL


@pytest.fixture
def clean_redis_queue(isolated_graph):
    """Clean Redis queue state before and after test."""
    from backend.database.redis_ops import redis_ops

    def _clean():
        with redis_ops() as r:
            keys = list(r.scan_iter(match="journal:*"))
            if keys:
                r.delete(*keys)

    _clean()
    yield
    _clean()


def _wait_for_result(result, timeout: float = 5.0, poll: float = 0.05):
    """Poll a Huey result until it is ready or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = result(blocking=False)
        if value is not None:
            return value
        time.sleep(poll)
    raise AssertionError("Timed out waiting for Huey task result")


# --- Fast mocked tests (no GPU required) ---


def test_inprocess_consumer_uses_same_redis_instance(huey_consumer):
    """Consumer thread should reuse the existing embedded Redis (no new pid)."""
    from backend.database import lifecycle

    initial_pid = getattr(lifecycle._db.client, "pid", None)

    @huey_consumer.huey.task()
    def ping():
        return "pong"

    result = ping()

    assert _wait_for_result(result, timeout=3.0) == "pong"
    assert getattr(lifecycle._db.client, "pid", None) == initial_pid


def test_start_consumer_is_idempotent(huey_consumer):
    """Starting the consumer twice should not spawn new Redis or a second worker."""
    from backend.database import lifecycle

    initial_pid = getattr(lifecycle._db.client, "pid", None)

    huey_consumer.start_huey_consumer()  # second call should be a no-op

    assert getattr(lifecycle._db.client, "pid", None) == initial_pid
    assert huey_consumer.is_huey_consumer_running()


def test_delete_episode_cleans_redis_and_skips_reenqueue(clean_redis_queue):
    """Deleting an episode should remove its Redis hash and prevent re-enqueue."""
    from backend import add_journal_entry
    from backend.database.persistence import delete_episode
    from backend.database.redis_ops import (
        enqueue_pending_episodes,
        get_episode_status,
        set_episode_status,
    )

    content = "Short entry to test deletion cleanup."
    episode_uuid = asyncio.run(add_journal_entry(content))

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    asyncio.run(delete_episode(episode_uuid, DEFAULT_JOURNAL))

    assert get_episode_status(episode_uuid, DEFAULT_JOURNAL) is None

    with patch("backend.services.tasks.extract_nodes_task"):
        enqueued = enqueue_pending_episodes()
    assert enqueued == 0


def test_inference_toggle_defers_work_to_orchestrator(clean_redis_queue):
    """Toggle only flips the flag; unload happens via cleanup, enqueue via orchestrator."""
    from backend.database.redis_ops import (
        set_episode_status,
        set_inference_enabled,
        remove_episode_from_queue,
    )
    from backend.inference import manager
    from backend.services.tasks import orchestrate_inference_work

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

    set_inference_enabled(True)
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


# --- Real inference tests (require GPU, run with -m inference) ---


@pytest.mark.inference
def test_consumer_processes_real_extraction_task(
    huey_consumer_with_orchestrator, clean_redis_queue, require_llm
):
    """Integration: consumer processes extract_nodes_task with real LLM."""
    from backend import add_journal_entry
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend.services.tasks import extract_nodes_task

    content = "I met Sarah at the park today."
    episode_uuid = asyncio.run(add_journal_entry(content))
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    result = extract_nodes_task(episode_uuid, DEFAULT_JOURNAL)

    value = _wait_for_result(result, timeout=30.0)
    assert value is not None
    assert value.get("episode_uuid") == episode_uuid
    assert isinstance(value.get("extracted_count"), int)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status in ("pending_edges", "done")


@pytest.mark.inference
def test_orchestrator_enqueues_and_processes_pending_episodes(
    huey_consumer_with_orchestrator, clean_redis_queue, require_llm
):
    """Integration: orchestrator finds pending episodes and triggers extraction."""
    from backend import add_journal_entry
    from backend.database.redis_ops import (
        get_episode_status,
        set_episode_status,
        set_inference_enabled,
    )
    from backend.services.tasks import orchestrate_inference_work

    content = "Bob visited the coffee shop on Main Street."
    episode_uuid = asyncio.run(add_journal_entry(content))
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    set_inference_enabled(True)

    orchestrate_inference_work.call_local(reschedule=False)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
        if status in ("pending_edges", "done"):
            break
        time.sleep(0.5)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status in ("pending_edges", "done"), f"Expected processed status, got {status}"
    set_inference_enabled(True)
