"""Integration tests for running the Huey consumer in-process."""

from __future__ import annotations

import time

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
