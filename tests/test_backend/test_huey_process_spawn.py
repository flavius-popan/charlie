from __future__ import annotations

from unittest.mock import patch

from charlie import CharlieApp


def test_ensure_huey_worker_running_starts_consumer_when_not_running():
    app = CharlieApp()

    with patch("charlie.is_huey_consumer_running", return_value=False):
        with patch("charlie.start_huey_consumer") as mock_start:
            with patch("charlie.atexit.register"):
                app._ensure_huey_worker_running()

    mock_start.assert_called_once()


def test_ensure_huey_worker_running_noop_when_already_running():
    app = CharlieApp()

    with patch("charlie.is_huey_consumer_running", return_value=True):
        with patch("charlie.start_huey_consumer") as mock_start:
            with patch("charlie.atexit.register"):
                app._ensure_huey_worker_running()

    mock_start.assert_not_called()
