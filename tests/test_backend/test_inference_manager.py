"""Tests for inference manager cleanup logic."""

from __future__ import annotations

from unittest.mock import patch

from backend.inference import manager


def test_cleanup_unloads_when_inference_disabled():
    """Models should unload immediately when inference is disabled."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=False):
        with patch("backend.inference.manager.unload_all_models") as mock_unload:
            with patch("backend.database.redis_ops.get_episodes_by_status") as mock_status:
                manager.cleanup_if_no_work()

                mock_unload.assert_called_once()
                mock_status.assert_not_called()


def test_cleanup_keeps_models_when_active_work():
    """Keep models loaded when pending work exists."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[["ep1"], []],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                manager.cleanup_if_no_work()

                mock_status.assert_any_call("pending_nodes")
                mock_status.assert_any_call("pending_edges")
                mock_unload.assert_not_called()


def test_cleanup_keeps_models_when_pending_edges_only():
    """pending_edges are treated as active work."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[[], ["edge-only"]],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                manager.cleanup_if_no_work()

                mock_status.assert_any_call("pending_nodes")
                mock_status.assert_any_call("pending_edges")
                mock_unload.assert_not_called()


def test_cleanup_unloads_when_no_active_work():
    """Unload models when no pending_nodes or pending_edges remain."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[[], []],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                manager.cleanup_if_no_work()

                mock_status.assert_any_call("pending_nodes")
                mock_status.assert_any_call("pending_edges")
                mock_unload.assert_called_once()
