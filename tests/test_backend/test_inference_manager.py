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
    """Keep models loaded when pending node work exists (and not editing)."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            return_value=["ep1"],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                    mock_redis = mock_redis_ops.return_value.__enter__.return_value
                    mock_redis.exists.return_value = False  # Not editing

                    manager.cleanup_if_no_work()

                    mock_status.assert_called_with("pending_nodes")
                    mock_unload.assert_not_called()


def test_cleanup_unloads_when_pending_edges_only():
    """pending_edges should not keep the model loaded (LLM used only for node extraction)."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            return_value=[],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                    mock_redis = mock_redis_ops.return_value.__enter__.return_value
                    mock_redis.exists.return_value = False

                    manager.cleanup_if_no_work()

                    mock_status.assert_called_with("pending_nodes")
                    mock_unload.assert_called_once()


def test_cleanup_unloads_when_no_active_work():
    """Unload models when no pending_nodes remain."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            return_value=[],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                    mock_redis = mock_redis_ops.return_value.__enter__.return_value
                    mock_redis.exists.return_value = False

                    manager.cleanup_if_no_work()

                    mock_status.assert_called_with("pending_nodes")
                    mock_unload.assert_called_once()


def test_cleanup_unloads_when_user_is_editing():
    """Unload models when editing:active key exists (inverted for snappy UI)."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch("backend.inference.manager.unload_all_models") as mock_unload:
            with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                mock_redis = mock_redis_ops.return_value.__enter__.return_value
                mock_redis.exists.return_value = True

                manager.cleanup_if_no_work()

                mock_redis.exists.assert_called_with("editing:active")
                mock_unload.assert_called_once()


def test_orchestrator_calls_cleanup():
    """Orchestrator should call cleanup_if_no_work on each cycle."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes") as mock_enqueue:
        with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
            orchestrate_inference_work.call_local(reschedule=False)

            mock_enqueue.assert_called_once()
            mock_cleanup.assert_called_once()


def test_is_model_loading_blocked_during_editing():
    """Model loading should be blocked when editing:active key exists."""
    import time

    with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
        mock_redis = mock_redis_ops.return_value.__enter__.return_value
        mock_redis.exists.return_value = True

        # Set startup time far enough in past that grace period has expired
        manager._app_startup_time = time.monotonic() - 100

        result = manager.is_model_loading_blocked()

        assert result is True
        mock_redis.exists.assert_called_with("editing:active")


def test_is_model_loading_blocked_during_startup_grace():
    """Model loading should be blocked during startup grace period."""
    import time

    # Set startup time to now
    manager._app_startup_time = time.monotonic()

    try:
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = False

            result = manager.is_model_loading_blocked()

            assert result is True
            # Redis should not be checked during startup grace
            mock_redis.exists.assert_not_called()
    finally:
        manager._app_startup_time = None


def test_is_model_loading_not_blocked_after_grace():
    """Model loading should NOT be blocked after startup grace expires."""
    import time

    # Set startup time to well in the past
    manager._app_startup_time = time.monotonic() - 100

    try:
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = False

            result = manager.is_model_loading_blocked()

            assert result is False
            mock_redis.exists.assert_called_with("editing:active")
    finally:
        manager._app_startup_time = None


def test_editing_presence_unloads_models_end_to_end():
    """Integration test: editing flag causes unload (inverted behavior for snappy UI)."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.inference.manager import MODELS, cleanup_if_no_work
    from backend.database.redis_ops import redis_ops
    from unittest.mock import MagicMock

    # Reset startup time so grace period doesn't interfere
    manager._app_startup_time = None

    try:
        # Step 1: Manually set a loaded model (simulating prior work)
        MODELS["llm"] = MagicMock()

        # Step 2: Simulate user editing (set editing:active key)
        with redis_ops() as r:
            r.set("editing:active", "active")

        # Step 3: Verify cleanup UNLOADS while editing (inverted behavior)
        with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
            cleanup_if_no_work()
            assert MODELS["llm"] is None, "Model should be unloaded when user is editing"

        # Step 4: Delete key and set model again
        with redis_ops() as r:
            r.delete("editing:active")
        MODELS["llm"] = MagicMock()

        # Step 5: Verify model stays loaded when not editing and work exists
        with patch("backend.database.redis_ops.get_episodes_by_status", return_value=["ep1"]):
            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                cleanup_if_no_work()
                assert MODELS["llm"] is not None, "Model should stay loaded when work exists"

    finally:
        # Clean up: ensure model cache is reset and Redis key is deleted
        MODELS["llm"] = None
        try:
            with redis_ops() as r:
                r.delete("editing:active")
        except Exception:
            pass
