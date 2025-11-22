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
    """Keep models loaded when pending node work exists."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[["ep1"], ["edge-pending"]],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                manager.cleanup_if_no_work()

                mock_status.assert_any_call("pending_nodes")
                mock_unload.assert_not_called()


def test_cleanup_unloads_when_pending_edges_only():
    """pending_edges should not keep the model loaded (LLM used only for node extraction)."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[[], ["edge-only"]],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                manager.cleanup_if_no_work()

                mock_status.assert_any_call("pending_nodes")
                mock_status.assert_any_call("pending_edges")
                mock_unload.assert_called_once()


def test_cleanup_unloads_when_no_active_work():
    """Unload models when no pending_nodes or pending_edges remain."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            side_effect=[[], []],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                    mock_redis = mock_redis_ops.return_value.__enter__.return_value
                    mock_redis.exists.return_value = False

                    manager.cleanup_if_no_work()

                    mock_status.assert_any_call("pending_nodes")
                    mock_status.assert_any_call("pending_edges")
                    mock_unload.assert_called_once()


def test_cleanup_keeps_models_when_user_is_editing():
    """Keep models loaded when editing:active key exists."""
    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch(
            "backend.database.redis_ops.get_episodes_by_status",
            return_value=[],
        ) as mock_status:
            with patch("backend.inference.manager.unload_all_models") as mock_unload:
                with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                    mock_redis = mock_redis_ops.return_value.__enter__.return_value
                    mock_redis.exists.return_value = True

                    manager.cleanup_if_no_work()

                    mock_redis.exists.assert_called_with("editing:active")
                    mock_unload.assert_not_called()


def test_orchestrator_loads_models_when_user_is_editing():
    """Orchestrator should pre-load models when editing:active key exists."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes"):
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = True

            with patch("backend.inference.manager.get_model") as mock_get_model:
                with patch("backend.inference.manager.cleanup_if_no_work"):
                    orchestrate_inference_work.call_local(reschedule=False)

                    mock_redis.exists.assert_called_with("editing:active")
                    mock_get_model.assert_called_once_with("llm")


def test_orchestrator_skips_loading_when_not_editing():
    """Orchestrator should not load models when editing:active key doesn't exist."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes"):
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = False

            with patch("backend.inference.manager.get_model") as mock_get_model:
                with patch("backend.inference.manager.cleanup_if_no_work"):
                    orchestrate_inference_work.call_local(reschedule=False)

                    mock_redis.exists.assert_called_with("editing:active")
                    mock_get_model.assert_not_called()


def test_orchestrator_handles_redis_errors_gracefully():
    """Orchestrator should continue working even if Redis check fails."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes"):
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis_ops.side_effect = Exception("Redis connection failed")

            with patch("backend.inference.manager.get_model") as mock_get_model:
                with patch("backend.inference.manager.cleanup_if_no_work"):
                    # Should not raise exception
                    orchestrate_inference_work.call_local(reschedule=False)

                    # Model loading should be skipped due to error
                    mock_get_model.assert_not_called()


def test_orchestrator_reuses_cached_model_if_already_loaded():
    """Orchestrator should use cached model if already loaded (idempotent)."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.inference.manager import MODELS
    from unittest.mock import MagicMock

    # Pre-load a mock model into the cache
    mock_model = MagicMock()
    MODELS["llm"] = mock_model

    try:
        with patch("backend.database.redis_ops.enqueue_pending_episodes"):
            with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                mock_redis = mock_redis_ops.return_value.__enter__.return_value
                mock_redis.exists.return_value = True

                # Use a spy to track get_model calls without replacing behavior
                from backend.inference.manager import get_model as real_get_model
                with patch("backend.inference.manager.get_model", wraps=real_get_model) as spy:
                    with patch("backend.inference.manager.cleanup_if_no_work"):
                        orchestrate_inference_work.call_local(reschedule=False)

                        # Should call get_model
                        spy.assert_called_once_with("llm")
                        # Should return the cached model (not reload)
                        assert MODELS["llm"] is mock_model
    finally:
        # Clean up: reset model cache
        MODELS["llm"] = None


def test_orchestrator_handles_model_loading_errors_gracefully():
    """Orchestrator should continue if model loading fails during editing."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes"):
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = True

            with patch("backend.inference.manager.get_model") as mock_get_model:
                mock_get_model.side_effect = RuntimeError("CUDA out of memory")

                with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                    # Should not raise exception, should log warning
                    orchestrate_inference_work.call_local(reschedule=False)

                    # cleanup should still be called despite model load failure
                    mock_cleanup.assert_called_once()


def test_editing_presence_keeps_models_warm_end_to_end():
    """Integration test: typing → model pre-load → cleanup respects editing flag."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.inference.manager import MODELS, cleanup_if_no_work
    from backend.database.redis_ops import redis_ops
    from unittest.mock import MagicMock

    try:
        # Step 1: Simulate user typing (set editing:active key)
        with redis_ops() as r:
            r.setex("editing:active", 120, "active")

        # Step 2: Verify models not loaded yet
        assert MODELS["llm"] is None, "Model should not be loaded initially"

        # Step 3: Run orchestrator (should detect editing and pre-load)
        with patch("backend.database.redis_ops.enqueue_pending_episodes"):
            with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
                with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                    # Mock the actual model loading to avoid loading real models in tests
                    mock_model = MagicMock()
                    with patch("backend.inference.manager.DspyLM", return_value=mock_model):
                        orchestrate_inference_work.call_local(reschedule=False)

                        # Step 4: Verify model is now loaded
                        assert MODELS["llm"] is not None, "Model should be loaded after orchestrator runs"

                        # Step 5: Verify cleanup doesn't unload while editing:active exists
                        cleanup_if_no_work()
                        assert MODELS["llm"] is not None, "Model should still be loaded while editing"

        # Step 6: Delete key and verify cleanup unloads
        with redis_ops() as r:
            r.delete("editing:active")

        with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                cleanup_if_no_work()
                assert MODELS["llm"] is None, "Model should be unloaded after editing stops"

    finally:
        # Clean up: ensure model cache is reset and Redis key is deleted
        MODELS["llm"] = None
        try:
            with redis_ops() as r:
                r.delete("editing:active")
        except Exception:
            pass
