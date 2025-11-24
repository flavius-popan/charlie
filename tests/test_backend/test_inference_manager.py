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


def test_orchestrator_loads_models_when_user_is_editing_and_enabled():
    """Orchestrator should pre-load models when editing flag is set and inference enabled."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes") as mock_enqueue:
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = True

            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                        orchestrate_inference_work.call_local(reschedule=False)

                        mock_enqueue.assert_called_once()
                        mock_redis.exists.assert_called_with("editing:active")
                        mock_get_model.assert_called_once_with("llm")
                        mock_cleanup.assert_called_once()


def test_orchestrator_skips_preload_when_inference_disabled():
    """Orchestrator should skip preloading when inference disabled."""
    from backend.services.tasks import orchestrate_inference_work

    with patch("backend.database.redis_ops.enqueue_pending_episodes"):
        with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
            mock_redis = mock_redis_ops.return_value.__enter__.return_value
            mock_redis.exists.return_value = True

            with patch("backend.database.redis_ops.get_inference_enabled", return_value=False):
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.inference.manager.cleanup_if_no_work"):
                        orchestrate_inference_work.call_local(reschedule=False)

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


def test_orchestrator_leaves_cached_model_untouched():
    """Orchestrator should not reload when model already cached."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.inference.manager import MODELS
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    MODELS["llm"] = mock_model

    try:
        with patch("backend.database.redis_ops.enqueue_pending_episodes"):
            with patch("backend.inference.manager.get_model") as mock_get_model:
                with patch("backend.inference.manager.cleanup_if_no_work"):
                    orchestrate_inference_work.call_local(reschedule=False)

                    mock_get_model.assert_not_called()
                    assert MODELS["llm"] is mock_model
    finally:
        MODELS["llm"] = None


def test_cleanup_respects_edit_idle_grace_before_unload(monkeypatch):
    """Models stay loaded for grace period after editing ends to avoid thrash."""
    from backend.inference import manager as mgr

    # Seed cached model
    mgr.MODELS["llm"] = object()
    monkeypatch.setattr(mgr, "_last_edit_seen", None)

    with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
        with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
            with patch("backend.database.redis_ops.redis_ops") as mock_redis_ops:
                mock_redis = mock_redis_ops.return_value.__enter__.return_value
                # First call: editing present; second call: editing absent
                mock_redis.exists.side_effect = [True, False]

                # First invocation marks last_edit_seen
                mgr.cleanup_if_no_work()

                with patch("backend.inference.manager.unload_all_models") as mock_unload:
                    with patch("backend.inference.manager.time.monotonic", return_value=(mgr._last_edit_seen or 0) + 5):
                        mgr.cleanup_if_no_work()  # within grace
                        mock_unload.assert_not_called()

                    with patch("backend.inference.manager.time.monotonic", return_value=(mgr._last_edit_seen or 0) + mgr.EDIT_IDLE_GRACE_SECONDS + 1):
                        mgr.cleanup_if_no_work()  # beyond grace
                        mock_unload.assert_called_once()


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
    """Integration test: editing flag prevents unload; no auto-preload."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.inference.manager import MODELS, cleanup_if_no_work
    from backend.database.redis_ops import redis_ops
    from unittest.mock import MagicMock

    try:
        # Step 1: Simulate user editing (set editing:active key)
        with redis_ops() as r:
            r.set("editing:active", "active")

        # Step 2: Manually set a loaded model (simulating prior work)
        MODELS["llm"] = MagicMock()

        # Step 3: Run orchestrator (should not preload/alter loaded model)
        with patch("backend.database.redis_ops.enqueue_pending_episodes"):
            with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
                with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                    orchestrate_inference_work.call_local(reschedule=False)
                    # Model remains loaded; no preload invoked
                    assert MODELS["llm"] is not None, "Model should remain loaded"

        # Step 4: Verify cleanup doesn't unload while editing:active exists
        with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                cleanup_if_no_work()
                assert MODELS["llm"] is not None, "Model should stay loaded while editing"

        # Step 5: Delete key and verify cleanup unloads
        with redis_ops() as r:
            r.delete("editing:active")

        with patch("backend.database.redis_ops.get_episodes_by_status", return_value=[]):
            with patch("backend.database.redis_ops.get_inference_enabled", return_value=True):
                    with patch("backend.inference.manager.time.monotonic", return_value=(manager._last_edit_seen or 0) + manager.EDIT_IDLE_GRACE_SECONDS + 1):
                        cleanup_if_no_work()
                        assert MODELS["llm"] is None, "Model should be unloaded after grace window expires"

    finally:
        # Clean up: ensure model cache is reset and Redis key is deleted
        MODELS["llm"] = None
        try:
            with redis_ops() as r:
                r.delete("editing:active")
        except Exception:
            pass
