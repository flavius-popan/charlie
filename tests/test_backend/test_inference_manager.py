"""Tests for the backend inference model manager (warm sessions)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from backend.inference import manager
from backend.inference.manager import get_model, unload_all_models, cleanup_if_no_work


def test_get_model_cold_start(reset_model_manager):
    """First call to get_model loads model (cold start)."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        mock_model = Mock()
        mock_dspy_lm.return_value = mock_model

        result = get_model("llm")

        assert result == mock_model
        mock_dspy_lm.assert_called_once()
        assert manager.MODELS["llm"] == mock_model


def test_get_model_warm_session(reset_model_manager):
    """Second call to get_model returns cached model (warm session)."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        mock_model = Mock()
        mock_dspy_lm.return_value = mock_model

        result1 = get_model("llm")
        result2 = get_model("llm")

        assert result1 == mock_model
        assert result2 == mock_model
        assert result1 is result2
        mock_dspy_lm.assert_called_once()


def test_get_model_invalid_type(reset_model_manager):
    """get_model raises ValueError for invalid model_type."""
    with pytest.raises(ValueError, match="Invalid model_type: invalid"):
        get_model("invalid")


def test_unload_all_models(reset_model_manager):
    """unload_all_models sets all models to None and triggers gc."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        with patch("backend.inference.manager.gc.collect") as mock_gc:
            mock_model = Mock()
            mock_dspy_lm.return_value = mock_model

            get_model("llm")
            assert manager.MODELS["llm"] is not None

            unload_all_models()

            assert manager.MODELS["llm"] is None
            mock_gc.assert_called_once()


def test_cleanup_if_no_work_with_pending_nodes(reset_model_manager):
    """cleanup_if_no_work keeps models loaded when pending_nodes exist."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
            mock_model = Mock()
            mock_dspy_lm.return_value = mock_model
            mock_get_episodes.side_effect = lambda status: (
                ["episode1", "episode2"] if status == "pending_nodes" else []
            )

            get_model("llm")
            assert manager.MODELS["llm"] is not None

            cleanup_if_no_work()

            assert manager.MODELS["llm"] is not None


def test_cleanup_if_no_work_with_pending_edges(reset_model_manager):
    """cleanup_if_no_work keeps models loaded when pending_edges exist."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
            mock_model = Mock()
            mock_dspy_lm.return_value = mock_model
            mock_get_episodes.side_effect = lambda status: (
                ["episode1"] if status == "pending_edges" else []
            )

            get_model("llm")
            assert manager.MODELS["llm"] is not None

            cleanup_if_no_work()

            assert manager.MODELS["llm"] is not None


def test_cleanup_if_no_work_unloads_when_queue_empty(reset_model_manager):
    """cleanup_if_no_work unloads models when both queues are empty."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
            with patch("backend.inference.manager.gc.collect") as mock_gc:
                mock_model = Mock()
                mock_dspy_lm.return_value = mock_model
                mock_get_episodes.return_value = []

                get_model("llm")
                assert manager.MODELS["llm"] is not None

                cleanup_if_no_work()

                assert manager.MODELS["llm"] is None
                mock_gc.assert_called_once()


def test_cleanup_if_no_work_checks_both_queues(reset_model_manager):
    """cleanup_if_no_work checks both pending_nodes and pending_edges queues."""
    with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
        mock_get_episodes.return_value = []

        cleanup_if_no_work()

        assert mock_get_episodes.call_count == 2
        mock_get_episodes.assert_any_call("pending_nodes")
        mock_get_episodes.assert_any_call("pending_edges")


def test_multiple_cold_warm_cycles(reset_model_manager):
    """Models can be loaded, unloaded, and reloaded (multiple cycles)."""
    with patch("backend.inference.manager.DspyLM") as mock_dspy_lm:
        with patch("backend.inference.manager.gc.collect"):
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_dspy_lm.side_effect = [mock_model1, mock_model2]

            result1 = get_model("llm")
            assert result1 == mock_model1
            unload_all_models()
            assert manager.MODELS["llm"] is None

            result2 = get_model("llm")
            assert result2 == mock_model2
            assert result2 is not result1

            assert mock_dspy_lm.call_count == 2


@pytest.mark.inference
def test_model_cold_start_loads_real_model(reset_model_manager):
    """First call loads model (cold start)."""
    model = get_model("llm")

    assert model is not None
    assert manager.MODELS["llm"] is model
    assert hasattr(model, "llm")


@pytest.mark.inference
def test_model_warm_session_returns_cached(reset_model_manager):
    """Second call returns cached model (warm session)."""
    model1 = get_model("llm")
    model2 = get_model("llm")

    assert model1 is model2
    assert manager.MODELS["llm"] is model1


@pytest.mark.inference
def test_unload_frees_model_from_memory(reset_model_manager):
    """unload_all_models frees model from memory."""
    model = get_model("llm")
    assert manager.MODELS["llm"] is not None

    unload_all_models()

    assert manager.MODELS["llm"] is None


@pytest.mark.inference
def test_cleanup_unloads_when_queue_empty(reset_model_manager):
    """cleanup_if_no_work unloads when queue empty."""
    with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
        mock_get_episodes.return_value = []

        model = get_model("llm")
        assert manager.MODELS["llm"] is not None

        cleanup_if_no_work()

        assert manager.MODELS["llm"] is None


@pytest.mark.inference
def test_cleanup_keeps_model_when_work_pending(reset_model_manager):
    """cleanup_if_no_work keeps model when work pending."""
    with patch("backend.database.redis_ops.get_episodes_by_status") as mock_get_episodes:
        mock_get_episodes.side_effect = lambda status: (
            ["episode1"] if status == "pending_nodes" else []
        )

        model = get_model("llm")
        assert manager.MODELS["llm"] is not None

        cleanup_if_no_work()

        assert manager.MODELS["llm"] is not None
        assert manager.MODELS["llm"] is model


@pytest.mark.inference
def test_multiple_load_unload_cycles(reset_model_manager):
    """Models can be loaded, unloaded, and reloaded."""
    model1 = get_model("llm")
    assert manager.MODELS["llm"] is model1

    unload_all_models()
    assert manager.MODELS["llm"] is None

    model2 = get_model("llm")
    assert manager.MODELS["llm"] is model2
    assert model2 is not model1


@pytest.mark.inference
def test_loaded_model_can_generate_text(reset_model_manager):
    """Loaded model can actually generate text."""
    model = get_model("llm")

    result = model.forward(
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Say hi"},
        ],
        max_tokens=16
    )

    assert hasattr(result, "choices")
    assert len(result.choices) > 0
    assert hasattr(result.choices[0], "message")
    assert isinstance(result.choices[0].message.content, str)
    assert len(result.choices[0].message.content) > 0
