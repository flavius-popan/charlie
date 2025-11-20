"""Tests for the backend inference utilities and DSPy LM wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import backend.dspy_cache  # noqa: F401  # enforce DSPy cache location
import dspy

from backend.inference import DspyLM, load_model


@pytest.mark.inference
def test_load_model():
    """llama.cpp models load successfully via HF auto-download."""
    model = load_model()
    assert model is not None


@pytest.mark.inference
def test_load_model_with_repo_id():
    """load_model accepts repo_id parameter."""
    model = load_model(repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF")
    assert model is not None


def test_load_model_uses_settings():
    """load_model uses MODEL_REPO_ID from settings when repo_id not provided."""
    from backend.settings import MODEL_REPO_ID
    from unittest.mock import patch, MagicMock

    with patch("backend.inference.loader.Llama") as mock_llama:
        mock_instance = MagicMock()
        mock_llama.from_pretrained.return_value = mock_instance

        load_model()

        mock_llama.from_pretrained.assert_called_once()
        call_kwargs = mock_llama.from_pretrained.call_args[1]
        assert call_kwargs["repo_id"] == MODEL_REPO_ID


def test_load_model_builds_filename_pattern():
    """load_model builds filename pattern from MODEL_QUANTIZATION."""
    from backend.settings import MODEL_QUANTIZATION
    from unittest.mock import patch, MagicMock

    with patch("backend.inference.loader.Llama") as mock_llama:
        mock_instance = MagicMock()
        mock_llama.from_pretrained.return_value = mock_instance

        load_model()

        call_kwargs = mock_llama.from_pretrained.call_args[1]
        assert call_kwargs["filename"] == f"*{MODEL_QUANTIZATION}.gguf"


def test_dspy_lm_accepts_repo_id():
    """DspyLM accepts repo_id parameter."""
    from unittest.mock import patch, MagicMock

    with patch("backend.inference.dspy_lm.load_model") as mock_load:
        mock_load.return_value = MagicMock()

        lm = DspyLM(repo_id="custom/repo")

        mock_load.assert_called_once_with("custom/repo")


def test_dspy_lm_uses_default_repo():
    """DspyLM uses default repo when repo_id not provided."""
    from unittest.mock import patch, MagicMock

    with patch("backend.inference.dspy_lm.load_model") as mock_load:
        mock_load.return_value = MagicMock()

        lm = DspyLM()

        mock_load.assert_called_once_with(None)


@pytest.mark.inference
def test_forward_returns_proper_structure():
    """forward() returns OpenAI-compatible response structure."""
    lm = DspyLM()
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi"},
    ]

    result = lm.forward(messages=messages, max_tokens=16)

    assert isinstance(result, SimpleNamespace)
    assert hasattr(result, "choices")
    assert len(result.choices) > 0
    assert hasattr(result.choices[0], "message")
    assert hasattr(result.choices[0].message, "content")
    assert isinstance(result.choices[0].message.content, str)


@pytest.mark.inference
def test_integrates_with_dspy_predict():
    """End-to-end test that DSPy can run real inference with llama.cpp LM."""
    class Echo(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = DspyLM(generation_config={"temp": 0.0})
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    predictor = dspy.Predict(Echo)
    output = predictor(prompt="Say 'OK'")

    assert hasattr(output, "answer")
    assert isinstance(output.answer, str)
    assert len(output.answer) > 0
