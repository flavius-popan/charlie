"""Tests for the MLX runtime utilities and DSPy LM wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import dspy

from mlx_runtime import MLX_LOCK, MLXDspyLM, load_mlx_model


def test_load_mlx_model(model_path):
    """MLX models load successfully via the shared loader."""
    model, tokenizer = load_mlx_model(model_path)
    assert model is not None
    assert tokenizer is not None


def test_mlx_lock_is_global():
    """The exported lock should be a single shared threading.Lock."""
    import threading

    assert isinstance(MLX_LOCK, threading.Lock)
    assert not MLX_LOCK.locked()


def test_forward_formats_messages_and_calls_generate(model_path):
    """forward() should format chat messages and call mlx_lm.generate once."""

    lm = MLXDspyLM(model_path=model_path)
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi"},
    ]

    with patch("mlx_runtime.dspy_lm.mlx_lm.generate") as mock_generate:
        mock_generate.return_value = "Hi."
        result = lm.forward(messages=messages, max_tokens=16)

    mock_generate.assert_called_once()
    assert isinstance(result, SimpleNamespace)
    assert result.choices[0].message.content == "Hi."


def test_lock_held_during_generate(model_path):
    """MLX_LOCK must be held whenever mlx_lm.generate executes."""

    lm = MLXDspyLM(model_path=model_path)

    def fake_generate(*args, **kwargs):
        assert MLX_LOCK.locked(), "Generate should run under MLX_LOCK"
        return "mock response"

    with patch("mlx_runtime.dspy_lm.mlx_lm.generate", side_effect=fake_generate):
        result = lm.forward(prompt="Test prompt")

    assert result.choices[0].message.content == "mock response"


def test_integrates_with_dspy_predict(model_path):
    """Smoke-test that DSPy can run end-to-end with the MLX LM."""

    class Echo(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = MLXDspyLM(model_path=model_path, generation_config={"temp": 0.0})
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    predictor = dspy.Predict(Echo)
    output = predictor(prompt="Say 'OK'")
    assert hasattr(output, "answer")
    assert isinstance(output.answer, str)
