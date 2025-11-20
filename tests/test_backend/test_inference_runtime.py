"""Tests for the inference runtime utilities and DSPy LM wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pipeline import _dspy_setup  # noqa: F401
import dspy

from inference_runtime import DspyLM, load_model


@pytest.mark.inference
def test_load_model():
    """llama.cpp models load successfully via HF auto-download."""
    model = load_model()
    assert model is not None


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
