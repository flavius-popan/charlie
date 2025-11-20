import pytest
import backend.dspy_cache  # noqa: F401  # enforce DSPy cache location
import dspy

from inference_runtime import DspyLM
from settings import MODEL_CONFIG


@pytest.mark.inference
def test_default_generation_config():
    """Verify DspyLM initializes with default config from settings."""
    lm = DspyLM()
    assert hasattr(lm, 'generation_config')
    assert lm.generation_config == MODEL_CONFIG
    assert lm.generation_config['temp'] == 0.7  # Qwen3 recommended default
    assert lm.generation_config['top_p'] == 0.8
    assert lm.generation_config['top_k'] == 20
    assert lm.generation_config['min_p'] == 0.0
    assert lm.generation_config['presence_penalty'] == 0.0


@pytest.mark.inference
def test_custom_generation_config():
    """Verify custom config overrides defaults."""
    custom_config = {
        "temp": 0.5,
        "top_p": 0.8,
        "min_p": 0.05,
    }
    lm = DspyLM(generation_config=custom_config)
    assert lm.generation_config == custom_config
    assert lm.generation_config['temp'] == 0.5

