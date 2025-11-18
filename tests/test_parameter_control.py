from pipeline import _dspy_setup  # noqa: F401
import dspy

from mlx_runtime import MLXDspyLM


def test_default_generation_config(model_path):
    """Verify MLXDspyLM initializes with default config."""
    lm = MLXDspyLM(model_path=model_path)
    assert hasattr(lm, 'generation_config')
    assert 'temp' in lm.generation_config
    assert 'top_p' in lm.generation_config
    assert lm.generation_config['temp'] == 0.0  # Default should be deterministic


def test_custom_generation_config(model_path):
    """Verify custom config is stored and accessible."""
    custom_config = {
        "temp": 0.5,
        "top_p": 0.8,
        "min_p": 0.05,
    }
    lm = MLXDspyLM(model_path=model_path, generation_config=custom_config)
    assert lm.generation_config == custom_config


def test_deterministic_generation_with_zero_temp(model_path):
    """Verify temp=0.0 produces consistent results."""
    config = {"temp": 0.0}  # Greedy decoding
    lm = MLXDspyLM(model_path=model_path, generation_config=config)

    # Configure DSPy
    dspy.settings.configure(lm=lm, adapter=dspy.ChatAdapter())

    # Simple unconstrained generation with open-ended prompt
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(SimpleSignature)

    # Run multiple times - with temp=0.0, should get identical results
    results = [predictor(question="What's up?").answer for _ in range(3)]

    # All results should be identical with greedy decoding
    assert results[0] == results[1] == results[2], \
        f"Expected identical results with temp=0.0, got: {results}"


def test_high_temp_produces_variance(model_path):
    """Verify high temp produces different outputs for same prompt."""
    config = {"temp": 1.5}  # Very high temp for variance
    lm = MLXDspyLM(model_path=model_path, generation_config=config)

    # Configure DSPy
    dspy.settings.configure(lm=lm, adapter=dspy.ChatAdapter())

    # Simple unconstrained generation with open-ended prompt
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(SimpleSignature)

    # Run multiple times - with high temp, should get different results
    results = [predictor(question="What's up?").answer for _ in range(5)]

    # At least some results should differ with high temp
    unique_results = set(results)
    assert len(unique_results) > 1, \
        f"Expected variance with temp=1.5, but got identical results: {results}"


def test_temp_comparison(model_path):
    """Compare outputs from low vs high temperature."""
    # Test with deterministic temp
    lm_low = MLXDspyLM(model_path=model_path, generation_config={"temp": 0.0})
    dspy.settings.configure(lm=lm_low, adapter=dspy.ChatAdapter())

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor_low = dspy.Predict(SimpleSignature)
    result_low = predictor_low(question="What's up?").answer

    # Test with high temp
    lm_high = MLXDspyLM(model_path=model_path, generation_config={"temp": 1.5})
    dspy.settings.configure(lm=lm_high, adapter=dspy.ChatAdapter())

    predictor_high = dspy.Predict(SimpleSignature)
    result_high = predictor_high(question="What's up?").answer

    # Verify both produce output
    assert result_low
    assert result_high

    # They may differ (not guaranteed, but with high temp likely)
    # This test mainly verifies no errors occur with different configs
