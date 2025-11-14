"""Test direct access to lm.model for all Outlines output types.

This test file demonstrates direct Outlines model usage without DSPy overhead,
covering all output type categories from the Outlines documentation.
"""

from typing import Literal, Dict
from enum import Enum
from pydantic import BaseModel
import pytest
from dspy_outlines import OutlinesLM

# TODO: TEST THIS ON BARE MODEL WITHOUT DSPy ADAPTER!


# Fixtures and helper classes


@pytest.fixture(scope="module")
def lm(request):
    """Shared OutlinesLM instance for all tests to avoid loading model 15+ times."""
    model_path = request.config.getoption("--model")
    return OutlinesLM(model_path=model_path)


class Sentiment(str, Enum):
    """Enum for multiple choice testing."""

    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Person(BaseModel):
    """Pydantic model for JSON schema testing."""

    name: str
    age: int


# Basic attribute tests


def test_model_attribute_exists(lm):
    """Test that OutlinesLM exposes .outlines_model attribute."""
    assert hasattr(lm, "model")
    assert lm.model is not None
    assert isinstance(lm.model, str), (
        "lm.model should be model path string for DSPy compatibility"
    )
    assert hasattr(lm, "outlines_model")
    assert lm.outlines_model is not None
    assert callable(lm.outlines_model)


def test_model_is_outlines_mlxlm(lm):
    """Test that .outlines_model is the Outlines MLXLM instance."""
    # Check it's an Outlines model (has __call__ with output_type parameter)
    import inspect

    sig = inspect.signature(lm.outlines_model.__call__)
    params = list(sig.parameters.keys())

    # Outlines models accept: model_input, output_type, backend, **inference_kwargs
    assert "output_type" in params or "kwargs" in params


def test_model_attribute_same_across_calls(lm):
    """Test that .model returns the same instance across calls."""
    model1 = lm.model
    model2 = lm.model

    assert model1 is model2


# Output Type Category 1: Basic Python Types


def test_basic_type_int(lm):
    """Test generation constrained to int type.

    Note: Constraints enforce format (valid integer), not semantic correctness.
    The model may not answer the question correctly, but the output will be a valid int.
    """
    prompt = "How many days are in a week? Answer with just the number:"
    result = lm.outlines_model(prompt, int, max_tokens=5)

    assert isinstance(result, str)
    # Constraint guarantees valid integer format
    assert result.strip().isdigit()
    # Can successfully parse as int
    parsed = int(result.strip())
    assert parsed > 0  # Reasonable sanity check
    # Note: We don't assert parsed == 7 because constraints control syntax, not semantics


def test_basic_type_float(lm):
    """Test generation constrained to float type.

    Note: Constraints enforce format (valid float), not semantic correctness.
    The model may not compute correctly, but the output will be a valid float.
    """
    prompt = "What is half of one? Answer with a decimal number:"
    result = lm.outlines_model(prompt, float, max_tokens=10)

    assert isinstance(result, str)
    # Constraint guarantees valid float format - can successfully parse
    parsed = float(result.strip())
    assert isinstance(parsed, float)
    # Note: We don't check the value because constraints control syntax, not semantics


def test_basic_type_bool(lm):
    """Test generation constrained to bool type."""
    prompt = "Is the sky blue during daytime? Answer True or False:"
    result = lm.outlines_model(prompt, bool, max_tokens=10)

    assert isinstance(result, str)
    assert result.strip() in ["True", "False"]


# Output Type Category 2: Multiple Choices


def test_multiple_choice_literal(lm):
    """Test multiple choice using Literal type."""
    prompt = "What color is the sky during daytime? Choose one:"
    result = lm.outlines_model(prompt, Literal["blue", "red", "green"], max_tokens=10)

    assert result.strip() in ["blue", "red", "green"]
    assert result.strip() == "blue"


def test_multiple_choice_enum(lm):
    """Test multiple choice using Enum type."""
    prompt = "Is this review positive or negative: 'This product is amazing!' Answer:"
    result = lm.outlines_model(prompt, Sentiment, max_tokens=10)

    assert result.strip() in ["positive", "negative", "neutral"]
    assert result.strip() == "positive"


def test_multiple_choice_choice(lm):
    """Test multiple choice using Choice type (dynamic list)."""
    from outlines.types import Choice

    # Dynamic choice list
    cities = ["Paris", "London", "Tokyo"]
    prompt = "Name a world capital city:"
    result = lm.outlines_model(prompt, Choice(cities), max_tokens=10)

    assert result.strip() in cities


# Output Type Category 3: JSON Schemas


def test_json_schema_pydantic(lm):
    """Test JSON schema generation using Pydantic model."""
    prompt = "Create a person named Alice who is 30 years old. Respond in JSON:"
    result = lm.outlines_model(prompt, Person, max_tokens=50)

    assert isinstance(result, str)
    # Should be valid JSON matching Person schema
    person = Person.model_validate_json(result)
    assert person.name == "Alice"
    assert person.age == 30


def test_json_schema_dict(lm):
    """Test JSON schema using dict type (basic structure)."""
    prompt = (
        "Create a simple key-value mapping with 'name' and 'value'. Respond in JSON:"
    )
    result = lm.outlines_model(prompt, Dict[str, str], max_tokens=50)

    assert isinstance(result, str)
    # Should be parseable as JSON dict
    import json

    parsed = json.loads(result)
    assert isinstance(parsed, dict)


# Output Type Category 4: Regex Patterns


def test_regex_pattern_custom(lm):
    """Test regex pattern for structured format (e.g., phone number)."""
    from outlines.types import Regex

    # Phone number format: XXX-XXXX
    pattern = r"\d{3}-\d{4}"
    prompt = "Generate a phone number in format XXX-XXXX:"
    result = lm.outlines_model(prompt, Regex(pattern), max_tokens=20)

    assert isinstance(result, str)
    # Verify it matches the pattern
    import re

    assert re.fullmatch(pattern, result.strip())


def test_regex_pattern_builtin(lm):
    """Test using built-in regex patterns from outlines.types."""
    from outlines.types import sentence

    prompt = "Write a short sentence about the weather:"
    result = lm.outlines_model(prompt, sentence, max_tokens=30)

    assert isinstance(result, str)
    # Built-in sentence pattern: starts with capital, ends with punctuation
    assert result[0].isupper()
    assert result.strip()[-1] in ".!?"


# Unconstrained generation (no output type)


def test_unconstrained_generation(lm):
    """Test direct model call without constraint (unconstrained generation)."""
    prompt = "Say hello in one word:"
    result = lm.outlines_model(prompt, max_tokens=10)

    assert isinstance(result, str)
    assert len(result) > 0


# Integration test


def test_direct_call_bypasses_dspy(lm):
    """Test that direct model calls don't require DSPy configuration."""
    # No dspy.configure() needed - direct Outlines usage
    prompt = "Choose A, B, or C:"
    result = lm.outlines_model(prompt, Literal["A", "B", "C"], max_tokens=5)

    assert result.strip() in ["A", "B", "C"]
