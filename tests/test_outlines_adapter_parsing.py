"""Tests for OutlinesAdapter parsing and helper methods."""

import pytest
from pydantic import BaseModel
from typing import Any
import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy.clients.lm import LM


class SimpleOutput(BaseModel):
    result: str


class SimpleSignature(dspy.Signature):
    """Simple test signature."""

    text: str = dspy.InputField()
    output: SimpleOutput = dspy.OutputField()


class MockLM(LM):
    """Mock LM for testing adapter fallback logic.

    Returns strings matching BaseLM._process_completion output format.
    """

    def __init__(self, responses=None, exception=None):
        """
        Args:
            responses: List of response strings to return
            exception: Exception to raise when called
        """
        self.responses = responses or []
        self.exception = exception
        self.call_count = 0

    def __call__(self, messages, **kwargs):
        """Return mock response or raise exception."""
        self.call_count += 1

        if self.exception:
            raise self.exception

        # Return as list of strings (matching BaseLM._process_completion output)
        response = self.responses[0] if self.responses else '{"output": {"result": "mock"}}'
        return [response]


def test_adapter_initialization():
    """Test that OutlinesAdapter initializes with metrics."""
    adapter = OutlinesAdapter()

    assert hasattr(adapter, "metrics")
    assert adapter.metrics["chat_success"] == 0
    assert adapter.metrics["json_success"] == 0
    assert adapter.metrics["outlines_json_success"] == 0
    assert adapter.metrics["chat_failures"] == 0
    assert adapter.metrics["json_failures"] == 0


def test_extract_constraint():
    """Test constraint extraction from signature."""
    adapter = OutlinesAdapter()
    constraint, field_name = adapter._extract_constraint(SimpleSignature)

    assert constraint == SimpleOutput
    assert field_name == "output"


def test_has_tool_calls_false():
    """Test ToolCalls detection returns False for non-tool signatures."""
    adapter = OutlinesAdapter()
    assert adapter._has_tool_calls(SimpleSignature) is False


# NOTE: _parse_json and _user_message_output_requirements methods removed
# OutlinesAdapter now delegates JSON parsing to stock JSONAdapter for 1-to-1 parity


# Integration Tests (with mock LM)


def test_chat_success():
    """Test Chat adapter succeeds on first try."""
    from unittest.mock import patch

    adapter = OutlinesAdapter()
    mock_lm = MockLM()

    # Mock _call_postprocess to succeed (tier 1 processing)
    with patch.object(adapter, '_call_postprocess', return_value=[{"output": SimpleOutput(result="mock")}]):
        result = adapter(
            lm=mock_lm,
            lm_kwargs={},
            signature=SimpleSignature,
            demos=[],
            inputs={"text": "test input"}
        )

    assert len(result) == 1
    assert result[0]["output"].result == "mock"
    assert adapter.metrics["chat_success"] == 1
    assert adapter.metrics["chat_failures"] == 0
    assert adapter.metrics["json_success"] == 0


def test_chat_fails_json_succeeds():
    """Test Chat fails, JSON fallback succeeds."""
    from unittest.mock import patch
    from dspy.utils.exceptions import AdapterParseError

    adapter = OutlinesAdapter()
    mock_lm = MockLM(responses=['{"output": {"result": "json_success"}}'])
    mock_lm.model = "test-model"  # JSONAdapter needs lm.model to be a string

    # Mock tier 1 (ChatAdapter processing) to fail
    def chat_fail(*args, **kwargs):
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=SimpleSignature,
            lm_response="bad response",
            message="Parse failed"
        )

    # Mock tier 2 (JSONAdapter) to succeed
    from dspy.adapters.json_adapter import JSONAdapter
    with patch.object(adapter.__class__.__bases__[0], '__call__', side_effect=chat_fail):
        with patch.object(JSONAdapter, '__call__', return_value=[{"output": SimpleOutput(result="json_success")}]):
            result = adapter(
                lm=mock_lm,
                lm_kwargs={},
                signature=SimpleSignature,
                demos=[],
                inputs={"text": "test input"}
            )

    assert len(result) == 1
    assert result[0]["output"].result == "json_success"
    assert adapter.metrics["chat_failures"] == 1
    assert adapter.metrics["json_success"] == 1
    assert adapter.metrics["outlines_json_success"] == 0


def test_chat_json_fail_outlines_succeeds():
    """Test Chat and JSON fail, OutlinesJSON succeeds."""
    from unittest.mock import patch
    from dspy.utils.exceptions import AdapterParseError

    adapter = OutlinesAdapter()
    mock_lm = MockLM(responses=['{"output": {"result": "outlines_success"}}'])
    mock_lm.model = "test-model"  # JSONAdapter needs lm.model to be a string

    # Mock tier 1 (ChatAdapter processing) to fail
    def chat_fail(*args, **kwargs):
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=SimpleSignature,
            lm_response="bad response",
            message="Parse failed"
        )

    # Mock tier 2 (JSONAdapter) to fail
    def json_fail(*args, **kwargs):
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=SimpleSignature,
            lm_response="bad json",
            message="JSON parse failed"
        )

    from dspy.adapters.json_adapter import JSONAdapter
    with patch.object(adapter.__class__.__bases__[0], '__call__', side_effect=chat_fail):
        with patch.object(JSONAdapter, '__call__', side_effect=json_fail):
            result = adapter(
                lm=mock_lm,
                lm_kwargs={},
                signature=SimpleSignature,
                demos=[],
                inputs={"text": "test input"}
            )

    assert len(result) == 1
    assert result[0]["output"].result == "outlines_success"
    assert adapter.metrics["chat_failures"] == 1
    assert adapter.metrics["json_failures"] == 1
    assert adapter.metrics["outlines_json_success"] == 1


def test_tool_calls_skips_outlines_json():
    """Test ToolCalls prevents OutlinesJSON fallback."""
    from unittest.mock import patch
    from dspy.utils.exceptions import AdapterParseError
    from dspy.adapters.types.tool import ToolCalls

    # Create signature with ToolCalls
    class ToolSignature(dspy.Signature):
        """Signature with tool calls."""
        text: str = dspy.InputField()
        tools: ToolCalls = dspy.OutputField()

    adapter = OutlinesAdapter()
    mock_lm = MockLM()
    mock_lm.model = "test-model"  # JSONAdapter needs lm.model to be a string

    # Mock tier 1 (ChatAdapter processing) to fail
    def chat_fail(*args, **kwargs):
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=ToolSignature,
            lm_response="bad response",
            message="Parse failed"
        )

    # Mock tier 2 (JSONAdapter) to fail
    def json_fail(*args, **kwargs):
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=ToolSignature,
            lm_response="bad json",
            message="JSON parse failed"
        )

    from dspy.adapters.json_adapter import JSONAdapter
    with patch.object(adapter.__class__.__bases__[0], '__call__', side_effect=chat_fail):
        with patch.object(JSONAdapter, '__call__', side_effect=json_fail):
            with pytest.raises(AdapterParseError) as exc_info:
                adapter(
                    lm=mock_lm,
                    lm_kwargs={},
                    signature=ToolSignature,
                    demos=[],
                    inputs={"text": "test input"}
                )

    assert "OutlinesJSON is skipped for ToolCalls" in str(exc_info.value)
    assert adapter.metrics["chat_failures"] == 1
    assert adapter.metrics["json_failures"] == 1
    assert adapter.metrics["outlines_json_success"] == 0


def test_multiple_completions_warning(caplog):
    """Test warning logged when n>1 with OutlinesJSON."""
    import logging
    from unittest.mock import patch

    caplog.set_level(logging.WARNING)

    adapter = OutlinesAdapter()
    mock_lm = MockLM()

    # Mock ChatAdapter to succeed (won't reach OutlinesJSON, but warning should still log)
    with patch.object(adapter.__class__.__bases__[0], '__call__', return_value=[{"output": SimpleOutput(result="test")}]):
        adapter(
            lm=mock_lm,
            lm_kwargs={'n': 3},
            signature=SimpleSignature,
            demos=[],
            inputs={"text": "test input"}
        )

    # Check warning was logged
    assert any("Multiple completions" in record.message for record in caplog.records)
    assert any("OutlinesJSON will return single completion only" in record.message for record in caplog.records)


def test_context_window_error_propagates():
    """Test ContextWindowExceededError doesn't trigger fallback."""
    from unittest.mock import patch
    from litellm import ContextWindowExceededError

    adapter = OutlinesAdapter()
    mock_lm = MockLM()

    # Mock tier 1 to raise ContextWindowExceededError
    def chat_context_error(*args, **kwargs):
        raise ContextWindowExceededError(
            message="Context window exceeded",
            model="test-model",
            llm_provider="test"
        )

    with patch.object(adapter, '_call_postprocess', side_effect=chat_context_error):
        with pytest.raises(ContextWindowExceededError):
            adapter(
                lm=mock_lm,
                lm_kwargs={},
                signature=SimpleSignature,
                demos=[],
                inputs={"text": "test input"}
            )

    # Should increment failures but not try JSON fallback
    assert adapter.metrics["chat_failures"] == 1
    assert adapter.metrics["json_success"] == 0
    assert mock_lm.call_count == 1  # LM called once in Chat tier, JSON fallback should not be attempted
