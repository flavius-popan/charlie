"""Smoke test for OutlinesAdapter three-tier fallback."""

import pytest
from pydantic import BaseModel
import dspy
from dspy_outlines.adapter import OutlinesAdapter


class SimpleOutput(BaseModel):
    result: str


class SimpleSignature(dspy.Signature):
    """Simple test signature."""
    text: str = dspy.InputField()
    output: SimpleOutput = dspy.OutputField()


def test_adapter_initialization():
    """Test that OutlinesAdapter initializes with metrics."""
    adapter = OutlinesAdapter()

    assert hasattr(adapter, 'metrics')
    assert adapter.metrics['tier1_success'] == 0
    assert adapter.metrics['tier2_success'] == 0
    assert adapter.metrics['tier3_success'] == 0
    assert adapter.metrics['tier1_failures'] == 0
    assert adapter.metrics['tier2_failures'] == 0


def test_extract_constraint():
    """Test constraint extraction from signature."""
    adapter = OutlinesAdapter()
    constraint = adapter._extract_constraint(SimpleSignature)

    assert constraint == SimpleOutput


def test_has_tool_calls_false():
    """Test ToolCalls detection returns False for non-tool signatures."""
    adapter = OutlinesAdapter()
    assert adapter._has_tool_calls(SimpleSignature) is False


def test_parse_json_valid():
    """Test JSON parsing with valid input."""
    adapter = OutlinesAdapter()
    completion = '{"output": {"result": "test"}}'

    parsed = adapter._parse_json(SimpleSignature, completion)

    assert 'output' in parsed
    assert isinstance(parsed['output'], SimpleOutput)
    assert parsed['output'].result == "test"


def test_parse_json_with_extra_text():
    """Test JSON parsing extracts JSON from text."""
    adapter = OutlinesAdapter()
    completion = 'Here is the result: {"output": {"result": "test"}} and more text'

    parsed = adapter._parse_json(SimpleSignature, completion)

    assert 'output' in parsed
    assert isinstance(parsed['output'], SimpleOutput)
    assert parsed['output'].result == "test"


def test_parse_json_missing_field():
    """Test JSON parsing fails when field is missing."""
    from dspy.utils.exceptions import AdapterParseError

    adapter = OutlinesAdapter()
    completion = '{"wrong_field": {"result": "test"}}'

    with pytest.raises(AdapterParseError):
        adapter._parse_json(SimpleSignature, completion)


def test_user_message_output_requirements():
    """Test JSON instruction generation."""
    adapter = OutlinesAdapter()
    instruction = adapter._user_message_output_requirements(SimpleSignature)

    assert "JSON object" in instruction
    assert "`output`" in instruction
