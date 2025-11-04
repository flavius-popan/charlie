"""Tests for extract_nodes pipeline module."""

import pytest
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

from pipeline import ExtractNodes, ExtractNodesOutput
from graphiti_core.nodes import EntityNode, EpisodicNode
from settings import MODEL_CONFIG


@pytest.fixture(scope="module")
def configure_dspy():
    """Configure DSPy once for all tests."""
    dspy.settings.configure(
        adapter=OutlinesAdapter(),
        lm=OutlinesLM(generation_config=MODEL_CONFIG),
    )


@pytest.mark.asyncio
async def test_extract_nodes_from_journal_text(configure_dspy):
    """Test extracting entities from journal text."""
    extractor = ExtractNodes(group_id="test_user")

    journal_text = """Today I met with Dr. Sarah Chen at Stanford University
    to discuss the AI ethics project. We agreed to collaborate with Microsoft."""

    result = await extractor(content=journal_text)

    # Validate output type
    assert isinstance(result, ExtractNodesOutput)

    # Validate episode created
    assert isinstance(result.episode, EpisodicNode)
    assert result.episode.content == journal_text
    assert result.episode.group_id == "test_user"
    assert result.episode.uuid is not None

    # Validate nodes extracted (don't check exact count or names)
    assert isinstance(result.nodes, list)
    assert all(isinstance(node, EntityNode) for node in result.nodes)
    if result.nodes:  # If any nodes extracted
        assert all(node.group_id == "test_user" for node in result.nodes)
        assert all(node.uuid is not None for node in result.nodes)

    # Validate UUID map
    assert isinstance(result.uuid_map, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.uuid_map.items())

    # Validate duplicate pairs
    assert isinstance(result.duplicate_pairs, list)

    # Validate metadata structure
    assert isinstance(result.metadata, dict)
    required_keys = {"extracted_count", "resolved_count", "exact_matches", "fuzzy_matches", "new_entities"}
    assert required_keys.issubset(result.metadata.keys())
    assert all(isinstance(result.metadata[k], int) for k in required_keys)


@pytest.mark.asyncio
async def test_extract_nodes_empty_text(configure_dspy):
    """Test handling of empty journal text."""
    extractor = ExtractNodes(group_id="test_user")

    result = await extractor(content="")

    # Should still create episode
    assert isinstance(result.episode, EpisodicNode)
    assert result.episode.content == ""

    # May or may not extract entities (model-dependent)
    assert isinstance(result.nodes, list)
    assert isinstance(result.uuid_map, dict)
