"""End-to-end tests for add_journal pipeline orchestrator."""

import pytest
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

from pipeline import add_journal, AddJournalResults
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
async def test_add_journal_basic(configure_dspy):
    """Test end-to-end journal processing through add_journal()."""
    journal_text = """Today I met with Dr. Sarah Chen at Stanford University
    to discuss the AI ethics project. We agreed to collaborate with Microsoft."""

    result = await add_journal(
        content=journal_text,
        group_id="test_user",
    )

    # Validate output type
    assert isinstance(result, AddJournalResults)

    # Validate episode created
    assert isinstance(result.episode, EpisodicNode)
    assert result.episode.content == journal_text
    assert result.episode.group_id == "test_user"
    assert result.episode.uuid is not None

    # Validate nodes extracted
    assert isinstance(result.nodes, list)
    assert all(isinstance(node, EntityNode) for node in result.nodes)
    if result.nodes:
        assert all(node.group_id == "test_user" for node in result.nodes)
        assert all(node.uuid is not None for node in result.nodes)

    # Validate UUID map
    assert isinstance(result.uuid_map, dict)
    assert all(
        isinstance(k, str) and isinstance(v, str) for k, v in result.uuid_map.items()
    )

    # Validate metadata structure
    assert isinstance(result.metadata, dict)
    required_keys = {
        "extracted_count",
        "resolved_count",
        "exact_matches",
        "fuzzy_matches",
        "new_entities",
    }
    assert required_keys.issubset(result.metadata.keys())
    assert all(isinstance(result.metadata[k], int) for k in required_keys)


@pytest.mark.asyncio
async def test_add_journal_with_default_group_id(configure_dspy):
    """Test add_journal() uses FalkorDB default group_id when not provided."""
    journal_text = "Met with Bob to discuss the project timeline."

    result = await add_journal(content=journal_text)

    # Should use FalkorDB default group_id '\\_'
    assert result.episode.group_id == "\\_"
    assert all(node.group_id == "\\_" for node in result.nodes)


@pytest.mark.asyncio
async def test_add_journal_validates_inputs(configure_dspy):
    """Test add_journal() validates inputs using graphiti-core validators."""
    from graphiti_core.errors import GroupIdValidationError

    journal_text = "Test content"

    # Invalid group_id should raise error
    with pytest.raises(GroupIdValidationError):
        await add_journal(content=journal_text, group_id="invalid@group!")
