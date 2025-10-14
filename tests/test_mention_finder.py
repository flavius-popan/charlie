"""
Tests for entity mention detection.

Tests the mention finder utility to ensure proper entity detection,
position tracking, and overlap handling.
"""

import pytest
from app.utils.mention_finder import (
    find_entity_mentions,
    generate_excerpt,
    EntityMention,
)
from app.models.graph import EntityNode
from datetime import datetime, timezone


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        EntityNode(
            uuid="uuid-1",
            name="John Smith",
            summary="A person named John Smith",
            created_at=datetime.now(timezone.utc),
        ),
        EntityNode(
            uuid="uuid-2",
            name="John",
            summary="Another person named John",
            created_at=datetime.now(timezone.utc),
        ),
        EntityNode(
            uuid="uuid-3",
            name="San Francisco",
            summary="A city in California",
            created_at=datetime.now(timezone.utc),
        ),
        EntityNode(
            uuid="uuid-4",
            name="Python",
            summary="A programming language",
            created_at=datetime.now(timezone.utc),
        ),
    ]


def test_find_single_mention(sample_entities):
    """Test finding a single entity mention."""
    content = "I met John Smith yesterday at the park."
    mentions = find_entity_mentions(content, [sample_entities[0]])

    assert len(mentions) == 1
    assert mentions[0].entity_name == "John Smith"
    assert mentions[0].start == 6
    assert mentions[0].end == 16
    assert mentions[0].original_text == "John Smith"


def test_find_multiple_mentions(sample_entities):
    """Test finding multiple mentions of the same entity."""
    content = "John Smith called. John Smith said hello. John Smith left."
    mentions = find_entity_mentions(content, [sample_entities[0]])

    assert len(mentions) == 3
    for mention in mentions:
        assert mention.entity_name == "John Smith"


def test_case_insensitive_matching(sample_entities):
    """Test that matching is case-insensitive."""
    content = "I saw john smith and JOHN SMITH and John SMITH."
    mentions = find_entity_mentions(content, [sample_entities[0]])

    assert len(mentions) == 3
    assert mentions[0].original_text == "john smith"
    assert mentions[1].original_text == "JOHN SMITH"
    assert mentions[2].original_text == "John SMITH"


def test_longest_match_first(sample_entities):
    """Test that longer entity names are matched before shorter ones."""
    content = "John Smith and John went to the store."
    # Include both "John Smith" and "John"
    mentions = find_entity_mentions(content, [sample_entities[0], sample_entities[1]])

    # Should find "John Smith" once and "John" once (not "John" inside "John Smith")
    assert len(mentions) == 2
    assert mentions[0].entity_name == "John Smith"
    assert mentions[0].start == 0
    assert mentions[1].entity_name == "John"
    assert mentions[1].start == 15


def test_no_overlapping_mentions(sample_entities):
    """Test that overlapping mentions are avoided."""
    content = "John Smith is here."
    # Try to match both "John Smith" and "John"
    mentions = find_entity_mentions(content, [sample_entities[0], sample_entities[1]])

    # Should only match "John Smith" once, not "John" separately
    assert len(mentions) == 1
    assert mentions[0].entity_name == "John Smith"


def test_multiple_entity_types(sample_entities):
    """Test finding mentions of different entities in same text."""
    content = "John Smith visited San Francisco to learn Python programming."
    mentions = find_entity_mentions(
        content, [sample_entities[0], sample_entities[2], sample_entities[3]]
    )

    assert len(mentions) == 3
    assert mentions[0].entity_name == "John Smith"
    assert mentions[1].entity_name == "San Francisco"
    assert mentions[2].entity_name == "Python"


def test_empty_content():
    """Test with empty content."""
    mentions = find_entity_mentions("", [])
    assert len(mentions) == 0


def test_no_entities():
    """Test with no entities."""
    content = "Some random text here."
    mentions = find_entity_mentions(content, [])
    assert len(mentions) == 0


def test_no_matches(sample_entities):
    """Test when no entity names are found in content."""
    content = "This text contains none of the entity names."
    mentions = find_entity_mentions(content, [sample_entities[0]])
    assert len(mentions) == 0


def test_mention_positions_sorted(sample_entities):
    """Test that mentions are returned in position order."""
    content = "Python is great. John Smith uses Python. San Francisco has Python."
    mentions = find_entity_mentions(
        content, [sample_entities[0], sample_entities[2], sample_entities[3]]
    )

    # Verify mentions are sorted by position
    for i in range(len(mentions) - 1):
        assert mentions[i].start < mentions[i + 1].start


def test_generate_excerpt_simple():
    """Test excerpt generation with simple case."""
    content = (
        "This is a long piece of text. John Smith is mentioned here. More text follows."
    )
    mention = EntityMention(
        entity_uuid="uuid-1",
        entity_name="John Smith",
        entity_summary="A person",
        start=30,
        end=40,
        original_text="John Smith",
    )

    excerpt = generate_excerpt(content, mention, context_chars=20)

    assert "John Smith" in excerpt
    assert len(excerpt) < len(content)


def test_generate_excerpt_at_start():
    """Test excerpt generation when mention is at start of content."""
    content = "John Smith is at the beginning of this text with more content after."
    mention = EntityMention(
        entity_uuid="uuid-1",
        entity_name="John Smith",
        entity_summary="A person",
        start=0,
        end=10,
        original_text="John Smith",
    )

    excerpt = generate_excerpt(content, mention, context_chars=20)

    assert excerpt.startswith("John Smith")
    assert "..." in excerpt  # Should have trailing ellipsis


def test_generate_excerpt_at_end():
    """Test excerpt generation when mention is at end of content."""
    content = "This is a long text that ends with John Smith"
    mention = EntityMention(
        entity_uuid="uuid-1",
        entity_name="John Smith",
        entity_summary="A person",
        start=36,
        end=46,
        original_text="John Smith",
    )

    excerpt = generate_excerpt(content, mention, context_chars=20)

    assert "John Smith" in excerpt
    assert excerpt.startswith("...")  # Should have leading ellipsis


def test_special_characters_in_entity_name():
    """Test entity names with special regex characters."""
    entity = EntityNode(
        uuid="uuid-special",
        name="C++",
        summary="A programming language",
        created_at=datetime.now(timezone.utc),
    )

    content = "I love programming in C++ and C++ is fast."
    mentions = find_entity_mentions(content, [entity])

    assert len(mentions) == 2
    assert mentions[0].entity_name == "C++"


def test_entity_mention_preserves_metadata():
    """Test that entity metadata is preserved in mentions."""
    entity = EntityNode(
        uuid="test-uuid",
        name="Test Entity",
        summary="Test summary for tooltip",
        created_at=datetime.now(timezone.utc),
    )

    content = "This mentions Test Entity in the text."
    mentions = find_entity_mentions(content, [entity])

    assert len(mentions) == 1
    assert mentions[0].entity_uuid == "test-uuid"
    assert mentions[0].entity_name == "Test Entity"
    assert mentions[0].entity_summary == "Test summary for tooltip"
