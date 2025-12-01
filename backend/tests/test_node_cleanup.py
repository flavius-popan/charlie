"""Tests for entity name cleanup module."""

import pytest

from backend.utils.node_cleanup import cleanup_entity_name, cleanup_extracted_entities


class TestArticleStripping:
    """Test 'the' stripping with proper noun preservation."""

    @pytest.mark.parametrize("input_name,expected", [
        # Lowercase "the" stripped
        ("the gym", "gym"),
        ("the park", "park"),
        ("the lake", "lake"),
        ("the apt", "apt"),

        # Capital "The" preserved (proper names)
        ("The Beatles", "The Beatles"),
        ("The Expanse", "The Expanse"),
        ("The New York Times", "The New York Times"),
    ])
    def test_article_stripping(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestDiscardPatterns:
    """Test entities that should be discarded entirely."""

    @pytest.mark.parametrize("input_name", [
        # Generic rooms
        "living room",
        "bedroom",
        "bathroom",
        "dining room",
        "laundry room",
        "room",

        # Single-word areas
        "balcony",
        "patio",
        "porch",
        "deck",
        "basement",
        "garage",
        "attic",
    ])
    def test_discard_patterns(self, input_name):
        assert cleanup_entity_name(input_name) is None


class TestDiscardAfterCleanup:
    """Test entities discarded after 'the' is stripped."""

    @pytest.mark.parametrize("input_name", [
        "the living room",
        "the balcony",
        "the patio",
    ])
    def test_discard_after_cleanup(self, input_name):
        assert cleanup_entity_name(input_name) is None


class TestPassthrough:
    """Test entities that pass through unchanged."""

    @pytest.mark.parametrize("input_name,expected", [
        # Normal entities - no changes
        ("yoga", "yoga"),
        ("meditation", "meditation"),
        ("Prospect Park", "Prospect Park"),
        ("Kenji", "Kenji"),
        ("New York", "New York"),

        # Verb phrases pass through (left to DSPy)
        ("morning walk", "morning walk"),
        ("walking around the lake", "walking around the lake"),
        ("watching The Expanse", "watching The Expanse"),
    ])
    def test_passthrough(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.parametrize("input_name,expected", [
        ("a", None),  # too short
        ("", None),   # empty
        ("AB", "AB"), # minimum length
    ])
    def test_edge_cases(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestCleanupExtractedEntities:
    """Test batch cleanup with deduplication."""

    def test_deduplication_case_insensitive(self):
        """First occurrence wins, case-insensitive matching."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="the park", entity_type_id=2),
            ExtractedEntity(name="Park", entity_type_id=2),
            ExtractedEntity(name="PARK", entity_type_id=2),
        ]

        result = cleanup_extracted_entities(entities)

        assert len(result) == 1
        assert result[0].name == "park"

    def test_filters_discarded_entities(self):
        """Entities matching discard patterns are removed."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="living room", entity_type_id=2),
            ExtractedEntity(name="Prospect Park", entity_type_id=2),
            ExtractedEntity(name="balcony", entity_type_id=2),
        ]

        result = cleanup_extracted_entities(entities)

        assert len(result) == 1
        assert result[0].name == "Prospect Park"

    def test_preserves_entity_type(self):
        """Entity type is preserved through cleanup."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="the gym", entity_type_id=2),
            ExtractedEntity(name="Kenji", entity_type_id=1),
            ExtractedEntity(name="yoga", entity_type_id=4),
        ]

        result = cleanup_extracted_entities(entities)

        assert len(result) == 3
        gym = next(e for e in result if e.name == "gym")
        kenji = next(e for e in result if e.name == "Kenji")
        yoga = next(e for e in result if e.name == "yoga")

        assert gym.entity_type_id == 2
        assert kenji.entity_type_id == 1
        assert yoga.entity_type_id == 4
