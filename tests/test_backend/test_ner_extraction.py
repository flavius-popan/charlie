"""Tests for NER-based entity extraction."""

import pytest
from backend.ner import predict_entities, format_entities
from backend.graph.entities_edges import get_type_name_from_ner_label, NER_LABEL_MAP


class TestNERExtraction:
    """Unit tests for predict_entities()."""

    def test_extract_person(self):
        """Test PER entity extraction."""
        entities = predict_entities("I met Sarah at the park.")
        names = [e["text"] for e in entities if e["label"] == "PER"]
        assert "Sarah" in names

    def test_extract_location(self):
        """Test LOC entity extraction."""
        entities = predict_entities("I visited Central Park in New York.")
        names = [e["text"] for e in entities if e["label"] == "LOC"]
        assert any("Park" in n or "New York" in n for n in names)

    def test_extract_organization(self):
        """Test ORG entity extraction."""
        entities = predict_entities("I work at Microsoft.")
        names = [e["text"] for e in entities if e["label"] == "ORG"]
        assert any("Microsoft" in n for n in names)

    def test_multiple_entities(self):
        """Test extraction of multiple entity types."""
        text = "Tim Cook visited Paris to meet with UNESCO officials."
        entities = predict_entities(text)
        labels = {e["label"] for e in entities}
        assert len(labels) >= 2  # At least PER and LOC/ORG

    def test_long_text_chunking(self):
        """Test sliding window chunking for long text."""
        # Create text longer than 512 tokens
        text = " ".join(
            [
                f"Alice met Bob in Seattle. " * 50,
                "Later, Charlie joined them in Portland.",
            ]
        )
        entities = predict_entities(text)
        names = [e["text"] for e in entities]
        assert any("Alice" in n for n in names)
        assert any("Charlie" in n for n in names)

    def test_empty_text(self):
        """Test empty input."""
        entities = predict_entities("")
        assert entities == []

    def test_no_entities(self):
        """Test text with no named entities."""
        entities = predict_entities("The quick brown fox jumps.")
        # May have few false positives, but should be minimal
        assert len(entities) <= 2

    def test_preserves_original_casing(self):
        """Test that original text casing is preserved."""
        entities = predict_entities("I met JOHN at the STORE.")
        names = [e["text"] for e in entities]
        # Should preserve the original uppercase
        assert any("JOHN" in n for n in names)

    def test_entity_has_confidence(self):
        """Test that entities include confidence scores."""
        entities = predict_entities("Microsoft announced a new product.")
        for entity in entities:
            assert "confidence" in entity
            assert 0.0 <= entity["confidence"] <= 1.0


class TestNERLabelMapping:
    """Tests for NER label to entity type mapping."""

    def test_per_maps_to_person(self):
        assert get_type_name_from_ner_label("PER") == "Person"

    def test_loc_maps_to_location(self):
        assert get_type_name_from_ner_label("LOC") == "Location"

    def test_org_maps_to_organization(self):
        assert get_type_name_from_ner_label("ORG") == "Organization"

    def test_misc_maps_to_miscellaneous(self):
        assert get_type_name_from_ner_label("MISC") == "Miscellaneous"

    def test_unknown_defaults_to_miscellaneous(self):
        assert get_type_name_from_ner_label("UNKNOWN") == "Miscellaneous"

    def test_all_labels_have_mappings(self):
        """Ensure all NER labels are mapped."""
        expected_labels = {"PER", "LOC", "ORG", "MISC"}
        assert set(NER_LABEL_MAP.keys()) == expected_labels


class TestFormatEntities:
    """Tests for format_entities()."""

    def test_plain_format(self):
        """Test plain text output."""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99},
            {"text": "Seattle", "label": "LOC", "confidence": 0.95},
        ]
        result = format_entities(entities)
        assert "Microsoft" in result
        assert "Seattle" in result

    def test_with_labels(self):
        """Test output with labels."""
        entities = [{"text": "Microsoft", "label": "ORG", "confidence": 0.99}]
        result = format_entities(entities, include_labels=True)
        assert result == ["Microsoft [ORG]"]

    def test_with_confidence(self):
        """Test output with labels and confidence."""
        entities = [{"text": "Microsoft", "label": "ORG", "confidence": 0.99}]
        result = format_entities(
            entities, include_labels=True, include_confidence=True
        )
        assert result == ["Microsoft [ORG:99%]"]

    def test_deduplication(self):
        """Test case-insensitive deduplication."""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99},
            {"text": "microsoft", "label": "ORG", "confidence": 0.95},
        ]
        result = format_entities(entities, deduplicate=True)
        assert len(result) == 1
