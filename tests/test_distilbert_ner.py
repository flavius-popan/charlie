"""
Tests for distilbert_ner module

Structure:
- Unit tests for EntityExtractor (mocked data)
- Unit tests for format_entities helper
- Integration test for end-to-end inference (uses actual model)
"""

import sys
from pathlib import Path

# Add parent directory to path to import distilbert_ner module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import distilbert_ner


# ============================================================================
# Unit Tests: EntityExtractor
# ============================================================================


class TestEntityExtractor:
    """Unit tests for EntityExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance"""
        return distilbert_ner.EntityExtractor()

    def test_group_tokens_simple_words(self, extractor):
        """Test grouping simple tokens without subwords"""
        tokens = ["[CLS]", "Apple", "Inc", "[SEP]"]
        labels = ["O", "B-ORG", "I-ORG", "O"]
        attention_mask = np.array([1, 1, 1, 1])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        assert len(words) == 2
        assert words[0] == {
            "tokens": ["Apple"],
            "labels": ["B-ORG"],
            "start_token": 1,
            "end_token": 1,
        }
        assert words[1] == {
            "tokens": ["Inc"],
            "labels": ["I-ORG"],
            "start_token": 2,
            "end_token": 2,
        }

    def test_group_tokens_with_subwords(self, extractor):
        """Test grouping tokens with subword markers (##)"""
        tokens = ["[CLS]", "Micro", "##soft", "CEO", "[SEP]"]
        labels = ["O", "B-ORG", "I-ORG", "O", "O"]
        attention_mask = np.array([1, 1, 1, 1, 1])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        assert len(words) == 2
        assert words[0] == {
            "tokens": ["Micro", "##soft"],
            "labels": ["B-ORG", "I-ORG"],
            "start_token": 1,
            "end_token": 2,
        }
        assert words[1] == {
            "tokens": ["CEO"],
            "labels": ["O"],
            "start_token": 3,
            "end_token": 3,
        }

    def test_group_tokens_skips_padding(self, extractor):
        """Test that padding tokens are ignored"""
        tokens = ["[CLS]", "Apple", "[PAD]", "[PAD]"]
        labels = ["O", "B-ORG", "O", "O"]
        attention_mask = np.array([1, 1, 0, 0])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        assert len(words) == 1
        assert words[0]["tokens"] == ["Apple"]

    def test_aggregate_simple_entity(self, extractor):
        """Test aggregating a simple single-word entity"""
        words = [
            {"tokens": ["Apple"], "labels": ["B-ORG"], "start_token": 1, "end_token": 1}
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 1
        assert entities[0] == {
            "label": "ORG",
            "start_token": 1,
            "end_token": 1,
            "tokens": ["Apple"],
        }

    def test_aggregate_multi_word_entity(self, extractor):
        """Test aggregating multi-word entity with I- continuation"""
        words = [
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 1,
                "end_token": 1,
            },
            {"tokens": ["Inc"], "labels": ["I-ORG"], "start_token": 2, "end_token": 2},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 1
        assert entities[0] == {
            "label": "ORG",
            "start_token": 1,
            "end_token": 2,
            "tokens": ["Apple", "Inc"],
        }

    def test_aggregate_separate_entities(self, extractor):
        """Test that different entity types are not merged"""
        words = [
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 1,
                "end_token": 1,
            },
            {"tokens": ["CEO"], "labels": ["O"], "start_token": 2, "end_token": 2},
            {"tokens": ["Tim"], "labels": ["B-PER"], "start_token": 3, "end_token": 3},
            {"tokens": ["Cook"], "labels": ["I-PER"], "start_token": 4, "end_token": 4},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 2
        assert entities[0]["label"] == "ORG"
        assert entities[0]["tokens"] == ["Apple"]
        assert entities[1]["label"] == "PER"
        assert entities[1]["tokens"] == ["Tim", "Cook"]

    def test_aggregate_handles_outside_tags(self, extractor):
        """Test that O (outside) tags properly separate entities"""
        words = [
            {"tokens": ["in"], "labels": ["O"], "start_token": 1, "end_token": 1},
            {
                "tokens": ["Seattle"],
                "labels": ["B-LOC"],
                "start_token": 2,
                "end_token": 2,
            },
            {"tokens": ["today"], "labels": ["O"], "start_token": 3, "end_token": 3},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 1
        assert entities[0]["label"] == "LOC"
        assert entities[0]["tokens"] == ["Seattle"]

    def test_extract_end_to_end(self, extractor):
        """Test the full extract() method with mock tokenizer"""
        tokens = ["[CLS]", "Apple", "Inc", "in", "Seattle", "[SEP]"]
        labels = ["O", "B-ORG", "I-ORG", "O", "B-LOC", "O"]
        attention_mask = np.array([1, 1, 1, 1, 1, 1])

        entities = extractor.extract(tokens, labels, attention_mask)

        assert len(entities) == 2
        assert entities[0]["label"] == "ORG"
        assert entities[0]["tokens"] == ["Apple", "Inc"]
        assert entities[1]["label"] == "LOC"
        assert entities[1]["tokens"] == ["Seattle"]

    def test_group_tokens_with_confidence(self, extractor):
        """Test grouping tokens with confidence scores"""
        tokens = ["[CLS]", "Apple", "Inc", "[SEP]"]
        labels = ["O", "B-ORG", "I-ORG", "O"]
        attention_mask = np.array([1, 1, 1, 1])
        # Mock probabilities and label IDs
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [CLS]
            [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Apple -> B-ORG
            [0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Inc -> I-ORG
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [SEP]
        ])
        label_ids = np.array([0, 1, 2, 0])

        words = extractor._group_tokens_into_words(
            tokens, labels, attention_mask, probabilities, label_ids
        )

        assert len(words) == 2
        assert words[0]["confidences"] == [0.99]
        assert words[1]["confidences"] == [0.98]

    def test_aggregate_with_confidence(self, extractor):
        """Test that confidence is calculated as mean for multi-token entities"""
        words = [
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 1,
                "end_token": 1,
                "confidences": [0.99],
            },
            {
                "tokens": ["Inc"],
                "labels": ["I-ORG"],
                "start_token": 2,
                "end_token": 2,
                "confidences": [0.97],
            },
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 1
        assert "confidence" in entities[0]
        # Mean of [0.99, 0.97] = 0.98
        assert abs(entities[0]["confidence"] - 0.98) < 0.001

    def test_aggregate_confidence_saved_before_O_tag(self, extractor):
        """Test that confidence is calculated when entity ends with O tag"""
        words = [
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 1,
                "end_token": 1,
                "confidences": [0.95],
            },
            {
                "tokens": ["in"],
                "labels": ["O"],
                "start_token": 2,
                "end_token": 2,
            },
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 1
        assert "confidence" in entities[0]
        assert entities[0]["confidence"] == 0.95

    def test_aggregate_consecutive_same_type_b_tags(self, extractor):
        """
        Test that consecutive B- tags of the SAME type create separate entities.

        According to BIO scheme and HuggingFace docs:
        "B-PER: Beginning of a person's name right after another person's name"

        This means B-PER → B-PER should be TWO separate people, not one merged entity.
        Example: "John" (B-PER) → "Smith" (B-PER) = two people, not "John Smith"
        """
        words = [
            {
                "tokens": ["Berlin"],
                "labels": ["B-LOC"],
                "start_token": 1,
                "end_token": 1,
            },
            {
                "tokens": ["Paris"],
                "labels": ["B-LOC"],  # New B-tag = new entity
                "start_token": 2,
                "end_token": 2,
            },
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create TWO separate location entities
        assert len(entities) == 2
        assert entities[0]["label"] == "LOC"
        assert entities[0]["tokens"] == ["Berlin"]
        assert entities[1]["label"] == "LOC"
        assert entities[1]["tokens"] == ["Paris"]

    def test_aggregate_multiple_consecutive_same_type(self, extractor):
        """Test three consecutive entities of same type"""
        words = [
            {"tokens": ["John"], "labels": ["B-PER"], "start_token": 1, "end_token": 1},
            {"tokens": ["Paul"], "labels": ["B-PER"], "start_token": 2, "end_token": 2},
            {"tokens": ["George"], "labels": ["B-PER"], "start_token": 3, "end_token": 3},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create THREE separate person entities
        assert len(entities) == 3
        assert all(e["label"] == "PER" for e in entities)
        assert entities[0]["tokens"] == ["John"]
        assert entities[1]["tokens"] == ["Paul"]
        assert entities[2]["tokens"] == ["George"]

    def test_aggregate_b_tag_followed_by_i_tag(self, extractor):
        """Test standard case: B- followed by I- should merge"""
        words = [
            {"tokens": ["New"], "labels": ["B-LOC"], "start_token": 1, "end_token": 1},
            {"tokens": ["York"], "labels": ["I-LOC"], "start_token": 2, "end_token": 2},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create ONE entity
        assert len(entities) == 1
        assert entities[0]["label"] == "LOC"
        assert entities[0]["tokens"] == ["New", "York"]

    def test_aggregate_consecutive_organizations(self, extractor):
        """Test consecutive B-ORG tags create separate organization entities"""
        words = [
            {"tokens": ["Microsoft"], "labels": ["B-ORG"], "start_token": 1, "end_token": 1},
            {"tokens": ["Apple"], "labels": ["B-ORG"], "start_token": 2, "end_token": 2},
            {"tokens": ["Google"], "labels": ["B-ORG"], "start_token": 3, "end_token": 3},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create THREE separate organization entities
        assert len(entities) == 3
        assert all(e["label"] == "ORG" for e in entities)
        assert entities[0]["tokens"] == ["Microsoft"]
        assert entities[1]["tokens"] == ["Apple"]
        assert entities[2]["tokens"] == ["Google"]

    def test_aggregate_consecutive_misc(self, extractor):
        """Test consecutive B-MISC tags create separate miscellaneous entities"""
        words = [
            {"tokens": ["iPhone"], "labels": ["B-MISC"], "start_token": 1, "end_token": 1},
            {"tokens": ["iPad"], "labels": ["B-MISC"], "start_token": 2, "end_token": 2},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create TWO separate MISC entities
        assert len(entities) == 2
        assert all(e["label"] == "MISC" for e in entities)
        assert entities[0]["tokens"] == ["iPhone"]
        assert entities[1]["tokens"] == ["iPad"]

    def test_aggregate_all_entity_types(self, extractor):
        """Test all four entity types: PER, ORG, LOC, MISC"""
        words = [
            {"tokens": ["John"], "labels": ["B-PER"], "start_token": 1, "end_token": 1},
            {"tokens": ["Apple"], "labels": ["B-ORG"], "start_token": 2, "end_token": 2},
            {"tokens": ["Paris"], "labels": ["B-LOC"], "start_token": 3, "end_token": 3},
            {"tokens": ["iPhone"], "labels": ["B-MISC"], "start_token": 4, "end_token": 4},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        assert len(entities) == 4
        assert entities[0]["label"] == "PER"
        assert entities[1]["label"] == "ORG"
        assert entities[2]["label"] == "LOC"
        assert entities[3]["label"] == "MISC"

    def test_aggregate_malformed_i_without_b(self, extractor):
        """
        Test handling of malformed BIO sequence: I- tag without preceding B- tag.

        Should treat I-tag as start of new entity (graceful degradation).
        """
        words = [
            {
                "tokens": ["Smith"],
                "labels": ["I-PER"],  # Malformed: I- without B-
                "start_token": 1,
                "end_token": 1,
            },
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should still create an entity (graceful handling)
        assert len(entities) == 1
        assert entities[0]["label"] == "PER"
        assert entities[0]["tokens"] == ["Smith"]

    def test_aggregate_all_o_tags(self, extractor):
        """Test sequence with no entities (all O tags)"""
        words = [
            {"tokens": ["the"], "labels": ["O"], "start_token": 1, "end_token": 1},
            {"tokens": ["quick"], "labels": ["O"], "start_token": 2, "end_token": 2},
            {"tokens": ["brown"], "labels": ["O"], "start_token": 3, "end_token": 3},
        ]

        entities = extractor._aggregate_words_into_entities(words)

        # Should create NO entities
        assert len(entities) == 0

    def test_group_tokens_entity_at_start(self, extractor):
        """Test entity at very start of sequence (after [CLS])"""
        tokens = ["[CLS]", "Apple", "is", "great", "[SEP]"]
        labels = ["O", "B-ORG", "O", "O", "O"]
        attention_mask = np.array([1, 1, 1, 1, 1])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        # Should handle entity at start correctly
        assert len(words) == 3  # "Apple", "is", "great"
        assert words[0]["tokens"] == ["Apple"]
        assert words[0]["labels"] == ["B-ORG"]

    def test_group_tokens_entity_at_end(self, extractor):
        """Test entity at very end of sequence (before [SEP])"""
        tokens = ["[CLS]", "I", "like", "Apple", "[SEP]"]
        labels = ["O", "O", "O", "B-ORG", "O"]
        attention_mask = np.array([1, 1, 1, 1, 1])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        # Should handle entity at end correctly
        assert len(words) == 3  # "I", "like", "Apple"
        assert words[2]["tokens"] == ["Apple"]
        assert words[2]["labels"] == ["B-ORG"]

    def test_group_tokens_empty_after_special_tokens(self, extractor):
        """Test sequence with only special tokens"""
        tokens = ["[CLS]", "[SEP]"]
        labels = ["O", "O"]
        attention_mask = np.array([1, 1])

        words = extractor._group_tokens_into_words(tokens, labels, attention_mask)

        # Should return empty list (no real tokens)
        assert len(words) == 0


# ============================================================================
# Unit Tests: _deduplicate_entities
# ============================================================================


class TestDeduplicateEntities:
    """Unit tests for _deduplicate_entities helper function"""

    def test_deduplicate_exact_duplicates(self):
        """Test removing exact duplicate entities"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.99},
            {"text": "Apple", "label": "ORG", "confidence": 0.98},
            {"text": "Apple", "label": "ORG", "confidence": 0.97},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 1
        assert result[0]["text"] == "Apple"
        assert result[0]["label"] == "ORG"
        # Confidence should be averaged: (0.99 + 0.98 + 0.97) / 3 = 0.98
        assert abs(result[0]["confidence"] - 0.98) < 0.001

    def test_deduplicate_case_insensitive(self):
        """Test case-insensitive deduplication"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.95},
            {"text": "APPLE", "label": "ORG", "confidence": 0.90},
            {"text": "apple", "label": "ORG", "confidence": 0.85},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 1
        # Should preserve first occurrence's casing
        assert result[0]["text"] == "Apple"
        # Confidence averaged: (0.95 + 0.90 + 0.85) / 3 = 0.90
        assert abs(result[0]["confidence"] - 0.90) < 0.001

    def test_deduplicate_different_labels_kept_separate(self):
        """Test that same text with different labels are not merged"""
        entities = [
            {"text": "Washington", "label": "PER", "confidence": 0.95},
            {"text": "Washington", "label": "LOC", "confidence": 0.90},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 2
        labels = {e["label"] for e in result}
        assert labels == {"PER", "LOC"}

    def test_deduplicate_no_duplicates(self):
        """Test that unique entities are unchanged"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.99},
            {"text": "Microsoft", "label": "ORG", "confidence": 0.98},
            {"text": "Google", "label": "ORG", "confidence": 0.97},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 3
        texts = {e["text"] for e in result}
        assert texts == {"Apple", "Microsoft", "Google"}

    def test_deduplicate_without_confidence(self):
        """Test deduplication works even without confidence scores"""
        entities = [
            {"text": "Apple", "label": "ORG"},
            {"text": "Apple", "label": "ORG"},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 1
        assert result[0]["text"] == "Apple"
        assert "confidence" not in result[0]

    def test_deduplicate_preserves_other_fields(self):
        """Test that deduplication preserves other entity fields"""
        entities = [
            {"text": "Apple", "label": "ORG", "start_token": 1, "end_token": 1, "tokens": ["Apple"], "confidence": 0.99},
            {"text": "Apple", "label": "ORG", "start_token": 5, "end_token": 5, "tokens": ["Apple"], "confidence": 0.97},
        ]

        result = distilbert_ner._deduplicate_entities(entities)

        assert len(result) == 1
        # Should preserve first occurrence's metadata
        assert result[0]["start_token"] == 1
        assert result[0]["end_token"] == 1
        assert result[0]["tokens"] == ["Apple"]
        # But average the confidence
        assert abs(result[0]["confidence"] - 0.98) < 0.001

    def test_deduplicate_empty_list(self):
        """Test deduplication handles empty list"""
        result = distilbert_ner._deduplicate_entities([])
        assert result == []


# ============================================================================
# Unit Tests: format_entities
# ============================================================================


class TestFormatEntities:
    """Unit tests for format_entities helper function"""

    def test_format_without_labels(self):
        """Test formatting entities as plain text list"""
        entities = [
            {"text": "Microsoft", "label": "ORG"},
            {"text": "Satya Nadella", "label": "PER"},
            {"text": "Seattle", "label": "LOC"},
        ]

        result = distilbert_ner.format_entities(entities, include_labels=False)

        assert result == ["Microsoft", "Satya Nadella", "Seattle"]

    def test_format_with_labels(self):
        """Test formatting entities with expanded labels"""
        entities = [
            {"text": "Microsoft", "label": "ORG"},
            {"text": "Satya Nadella", "label": "PER"},
            {"text": "Seattle", "label": "LOC"},
            {"text": "AI Summit", "label": "MISC"},
        ]

        result = distilbert_ner.format_entities(entities, include_labels=True)

        assert result == [
            "Microsoft (Organization)",
            "Satya Nadella (Person)",
            "Seattle (Location)",
            "AI Summit",  # MISC returns plain text
        ]

    def test_format_empty_list(self):
        """Test formatting empty entity list"""
        result = distilbert_ner.format_entities([], include_labels=False)
        assert result == []

        result = distilbert_ner.format_entities([], include_labels=True)
        assert result == []

    def test_format_preserves_order(self):
        """Test that entity order is preserved"""
        entities = [
            {"text": "First", "label": "ORG"},
            {"text": "Second", "label": "PER"},
            {"text": "Third", "label": "LOC"},
        ]

        result = distilbert_ner.format_entities(entities)
        assert result == ["First", "Second", "Third"]

    def test_format_with_confidence(self):
        """Test formatting entities with confidence scores"""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.9876},
            {"text": "Satya Nadella", "label": "PER", "confidence": 0.9234},
            {"text": "Seattle", "label": "LOC", "confidence": 0.9567},
        ]

        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )

        assert result == [
            "Microsoft (entity_type:Organization, conf:0.99)",
            "Satya Nadella (entity_type:Person, conf:0.92)",
            "Seattle (entity_type:Location, conf:0.96)",
        ]

    def test_format_misc_always_plain_text(self):
        """Test that MISC entities always return plain text, even with labels/confidence"""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99},
            {"text": "iPhone 15", "label": "MISC", "confidence": 0.85},
            {"text": "Nobel Prize", "label": "MISC", "confidence": 0.95},
        ]

        # With labels only
        result = distilbert_ner.format_entities(entities, include_labels=True)
        assert result == [
            "Microsoft (Organization)",
            "iPhone 15",  # MISC: plain text
            "Nobel Prize",  # MISC: plain text
        ]

        # With labels and confidence
        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )
        assert result == [
            "Microsoft (entity_type:Organization, conf:0.99)",
            "iPhone 15",  # MISC: plain text, no metadata
            "Nobel Prize",  # MISC: plain text, no metadata
        ]

    def test_format_confidence_requires_labels(self):
        """Test that confidence without labels still returns plain text"""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99},
        ]

        # Confidence without labels should be ignored
        result = distilbert_ner.format_entities(
            entities, include_labels=False, include_confidence=True
        )
        assert result == ["Microsoft"]

    def test_format_with_deduplication_enabled(self):
        """Test that deduplication is enabled by default"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.99},
            {"text": "Apple", "label": "ORG", "confidence": 0.97},
            {"text": "Microsoft", "label": "ORG", "confidence": 0.95},
        ]

        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )

        # Should deduplicate by default
        assert len(result) == 2
        assert "Apple" in result[0]
        assert "Microsoft" in result[1]
        # Apple confidence should be averaged
        assert "conf:0.98" in result[0]

    def test_format_with_deduplication_disabled(self):
        """Test that deduplication can be disabled"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.99},
            {"text": "Apple", "label": "ORG", "confidence": 0.97},
            {"text": "Microsoft", "label": "ORG", "confidence": 0.95},
        ]

        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True, deduplicate=False
        )

        # Should NOT deduplicate when disabled
        assert len(result) == 3
        assert result[0] == "Apple (entity_type:Organization, conf:0.99)"
        assert result[1] == "Apple (entity_type:Organization, conf:0.97)"
        assert result[2] == "Microsoft (entity_type:Organization, conf:0.95)"

    def test_format_deduplication_case_insensitive(self):
        """Test that format_entities uses case-insensitive deduplication"""
        entities = [
            {"text": "Apple", "label": "ORG", "confidence": 0.95},
            {"text": "APPLE", "label": "ORG", "confidence": 0.90},
            {"text": "apple", "label": "ORG", "confidence": 0.85},
        ]

        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )

        # Should deduplicate case-insensitively
        assert len(result) == 1
        # Should preserve first occurrence's casing
        assert result[0].startswith("Apple")
        # Should average confidence
        assert "conf:0.90" in result[0]


# ============================================================================
# Integration Tests: End-to-End Inference
# ============================================================================


class TestEndToEndInference:
    """Integration test using the actual ONNX model"""

    def test_predict_entities_real_model(self):
        """
        End-to-end test with actual model inference.
        """
        text = "Apple Inc. is located in Cupertino, California."

        entities = distilbert_ner.predict_entities(text)

        # Verify we got entities
        assert len(entities) > 0

        # Verify structure of entities (now includes confidence)
        for entity in entities:
            assert "text" in entity
            assert "label" in entity
            assert "start_token" in entity
            assert "end_token" in entity
            assert "tokens" in entity
            assert "confidence" in entity
            # Confidence should be a float between 0 and 1
            assert isinstance(entity["confidence"], float)
            assert 0.0 <= entity["confidence"] <= 1.0

        # Verify expected entities (may vary based on model)
        entity_texts = [e["text"] for e in entities]
        assert "Apple Inc." in entity_texts or "Apple" in entity_texts  # Organization
        assert "Cupertino" in entity_texts  # Location
        assert "California" in entity_texts  # Location

    def test_format_entities_integration(self):
        """Test format_entities with real prediction results"""
        text = "Microsoft CEO Satya Nadella announced a partnership with OpenAI in Seattle."

        entities = distilbert_ner.predict_entities(text)
        plain_texts = distilbert_ner.format_entities(entities, include_labels=False)
        labeled_texts = distilbert_ner.format_entities(entities, include_labels=True)
        confidence_texts = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )

        # Verify formatting works
        assert len(plain_texts) == len(entities)
        assert len(labeled_texts) == len(entities)
        assert len(confidence_texts) == len(entities)

        # Verify labels are expanded (but MISC returns plain text)
        for entity, labeled in zip(entities, labeled_texts):
            if entity["label"] == "MISC":
                # MISC should be plain text
                assert "(" not in labeled
            else:
                # PER/ORG/LOC should have labels
                assert "(" in labeled and ")" in labeled
                assert any(
                    exp in labeled
                    for exp in ["Organization", "Person", "Location"]
                )

        # Verify confidence formatting (but MISC returns plain text)
        for entity, conf_text in zip(entities, confidence_texts):
            if entity["label"] == "MISC":
                # MISC should be plain text
                assert "entity_type:" not in conf_text
                assert "conf:" not in conf_text
            else:
                # PER/ORG/LOC should have confidence
                assert "entity_type:" in conf_text
                assert "conf:" in conf_text

    def test_misc_entity_handling(self):
        """Test that MISC entities are included but without metadata"""
        text = "I bought an iPhone 15 and won the Nobel Prize."

        entities = distilbert_ner.predict_entities(text)

        # Check if we detected any MISC entities
        misc_entities = [e for e in entities if e["label"] == "MISC"]

        if misc_entities:
            # Format with labels and confidence
            formatted = distilbert_ner.format_entities(
                entities, include_labels=True, include_confidence=True
            )

            # MISC entities should appear as plain text
            for entity, formatted_text in zip(entities, formatted):
                if entity["label"] == "MISC":
                    # Should be just the text, no metadata
                    assert formatted_text == entity["text"]
                    assert "entity_type:" not in formatted_text
                    assert "conf:" not in formatted_text

    def test_deduplication_integration(self):
        """Test deduplication with real model on text with repeated entities"""
        # Text designed to trigger duplicate entity detections
        text = "Apple announced new products. Apple CEO Tim Cook said Apple is committed to innovation."

        entities = distilbert_ner.predict_entities(text)

        # Format without deduplication
        formatted_no_dedup = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True, deduplicate=False
        )

        # Format with deduplication (default)
        formatted_with_dedup = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True, deduplicate=True
        )

        # Count "Apple" occurrences in both
        apple_count_no_dedup = sum(1 for item in formatted_no_dedup if item.startswith("Apple"))
        apple_count_with_dedup = sum(1 for item in formatted_with_dedup if item.startswith("Apple"))

        # With deduplication, we should have fewer or equal Apple mentions
        assert apple_count_with_dedup <= apple_count_no_dedup

        # If there were duplicates, verify deduplication worked
        if apple_count_no_dedup > apple_count_with_dedup:
            # Verify at least one duplicate was removed
            assert len(formatted_with_dedup) < len(formatted_no_dedup)

    def test_consecutive_locations_real_inference(self):
        """
        Test that consecutive locations are detected as SEPARATE entities.

        According to BIO scheme: B-LOC after B-LOC = two separate locations.
        Example: "Berlin to Paris to London" should be 3 locations, not 1.
        """
        text = "I traveled from Berlin to Paris to London last summer."

        entities = distilbert_ner.predict_entities(text)
        locations = [e for e in entities if e["label"] == "LOC"]

        # Should detect at least 2-3 separate location entities
        # (model might miss some, but should NOT merge them into one)
        assert len(locations) >= 2, f"Expected multiple locations, got: {locations}"

        # Verify they are separate entities with different text
        location_texts = [e["text"].strip() for e in locations]
        assert len(set(location_texts)) >= 2, f"Locations should be distinct: {location_texts}"

        # Common case: should detect "Berlin", "Paris", "London" as separate
        if len(locations) == 3:
            assert "Berlin" in location_texts
            assert "Paris" in location_texts
            assert "London" in location_texts

    def test_consecutive_people_list_real_inference(self):
        """
        Test that people in a list are detected as separate entities.

        Example: "John, Paul, George" should be 3 people, not 1 merged entity.
        """
        text = "The meeting included John, Paul, and George from the team."

        entities = distilbert_ner.predict_entities(text)
        people = [e for e in entities if e["label"] == "PER"]

        # Should detect multiple separate person entities
        assert len(people) >= 2, f"Expected multiple people, got: {people}"

        # Verify they are separate with different names
        person_names = [e["text"].strip() for e in people]
        assert len(set(person_names)) >= 2, f"People should be distinct: {person_names}"

    def test_consecutive_organizations_real_inference(self):
        """
        Test that consecutive organizations are detected separately.

        Example: "Microsoft, Apple, Google" should be 3 orgs, not 1.
        """
        text = "Microsoft, Apple, and Google are the top tech companies."

        entities = distilbert_ner.predict_entities(text)
        orgs = [e for e in entities if e["label"] == "ORG"]

        # Should detect multiple organizations
        assert len(orgs) >= 2, f"Expected multiple organizations, got: {orgs}"

        # Verify they are distinct
        org_names = [e["text"].strip() for e in orgs]
        assert len(set(org_names)) >= 2, f"Organizations should be distinct: {org_names}"

    def test_entity_at_sentence_start_real_inference(self):
        """Test that entities at the start of a sentence are detected correctly"""
        text = "Apple announced new products today."

        entities = distilbert_ner.predict_entities(text)

        # Should detect at least one entity (Apple)
        assert len(entities) >= 1

        # First entity should be Apple (organization)
        first_entity = entities[0]
        assert "Apple" in first_entity["text"]
        assert first_entity["label"] == "ORG"

    def test_entity_at_sentence_end_real_inference(self):
        """Test that entities at the end of a sentence are detected correctly"""
        text = "The new CEO is Tim Cook."

        entities = distilbert_ner.predict_entities(text)
        people = [e for e in entities if e["label"] == "PER"]

        # Should detect Tim Cook at the end
        assert len(people) >= 1
        assert "Cook" in people[0]["text"] or "Tim Cook" in people[0]["text"]

    def test_mixed_entity_types_consecutive_real_inference(self):
        """
        Test consecutive entities of DIFFERENT types.

        Example: "Apple CEO Tim Cook" has ORG immediately followed by PER.
        """
        text = "Apple CEO Tim Cook announced the new product."

        entities = distilbert_ner.predict_entities(text)

        # Should detect both organization and person
        orgs = [e for e in entities if e["label"] == "ORG"]
        people = [e for e in entities if e["label"] == "PER"]

        assert len(orgs) >= 1, "Should detect Apple (ORG)"
        assert len(people) >= 1, "Should detect Tim Cook (PER)"

        # Verify they are NOT merged together
        org_texts = [e["text"] for e in orgs]
        person_texts = [e["text"] for e in people]

        # Apple should be in orgs, not people
        assert any("Apple" in text for text in org_texts)
        # Tim/Cook should be in people, not orgs
        assert any("Tim" in text or "Cook" in text for text in person_texts)

    def test_all_entity_types_in_one_sentence_real_inference(self):
        """Test detection of all four entity types in a single sentence"""
        text = "Tim Cook visited Paris to meet with UNESCO about the iPhone release."

        entities = distilbert_ner.predict_entities(text)

        # Collect entities by type
        by_type = {}
        for entity in entities:
            label = entity["label"]
            if label not in by_type:
                by_type[label] = []
            by_type[label].append(entity["text"])

        # Should ideally detect:
        # - PER: Tim Cook
        # - LOC: Paris
        # - ORG: UNESCO
        # - MISC: iPhone (maybe)

        # At minimum, should detect at least 2 different entity types
        assert len(by_type) >= 2, f"Expected multiple entity types, got: {by_type}"


# ============================================================================
# Test Configuration
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def suppress_model_loading_output():
    """Suppress verbose output during model loading in tests"""
    # This runs once per test session
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
