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
        probabilities = np.array(
            [
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [CLS]
                [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Apple -> B-ORG
                [0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Inc -> I-ORG
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [SEP]
            ]
        )
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
            {
                "tokens": ["George"],
                "labels": ["B-PER"],
                "start_token": 3,
                "end_token": 3,
            },
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
            {
                "tokens": ["Microsoft"],
                "labels": ["B-ORG"],
                "start_token": 1,
                "end_token": 1,
            },
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 2,
                "end_token": 2,
            },
            {
                "tokens": ["Google"],
                "labels": ["B-ORG"],
                "start_token": 3,
                "end_token": 3,
            },
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
            {
                "tokens": ["iPhone"],
                "labels": ["B-MISC"],
                "start_token": 1,
                "end_token": 1,
            },
            {
                "tokens": ["iPad"],
                "labels": ["B-MISC"],
                "start_token": 2,
                "end_token": 2,
            },
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
            {
                "tokens": ["Apple"],
                "labels": ["B-ORG"],
                "start_token": 2,
                "end_token": 2,
            },
            {
                "tokens": ["Paris"],
                "labels": ["B-LOC"],
                "start_token": 3,
                "end_token": 3,
            },
            {
                "tokens": ["iPhone"],
                "labels": ["B-MISC"],
                "start_token": 4,
                "end_token": 4,
            },
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
            {
                "text": "Apple",
                "label": "ORG",
                "start_token": 1,
                "end_token": 1,
                "tokens": ["Apple"],
                "confidence": 0.99,
            },
            {
                "text": "Apple",
                "label": "ORG",
                "start_token": 5,
                "end_token": 5,
                "tokens": ["Apple"],
                "confidence": 0.97,
            },
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
        """Test formatting entities with short labels"""
        entities = [
            {"text": "Microsoft", "label": "ORG"},
            {"text": "Satya Nadella", "label": "PER"},
            {"text": "Seattle", "label": "LOC"},
            {"text": "AI Summit", "label": "MISC"},
        ]

        result = distilbert_ner.format_entities(entities, include_labels=True)

        assert result == [
            "Microsoft [ORG]",
            "Satya Nadella [PER]",
            "Seattle [LOC]",
            "AI Summit [MISC]",
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
        """Test formatting entities with confidence scores as percentages"""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.9876},
            {"text": "Satya Nadella", "label": "PER", "confidence": 0.9234},
            {"text": "Seattle", "label": "LOC", "confidence": 0.9567},
        ]

        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )

        assert result == [
            "Microsoft [ORG:98%]",
            "Satya Nadella [PER:92%]",
            "Seattle [LOC:95%]",
        ]

    def test_format_misc_with_labels(self):
        """Test that MISC entities now get formatted with labels like other entity types"""
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99},
            {"text": "iPhone 15", "label": "MISC", "confidence": 0.85},
            {"text": "Nobel Prize", "label": "MISC", "confidence": 0.95},
        ]

        # With labels only
        result = distilbert_ner.format_entities(entities, include_labels=True)
        assert result == [
            "Microsoft [ORG]",
            "iPhone 15 [MISC]",
            "Nobel Prize [MISC]",
        ]

        # With labels and confidence
        result = distilbert_ner.format_entities(
            entities, include_labels=True, include_confidence=True
        )
        assert result == [
            "Microsoft [ORG:99%]",
            "iPhone 15 [MISC:85%]",
            "Nobel Prize [MISC:95%]",
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
        # Apple confidence should be averaged (0.98 -> 98%)
        assert "98%" in result[0]

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
        assert result[0] == "Apple [ORG:99%]"
        assert result[1] == "Apple [ORG:97%]"
        assert result[2] == "Microsoft [ORG:95%]"

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
        # Should average confidence (0.90 -> 90%)
        assert "90%" in result[0]


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

        # Verify all entity types get labels in brackets
        for entity, labeled in zip(entities, labeled_texts):
            # All entities (including MISC) should have labels in brackets
            assert "[" in labeled and "]" in labeled
            assert any(exp in labeled for exp in ["ORG", "PER", "LOC", "MISC"])

        # Verify all entity types get confidence formatting
        for entity, conf_text in zip(entities, confidence_texts):
            # All entities (including MISC) should have confidence percentage
            assert "[" in conf_text and "]" in conf_text
            assert "%" in conf_text

    def test_misc_entity_handling(self):
        """Test that MISC entities are now formatted with labels like other entity types"""
        text = "I bought an iPhone 15 and won the Nobel Prize."

        entities = distilbert_ner.predict_entities(text)

        # Check if we detected any MISC entities
        misc_entities = [e for e in entities if e["label"] == "MISC"]

        if misc_entities:
            # Format with labels and confidence
            formatted = distilbert_ner.format_entities(
                entities, include_labels=True, include_confidence=True
            )

            # MISC entities should now have labels and confidence
            for entity, formatted_text in zip(entities, formatted):
                if entity["label"] == "MISC":
                    # Should have label and confidence like other entity types
                    assert entity["text"] in formatted_text
                    assert "[MISC:" in formatted_text
                    assert "%" in formatted_text

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
        apple_count_no_dedup = sum(
            1 for item in formatted_no_dedup if item.startswith("Apple")
        )
        apple_count_with_dedup = sum(
            1 for item in formatted_with_dedup if item.startswith("Apple")
        )

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
        assert len(set(location_texts)) >= 2, (
            f"Locations should be distinct: {location_texts}"
        )

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
        assert len(set(org_names)) >= 2, (
            f"Organizations should be distinct: {org_names}"
        )

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
# Integration Tests: Chunking and Stride for Long Text
# ============================================================================


class TestChunkingAndStride:
    """
    Tests for sliding window chunking to handle long text.

    The model has MAX_LENGTH=512 tokens. For longer text, we use a sliding window
    with configurable stride to process overlapping chunks and merge results.
    """

    def test_single_chunk_short_text(self):
        """
        Test that short text (< 512 tokens) works correctly with single chunk.

        This is a regression test to ensure chunking doesn't break normal operation.
        """
        text = (
            "Alice works at Microsoft in Seattle. Bob works at Google in San Francisco."
        )

        entities = distilbert_ner.predict_entities(text)

        # Should detect entities from both sentences
        entity_texts = [e["text"] for e in entities]

        assert len(entities) >= 4  # At minimum: Alice, Microsoft, Bob, Google
        assert any("Alice" in text for text in entity_texts)
        assert any("Microsoft" in text for text in entity_texts)
        assert any("Bob" in text for text in entity_texts)
        assert any("Google" in text for text in entity_texts)

        # All entities should have confidence scores
        for entity in entities:
            assert "confidence" in entity
            assert 0.0 <= entity["confidence"] <= 1.0

        # Should NOT have chunk metadata in final output
        for entity in entities:
            assert "chunk_idx" not in entity

    def test_multi_chunk_long_text(self):
        """
        Test that text requiring multiple chunks extracts entities from all chunks.

        This test uses a long multi-paragraph text that exceeds 512 tokens,
        requiring the sliding window to process multiple chunks.
        """
        # Create text with entities distributed across ~260+ tokens (needs 3+ chunks)
        paragraph1 = "Alice worked at Microsoft in Seattle with her colleague Bob. They collaborated on a project with Carol from the New York office. "
        paragraph2 = "Meanwhile, David was based in the London office and coordinated with Emma in Paris. The team also included Frank from the Tokyo branch. "
        paragraph3 = "Later that month, Grace flew to Berlin to meet with Henry at the Munich office. They discussed the expansion into Rome and Athens. "
        paragraph4 = "The final meeting included Isabel from the Boston headquarters and Jack from the Chicago office discussing partnerships with firms in Miami. "

        text = paragraph1 + paragraph2 + paragraph3 + paragraph4

        entities = distilbert_ner.predict_entities(text)
        entity_texts = [e["text"] for e in entities]

        # Verify we extracted entities from ALL paragraphs, not just the first
        # Paragraph 1 entities
        assert any("Alice" in text for text in entity_texts), (
            "Should find Alice from paragraph 1"
        )
        assert any("Microsoft" in text for text in entity_texts), (
            "Should find Microsoft from paragraph 1"
        )

        # Paragraph 2 entities
        assert any("London" in text for text in entity_texts), (
            "Should find London from paragraph 2"
        )
        assert any("Paris" in text for text in entity_texts), (
            "Should find Paris from paragraph 2"
        )

        # Paragraph 3 entities
        assert any("Berlin" in text for text in entity_texts), (
            "Should find Berlin from paragraph 3"
        )
        assert any("Munich" in text for text in entity_texts), (
            "Should find Munich from paragraph 3"
        )

        # Paragraph 4 entities
        assert any("Boston" in text for text in entity_texts), (
            "Should find Boston from paragraph 4"
        )
        assert any("Chicago" in text for text in entity_texts), (
            "Should find Chicago from paragraph 4"
        )

        # All entities should have confidence scores
        for entity in entities:
            assert "confidence" in entity

        # Should NOT have chunk metadata in final output
        for entity in entities:
            assert "chunk_idx" not in entity

    def test_chunk_deduplication(self):
        """
        Test that entities appearing in overlapping chunk regions are deduplicated.

        With stride < max_length, chunks overlap. An entity near the boundary
        might appear in both chunks and should be deduplicated (keeping highest confidence).
        """
        # Create text where an entity will likely appear in multiple chunks
        # Repeat "Microsoft" multiple times to increase chance of overlap
        text = " ".join(
            [
                "Alice works at Microsoft.",
                "Bob also works at Microsoft.",
                "Charlie joined Microsoft recently.",
                "David leads the Microsoft Azure team.",
                "Emma manages Microsoft Office products.",
                "Frank develops for Microsoft Windows.",
                "Grace works on Microsoft Teams.",
                "Henry maintains Microsoft SQL Server.",
                "Isabel oversees Microsoft Dynamics.",
                "Jack contributes to Microsoft Edge browser.",
            ]
        )

        # Get entities WITHOUT deduplication
        entities = distilbert_ner.predict_entities(text)

        # Count how many times "Microsoft" appears
        microsoft_entities = [
            e for e in entities if "Microsoft" in e["text"] and e["label"] == "ORG"
        ]

        # Due to chunk overlap and _deduplicate_chunk_entities, we should have
        # fewer Microsoft entities than the 10 mentions in the text
        # (The function deduplicates across chunks, keeping highest confidence)
        assert len(microsoft_entities) >= 1, "Should find at least one Microsoft entity"

        # If we got multiple Microsoft entities, they should have different contexts
        # (e.g., "Microsoft Azure", "Microsoft Office", etc.)
        # or be exact duplicates from different chunks that got deduplicated

        # Verify all have confidence
        for entity in microsoft_entities:
            assert "confidence" in entity

    def test_custom_stride_parameter(self):
        """
        Test that custom stride parameter works correctly.

        Smaller stride = more overlap = more thorough but slower.
        Larger stride = less overlap = faster but might miss boundary entities.
        """
        # Text long enough to require chunking
        text = " ".join(
            [
                "Alice works at Microsoft in Seattle.",
                "Bob collaborates with Carol at Google in San Francisco.",
                "David leads the team at Apple in Cupertino.",
                "Emma manages the project at Amazon in San Jose.",
                "Frank coordinates with Grace at Meta in Menlo Park.",
                "Henry works with Isabel at Tesla in Palo Alto.",
            ]
        )

        # Test with default stride (64)
        entities_default = distilbert_ner.predict_entities(text, stride=64)

        # Test with smaller stride (more overlap)
        entities_small_stride = distilbert_ner.predict_entities(text, stride=32)

        # Test with larger stride (less overlap)
        entities_large_stride = distilbert_ner.predict_entities(text, stride=96)

        # All should extract entities
        assert len(entities_default) > 0
        assert len(entities_small_stride) > 0
        assert len(entities_large_stride) > 0

        # Smaller stride might find more entities (or same amount after deduplication)
        # This is probabilistic, so we just verify it runs without error
        # and returns reasonable results

        # All should have confidence
        for entity in entities_default + entities_small_stride + entities_large_stride:
            assert "confidence" in entity
            assert "chunk_idx" not in entity

    def test_very_long_text_multiple_paragraphs(self):
        """
        Test extraction from very long multi-paragraph text (500+ tokens).

        This simulates real-world usage with journal entries or documents.
        """
        # Construct a long text with entities distributed throughout
        paragraphs = [
            "In January, Alice started working at Microsoft in Seattle. She collaborated with Bob on the Azure platform.",
            "By February, the team expanded. Carol joined from the New York office and David came from London.",
            "March brought more changes. Emma relocated from Paris to work with Frank in the Tokyo office.",
            "April saw partnerships form. Grace traveled to Berlin to meet with Henry from the Munich division.",
            "In May, Isabel from Boston coordinated with Jack in Chicago on the expansion to Miami.",
            "June meetings included Karen from Denver and Leo from Portland discussing the Seattle headquarters.",
            "July brought international focus. Maria flew to Rome while Nathan visited Athens for client meetings.",
            "By August, the team included Oscar from Dublin and Paula from Amsterdam working on European projects.",
            "September saw Quinn from Toronto and Rachel from Vancouver joining the North American expansion.",
            "October concluded with Sam from Sydney and Tara from Melbourne leading the Asia-Pacific initiatives.",
        ]

        text = " ".join(paragraphs)

        entities = distilbert_ner.predict_entities(text)
        entity_texts = [e["text"] for e in entities]

        # Verify we extracted entities from EARLY paragraphs
        assert any("Alice" in text or "Microsoft" in text for text in entity_texts), (
            "Should find entities from first paragraph"
        )
        assert any("Seattle" in text for text in entity_texts), (
            "Should find Seattle from first paragraph"
        )

        # Verify we extracted entities from MIDDLE paragraphs
        assert any("Berlin" in text or "Munich" in text for text in entity_texts), (
            "Should find entities from middle paragraphs (April)"
        )

        # Verify we extracted entities from LATE paragraphs
        assert any("Sydney" in text or "Melbourne" in text for text in entity_texts), (
            "Should find entities from final paragraph (October)"
        )

        # Should have many entities across the long text
        assert len(entities) >= 15, (
            f"Expected many entities from long text, got {len(entities)}"
        )

        # All should have confidence and no chunk metadata
        for entity in entities:
            assert "confidence" in entity
            assert "chunk_idx" not in entity

    def test_chunk_deduplication_helper_function(self):
        """
        Unit test for _deduplicate_chunk_entities helper function.

        This function handles deduplication of entities that appear in multiple chunks.
        """
        # Simulate entities from overlapping chunks
        entities = [
            {
                "text": "Microsoft",
                "label": "ORG",
                "confidence": 0.99,
                "chunk_idx": 0,
                "start_token": 5,
                "end_token": 5,
                "tokens": ["Microsoft"],
            },
            {
                "text": "Microsoft",
                "label": "ORG",
                "confidence": 0.97,
                "chunk_idx": 1,
                "start_token": 3,
                "end_token": 3,
                "tokens": ["Microsoft"],
            },
            {
                "text": "Apple",
                "label": "ORG",
                "confidence": 0.95,
                "chunk_idx": 1,
                "start_token": 10,
                "end_token": 10,
                "tokens": ["Apple"],
            },
        ]

        result = distilbert_ner._deduplicate_chunk_entities(entities)

        # Should have 2 unique entities (Microsoft appears twice, keep highest confidence)
        assert len(result) == 2

        # Find Microsoft in result
        microsoft = [e for e in result if e["text"] == "Microsoft"][0]

        # Should keep the one with highest confidence (0.99)
        assert microsoft["confidence"] == 0.99
        assert microsoft["chunk_idx"] == 0

        # Apple should be unchanged
        apple = [e for e in result if e["text"] == "Apple"][0]
        assert apple["confidence"] == 0.95

    def test_chunk_deduplication_case_insensitive(self):
        """
        Test that chunk deduplication is case-insensitive.
        """
        entities = [
            {"text": "Microsoft", "label": "ORG", "confidence": 0.99, "chunk_idx": 0},
            {"text": "MICROSOFT", "label": "ORG", "confidence": 0.97, "chunk_idx": 1},
            {"text": "microsoft", "label": "ORG", "confidence": 0.95, "chunk_idx": 2},
        ]

        result = distilbert_ner._deduplicate_chunk_entities(entities)

        # Should deduplicate to one entity
        assert len(result) == 1

        # Should preserve first occurrence's casing
        assert result[0]["text"] == "Microsoft"

        # Should keep highest confidence
        assert result[0]["confidence"] == 0.99

    def test_chunk_deduplication_different_labels_kept_separate(self):
        """
        Test that same text with different labels are NOT deduplicated.

        Example: "Washington" could be a person (George Washington) or location.
        """
        entities = [
            {"text": "Washington", "label": "PER", "confidence": 0.90, "chunk_idx": 0},
            {"text": "Washington", "label": "LOC", "confidence": 0.85, "chunk_idx": 1},
        ]

        result = distilbert_ner._deduplicate_chunk_entities(entities)

        # Should keep both (different labels)
        assert len(result) == 2

        labels = {e["label"] for e in result}
        assert labels == {"PER", "LOC"}

    def test_entities_at_chunk_boundary(self):
        """
        Test that entities near chunk boundaries are correctly extracted.

        This is a stress test: entities appearing exactly at the boundary
        between chunks should be captured correctly.
        """
        # Create text where entity appears around the 512-token boundary
        # Use repetitive text to reach boundary, then place entity
        filler = "The quick brown fox jumps over the lazy dog. " * 60  # ~540 tokens
        text = filler + "Alice works at Microsoft in Seattle."

        entities = distilbert_ner.predict_entities(text)
        entity_texts = [e["text"] for e in entities]

        # Should still find entities near/after the boundary
        assert any("Alice" in text for text in entity_texts), (
            "Should find Alice even though it's past the 512-token boundary"
        )
        assert any("Microsoft" in text for text in entity_texts), (
            "Should find Microsoft even though it's past the 512-token boundary"
        )

        # All should have confidence and no chunk metadata
        for entity in entities:
            assert "confidence" in entity
            assert "chunk_idx" not in entity

    def test_no_entities_in_long_text(self):
        """
        Test that long text with NO entities returns empty list.

        Regression test to ensure chunking doesn't create false positives.
        """
        # Long text with no named entities
        text = " ".join(
            [
                "The system works well when configured properly.",
                "It processes data efficiently and handles errors gracefully.",
                "Performance metrics show consistent results across tests.",
                "Integration with existing infrastructure proceeds smoothly.",
                "Documentation provides clear guidance for implementation.",
            ]
            * 5
        )  # Repeat to ensure multiple chunks

        entities = distilbert_ner.predict_entities(text)

        # Should find no entities (or very few false positives)
        assert len(entities) <= 2, (
            f"Expected no/few entities in text without names, got {len(entities)}: {entities}"
        )

    def test_entity_exactly_at_512_boundary(self):
        """
        Test entity extraction when an entity appears exactly at the 512-token boundary.

        This is a critical edge case - entities appearing right at max_length should
        be captured by the sliding window overlap.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-ner-uncased-onnx")

        # Build text that reaches exactly ~510 tokens, then add entity
        base = "The quick brown fox jumps over the lazy dog. "

        # Calculate how many repetitions to get close to 510 tokens
        base_tokens = len(tokenizer.tokenize(base))
        repetitions = 510 // base_tokens

        filler = base * repetitions

        # Add entity right at the boundary
        boundary_text = filler + "Alice works at Microsoft in Seattle."

        # Verify total tokens
        total_tokens = len(tokenizer.tokenize(boundary_text))
        print(f"Total tokens: {total_tokens} (target: ~512)")

        # Extract entities with max_length=512
        entities = distilbert_ner.predict_entities(
            boundary_text, max_length=512, stride=256
        )
        entity_texts = [e["text"] for e in entities]

        # Should find entities even though they're near/past the 512 boundary
        assert any("Alice" in text for text in entity_texts), (
            f"Should find Alice at 512-token boundary. Found: {entity_texts}"
        )
        assert any("Microsoft" in text for text in entity_texts), (
            f"Should find Microsoft at 512-token boundary. Found: {entity_texts}"
        )

    def test_entity_crosses_512_boundary(self):
        """
        Test entity that spans across the 512-token chunk boundary.

        With stride=256, there's a 256-token overlap. An entity near token 500
        should appear in both chunk 0 (tokens 0-512) and chunk 1 (tokens 256-768).
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-ner-uncased-onnx")

        # Build text to ~500 tokens, add multi-token entity, continue to ~520
        base = "The system processes data efficiently and handles errors gracefully. "
        base_tokens = len(tokenizer.tokenize(base))

        # Get to ~500 tokens
        repetitions = 500 // base_tokens
        filler_before = base * repetitions

        # Add entity that will straddle the boundary
        entity_text = "The New York office coordinated with the San Francisco team. "

        # Add more text after to push past 512
        filler_after = base * 2

        full_text = filler_before + entity_text + filler_after

        total_tokens = len(tokenizer.tokenize(full_text))
        print(f"Total tokens: {total_tokens} (should be > 512)")

        # Extract with max_length=512, stride=256
        entities = distilbert_ner.predict_entities(
            full_text, max_length=512, stride=256
        )
        entity_texts = [e["text"] for e in entities]

        # Should find locations even though they span the boundary
        assert any("New York" in text for text in entity_texts), (
            f"Should find 'New York' despite boundary crossing. Found: {entity_texts}"
        )
        assert any("San Francisco" in text for text in entity_texts), (
            f"Should find 'San Francisco' despite boundary crossing. Found: {entity_texts}"
        )

        # Verify no duplicates due to overlap (deduplication should work)
        ny_count = sum(
            1 for e in entities if "New York" in e["text"] and e["label"] == "LOC"
        )
        assert ny_count <= 1, (
            f"New York should be deduplicated, found {ny_count} instances"
        )

    def test_text_just_under_512_tokens(self):
        """
        Test text with exactly 511 tokens (just under the limit).

        Should process in a single chunk with no chunking overhead.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-ner-uncased-onnx")

        # Build text to exactly 511 tokens
        base = "Alice works at Microsoft. "
        base_tokens = len(tokenizer.tokenize(base))

        # Calculate repetitions to get ~511 tokens
        target = 511
        repetitions = target // base_tokens
        text = base * repetitions

        # Fine-tune to get exactly 511
        current_tokens = len(tokenizer.tokenize(text))
        while current_tokens < target:
            text += "Bob works at Google. "
            current_tokens = len(tokenizer.tokenize(text))

        # Trim if we went over
        while current_tokens > target:
            text = " ".join(text.split()[:-1])
            current_tokens = len(tokenizer.tokenize(text))

        print(f"Text has {current_tokens} tokens (target: 511)")

        # Should process without chunking
        entities = distilbert_ner.predict_entities(text, max_length=512, stride=256)

        # Should find entities
        assert len(entities) > 0, "Should find entities in 511-token text"

        # All should have confidence and no chunk artifacts
        for entity in entities:
            assert "confidence" in entity
            assert "chunk_idx" not in entity

    def test_text_just_over_512_tokens(self):
        """
        Test text with exactly 513 tokens (just over the limit).

        Should require minimal chunking (2 chunks with stride=256).
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-ner-uncased-onnx")

        # Build text to exactly 513 tokens
        base = "The meeting included representatives from various offices. "
        base_tokens = len(tokenizer.tokenize(base))

        # Build to ~513 tokens
        target = 513
        repetitions = target // base_tokens
        text = base * repetitions

        # Add specific entities at start and end
        text = (
            "Alice from Seattle started the meeting. "
            + text
            + "Bob from Chicago concluded the discussion."
        )

        current_tokens = len(tokenizer.tokenize(text))
        print(f"Text has {current_tokens} tokens (target: 513)")

        # Extract with chunking
        entities = distilbert_ner.predict_entities(text, max_length=512, stride=256)
        entity_texts = [e["text"] for e in entities]

        # Should find entities from both start and end (across chunks)
        assert any("Alice" in text or "Seattle" in text for text in entity_texts), (
            "Should find Alice/Seattle from first chunk"
        )
        assert any("Bob" in text or "Chicago" in text for text in entity_texts), (
            "Should find Bob/Chicago from second chunk"
        )

        # All should have confidence
        for entity in entities:
            assert "confidence" in entity
            assert "chunk_idx" not in entity

    def test_very_large_text_1000_plus_tokens(self):
        """
        Test text with 1000+ tokens requiring multiple chunks at 512 max_length.

        Verifies that entities are extracted from early, middle, and late sections.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-ner-uncased-onnx")

        # Build structured text with entities at different positions
        paragraphs = []

        # Early section (tokens 0-200)
        paragraphs.append("Alice Johnson works at Microsoft in Seattle. " * 10)

        # Middle section (tokens 400-600)
        filler = "The system processes information and generates reports. " * 20
        paragraphs.append(filler)
        paragraphs.append("Bob Smith coordinates with the London office. " * 10)

        # Late section (tokens 800-1000)
        paragraphs.append(filler)
        paragraphs.append("Carol Davis manages the Tokyo branch. " * 10)

        text = " ".join(paragraphs)

        total_tokens = len(tokenizer.tokenize(text))
        print(f"Text has {total_tokens} tokens (target: 1000+)")

        # Extract with max_length=512
        entities = distilbert_ner.predict_entities(text, max_length=512, stride=256)
        entity_texts = [e["text"] for e in entities]

        # Verify extraction from all sections
        assert any("Alice" in text or "Johnson" in text for text in entity_texts), (
            "Should find Alice from early section (tokens 0-200)"
        )
        assert any("Bob" in text or "Smith" in text for text in entity_texts), (
            "Should find Bob from middle section (tokens 400-600)"
        )
        assert any("Carol" in text or "Davis" in text for text in entity_texts), (
            "Should find Carol from late section (tokens 800-1000)"
        )

        # Should find location entities across all sections
        assert any("Seattle" in text for text in entity_texts), "Should find Seattle"
        assert any("London" in text for text in entity_texts), "Should find London"
        assert any("Tokyo" in text for text in entity_texts), "Should find Tokyo"

        # Verify we got a substantial number of entities
        assert len(entities) >= 6, (
            f"Expected at least 6 entities from 1000+ token text, got {len(entities)}"
        )


# ============================================================================
# Bug Tests: Entity Splitting with Dots and Delimiters
# ============================================================================


class TestEntitySplittingBugs:
    """
    Tests for entity splitting bugs with dots, spaces, and delimiters.

    These tests currently FAIL and demonstrate issues where entities with
    special characters (dots, periods) are incorrectly split into multiple entities.
    """

    def test_abbreviations_with_dots_should_be_single_entities(self):
        """
        Test that abbreviated names with dots are recognized as single entities.

        CURRENTLY FAILS: Entities like "J.K. Rowling", "T.S. Eliot", "C.S. Lewis"
        may be split into multiple fragments.

        EXPECTED: Each should be detected as a single person entity.
        """
        text = "J.K. Rowling wrote Harry Potter. T.S. Eliot wrote poetry. C.S. Lewis wrote Narnia."

        entities = distilbert_ner.predict_entities(text)
        person_entities = [e for e in entities if e["label"] == "PER"]

        # Should detect 3 separate people
        assert len(person_entities) >= 3, (
            f"Expected at least 3 person entities, got {len(person_entities)}: {person_entities}"
        )

        # Each person entity should contain both initials and surname
        # J.K. Rowling should have J, K, and Rowling
        rowling_entities = [e for e in person_entities if "Rowling" in e["text"]]
        assert len(rowling_entities) == 1, (
            f"Should find exactly one entity for Rowling, got: {rowling_entities}"
        )

        rowling_text = rowling_entities[0]["text"]
        assert (
            "J" in rowling_text and "K" in rowling_text and "Rowling" in rowling_text
        ), f"Rowling entity '{rowling_text}' should contain J, K, and Rowling"

        # T.S. Eliot should have T, S, and Eliot
        eliot_entities = [e for e in person_entities if "Eliot" in e["text"]]
        assert len(eliot_entities) == 1, (
            f"Should find exactly one entity for Eliot, got: {eliot_entities}"
        )

        eliot_text = eliot_entities[0]["text"]
        assert "T" in eliot_text and "S" in eliot_text and "Eliot" in eliot_text, (
            f"Eliot entity '{eliot_text}' should contain T, S, and Eliot"
        )

        # C.S. Lewis should have C, S, and Lewis
        lewis_entities = [e for e in person_entities if "Lewis" in e["text"]]
        assert len(lewis_entities) == 1, (
            f"Should find exactly one entity for Lewis, got: {lewis_entities}"
        )

        lewis_text = lewis_entities[0]["text"]
        assert "C" in lewis_text and "S" in lewis_text and "Lewis" in lewis_text, (
            f"Lewis entity '{lewis_text}' should contain C, S, and Lewis"
        )

        # No periods should be standalone entities
        period_only = [e for e in person_entities if e["text"].strip() in [".", ".."]]
        assert len(period_only) == 0, (
            f"Periods should not be standalone entities: {period_only}"
        )

    def test_titles_with_dots_should_merge_with_names(self):
        """
        Test that titles like "Dr.", "Mr.", "Mrs." merge with following names.

        EXPECTED: "Dr. Smith" should be one entity, not ["Dr", ".", "Smith"]
        """
        text = "Dr. Smith met with Mr. Johnson and Mrs. Anderson at the conference."

        entities = distilbert_ner.predict_entities(text)
        person_entities = [e for e in entities if e["label"] == "PER"]

        # Should detect 3 people
        assert len(person_entities) >= 3, (
            f"Expected at least 3 people, got {len(person_entities)}: {person_entities}"
        )

        # Each entity should contain the full name including title
        entity_texts = [e["text"] for e in person_entities]

        # Check that we have complete names (not fragments)
        smith_found = any("Smith" in text for text in entity_texts)
        johnson_found = any("Johnson" in text for text in entity_texts)
        anderson_found = any("Anderson" in text for text in entity_texts)

        assert smith_found, f"Should find Smith in entities: {entity_texts}"
        assert johnson_found, f"Should find Johnson in entities: {entity_texts}"
        assert anderson_found, f"Should find Anderson in entities: {entity_texts}"

        # Periods should NOT be standalone person entities
        period_entities = [e for e in person_entities if e["text"].strip() == "."]
        assert len(period_entities) == 0, (
            f"Period should not be a person entity: {period_entities}"
        )

    def test_usa_and_location_abbreviations(self):
        """
        Test that location abbreviations with dots are single entities.

        EXPECTED: "U.S.A.", "U.K.", "U.S." should each be single location entities.
        """
        text = "The U.S.A. and the U.K. signed a treaty. The U.S. economy is growing."

        entities = distilbert_ner.predict_entities(text)
        location_entities = [e for e in entities if e["label"] == "LOC"]

        # Should detect 2-3 location entities (U.S.A., U.K., possibly U.S.)
        assert len(location_entities) >= 2, (
            f"Expected at least 2 locations, got {len(location_entities)}: {location_entities}"
        )

        entity_texts = [e["text"] for e in location_entities]

        # Check for complete abbreviations (not fragments)
        # Each entity should have multiple characters
        for entity in location_entities:
            text = entity["text"].strip()
            # Should be longer than a single letter or period
            assert len(text) >= 2, (
                f"Location entity '{text}' seems too short - likely a fragment"
            )

        # Periods should NOT be standalone location entities
        period_entities = [
            e for e in location_entities if e["text"].strip() in [".", ".."]
        ]
        assert len(period_entities) == 0, (
            f"Periods should not be location entities: {period_entities}"
        )

    def test_same_type_split_entities_should_merge_loc(self):
        """
        Test that consecutive LOCATION fragments with dots should merge into single entity.

        SAME TYPE merging: When multiple consecutive LOC entities have dots between them,
        they should be merged into one location.
        """
        text = "The U.S.A. has strong ties with the U.K. and U.S. territories."

        entities = distilbert_ner.predict_entities(text)
        location_entities = [e for e in entities if e["label"] == "LOC"]

        # Check each location entity is complete (not fragmented)
        for entity in location_entities:
            text = entity["text"].strip()

            # No single-character location entities (would indicate splitting)
            assert len(text) > 1, (
                f"Location '{text}' is too short - likely a fragment from splitting"
            )

            # No standalone periods
            assert text not in [".", "..", "..."], (
                f"Standalone period '{text}' should not be a location entity"
            )

        # Should have 2-3 complete location entities (U.S.A., U.K., U.S.)
        assert len(location_entities) >= 2, (
            f"Expected at least 2 complete locations, got {len(location_entities)}: {location_entities}"
        )

    def test_same_type_split_entities_should_merge_org(self):
        """
        Test that organization abbreviations with dots should be single entities.

        SAME TYPE merging: ORG entities with internal punctuation should not split.
        """
        text = "He works at I.B.M. and previously worked at A.T.&T. Corp."

        entities = distilbert_ner.predict_entities(text)
        org_entities = [e for e in entities if e["label"] == "ORG"]

        # Should detect organizations
        assert len(org_entities) >= 1, (
            f"Expected at least 1 organization, got {len(org_entities)}: {org_entities}"
        )

        # Each org should be reasonably complete (not single letters)
        for entity in org_entities:
            text = entity["text"].strip()

            # Organizations should be longer than 1 character
            assert len(text) > 1, (
                f"Organization '{text}' is too short - likely a fragment"
            )

            # No standalone periods as organizations
            assert text not in [".", "..", "&"], (
                f"Punctuation '{text}' should not be an organization entity"
            )

    def test_different_types_should_not_merge(self):
        """
        Test that adjacent entities of DIFFERENT types should remain separate.

        DIFFERENT TYPE separation: This is CORRECT behavior that should be preserved.
        A fix for same-type merging should NOT break this.

        Examples:
        - "Microsoft CEO" → ORG + PER (separate)
        - "Seattle employee" → LOC + not-entity (separate)
        - "U.S. Army" → LOC + ORG (separate)
        """
        test_cases = [
            {
                "text": "Microsoft CEO Satya Nadella announced the news.",
                "expected_types": {"ORG", "PER"},
                "org_text": "Microsoft",
                "per_text": "Nadella",
            },
            {
                "text": "Apple Inc. founder Steve Jobs lived in California.",
                "expected_types": {"ORG", "PER", "LOC"},
                "org_text": "Apple",
                "per_text": "Jobs",
                "loc_text": "California",
            },
            {
                "text": "Tim Cook visited Seattle to meet with Microsoft executives.",
                "expected_types": {"PER", "LOC", "ORG"},
                "per_text": "Cook",
                "loc_text": "Seattle",
                "org_text": "Microsoft",
            },
        ]

        for case in test_cases:
            entities = distilbert_ner.predict_entities(case["text"])

            # Verify we detected multiple entity types
            detected_types = {e["label"] for e in entities}
            assert case["expected_types"].issubset(detected_types), (
                f"Expected types {case['expected_types']}, got {detected_types} in: {entities}"
            )

            # Verify specific entities are separate (not merged)
            if "org_text" in case:
                org_entities = [e for e in entities if e["label"] == "ORG"]
                assert any(case["org_text"] in e["text"] for e in org_entities), (
                    f"Should find separate ORG '{case['org_text']}' in: {entities}"
                )

            if "per_text" in case:
                per_entities = [e for e in entities if e["label"] == "PER"]
                assert any(case["per_text"] in e["text"] for e in per_entities), (
                    f"Should find separate PER '{case['per_text']}' in: {entities}"
                )

            if "loc_text" in case:
                loc_entities = [e for e in entities if e["label"] == "LOC"]
                assert any(case["loc_text"] in e["text"] for e in loc_entities), (
                    f"Should find separate LOC '{case['loc_text']}' in: {entities}"
                )

    def test_mixed_adjacent_same_and_different_types(self):
        """
        Test complex scenarios with both same-type merging and different-type separation.

        This tests the interaction between:
        1. Same-type entities with dots → SHOULD merge
        2. Different-type entities adjacent → should NOT merge
        """
        text = "G.I. Joe visited Washington D.C. and met with U.S. officials from I.B.M. Corporation."

        entities = distilbert_ner.predict_entities(text)

        # G.I. Joe should be ONE person entity (PER-PER-PER merge)
        person_entities = [e for e in entities if e["label"] == "PER"]
        gi_joe_entities = [
            e for e in person_entities if "Joe" in e["text"] or "G" in e["text"]
        ]

        # Should have ONE entity containing G.I. Joe (not 3 separate PER entities)
        # Note: This will fail until bug is fixed
        joe_complete = [
            e for e in gi_joe_entities if "Joe" in e["text"] and "G" in e["text"]
        ]
        assert len(joe_complete) >= 1, (
            f"G.I. Joe should be merged into one entity, got fragments: {gi_joe_entities}"
        )

        # Washington D.C. should be ONE location entity (LOC with punctuation)
        location_entities = [e for e in entities if e["label"] == "LOC"]
        washington_entities = [
            e for e in location_entities if "Washington" in e["text"]
        ]

        if washington_entities:
            # If detected, should be complete (not fragmented)
            assert len(washington_entities) <= 2, (
                f"Washington D.C. should be 1-2 entities max, got: {washington_entities}"
            )

        # U.S. should be ONE location (LOC-LOC merge with dots)
        us_entities = [
            e for e in location_entities if "U" in e["text"] and "S" in e["text"]
        ]
        if us_entities:
            # Should be merged, not "U" + "." + "S" as separate entities
            us_complete = [e for e in us_entities if len(e["text"].strip()) > 2]
            assert len(us_complete) >= 1, (
                f"U.S. should be complete, not fragmented: {us_entities}"
            )

        # I.B.M. should be ONE organization (ORG-ORG merge)
        org_entities = [e for e in entities if e["label"] == "ORG"]
        ibm_entities = [e for e in org_entities if "B" in e["text"]]

        if ibm_entities:
            # Should be reasonably complete
            ibm_complete = [e for e in ibm_entities if len(e["text"].strip()) > 2]
            assert len(ibm_complete) >= 1, (
                f"I.B.M. should be complete, not fragmented: {ibm_entities}"
            )

        # But PER and LOC and ORG should be SEPARATE entities (different types)
        entity_types = {e["label"] for e in entities}
        assert len(entity_types) >= 2, (
            f"Should have multiple entity types (PER, LOC, ORG), got: {entity_types}"
        )

    def test_single_letter_with_period_edge_cases(self):
        """
        Test edge cases with single letters followed by periods.

        These are tricky because:
        - "J." could be an initial (should merge with following name)
        - "A" could be an article (not an entity)
        - "I" could be a pronoun (not an entity)
        """
        text = "J. Edgar Hoover ran the F.B.I. in Washington."

        entities = distilbert_ner.predict_entities(text)
        person_entities = [e for e in entities if e["label"] == "PER"]

        # J. Edgar Hoover should be ONE person
        hoover_entities = [e for e in person_entities if "Hoover" in e["text"]]
        assert len(hoover_entities) == 1, (
            f"J. Edgar Hoover should be one entity, got: {hoover_entities}"
        )

        hoover_text = hoover_entities[0]["text"]

        # Should contain J, Edgar, and Hoover (not split)
        assert "Hoover" in hoover_text, f"Should contain Hoover: {hoover_text}"
        # May or may not contain J. Edgar depending on how tokenizer handles it

        # F.B.I. should be ONE organization
        org_entities = [e for e in entities if e["label"] == "ORG"]

        if org_entities:
            # Check for FBI entity
            fbi_entities = [
                e for e in org_entities if "B" in e["text"] or "FBI" in e["text"]
            ]

            if fbi_entities:
                # Should not be single-letter fragments
                for entity in fbi_entities:
                    assert len(entity["text"].strip()) > 1, (
                        f"FBI entity '{entity['text']}' is too short - likely a fragment"
                    )


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
