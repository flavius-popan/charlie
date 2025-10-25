"""
Tests for distilbert-ner module

Structure:
- Unit tests for EntityExtractor (mocked data)
- Unit tests for format_entities helper
- Integration test for end-to-end inference (uses actual model)
"""

import numpy as np
import pytest

# Import the module
import sys
from pathlib import Path

# Add parent directory to path to import distilbert-ner module
sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "distilbert_ner", Path(__file__).parent.parent / "distilbert-ner.py"
)
distilbert_ner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distilbert_ner)


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
            "AI Summit (Miscellaneous)",
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

        # Verify structure of entities
        for entity in entities:
            assert "text" in entity
            assert "label" in entity
            assert "start_token" in entity
            assert "end_token" in entity
            assert "tokens" in entity

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

        # Verify formatting works
        assert len(plain_texts) == len(entities)
        assert len(labeled_texts) == len(entities)

        # Verify labels are expanded
        for labeled in labeled_texts:
            assert "(" in labeled and ")" in labeled
            # Check for expanded labels
            assert any(
                exp in labeled
                for exp in ["Organization", "Person", "Location", "Miscellaneous"]
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
