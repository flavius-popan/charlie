#!/usr/bin/env python3
"""
DistilBERT NER - ONNX Inference Module

Lightweight NER inference using ONNX Runtime (no PyTorch required).
Dependencies: onnxruntime (~10MB), transformers (tokenizer only), numpy

Auto-downloads the ONNX model from HuggingFace if not present (~249MB).
Tokenizer files are included in the repository.

Usage:
    # As a module
    from distilbert_ner import predict_entities, format_entities

    entities = predict_entities("Apple Inc. is in Cupertino.")
    texts = format_entities(entities)  # ["Apple Inc.", "Cupertino"]
    labeled = format_entities(entities, include_labels=True)  # ["Apple Inc. (Organization)", ...]
    with_conf = format_entities(entities, include_labels=True, include_confidence=True)
    # ["Apple Inc. (entity_type:Organization, conf:0.99)", ...]
    # MISC entities return plain text without metadata: ["iPhone 15"]

    # As a script
    python distilbert-ner.py  # Starts interactive mode
"""

import os

# Suppress transformers warning about missing PyTorch/TensorFlow/Flax
# We only use transformers for the tokenizer, inference is done with ONNX Runtime
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# Configuration
MODEL_PATH = "distilbert-ner-onnx/onnx/model.onnx"
TOKENIZER_PATH = "distilbert-ner-onnx"
MAX_LENGTH = 512  # DistilBERT's max_position_embeddings limit

# HuggingFace model repository
HF_MODEL_REPO = "onnx-community/distilbert-NER-ONNX"
HF_MODEL_FILE = "onnx/model.onnx"

# Label mapping from model config (https://huggingface.co/dslim/distilbert-NER)
# According to CoNLL-2003 BIO tagging scheme used by this model
ID2LABEL = {
    0: "O",  # Outside of a named entity
    1: "B-PER",  # Beginning of a person's name right after another person's name
    2: "I-PER",  # Person's name
    3: "B-ORG",  # Beginning of an organization right after another organization
    4: "I-ORG",  # Organization
    5: "B-LOC",  # Beginning of a location right after another location
    6: "I-LOC",  # Location
    7: "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    8: "I-MISC",  # Miscellaneous entity
}

# Label expansion for human-readable output
LABEL_EXPANSION = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax probabilities from logits"""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


class ModelLoader:
    """Handles ONNX model downloading and loading"""

    @staticmethod
    def ensure_downloaded(model_path: str = MODEL_PATH) -> None:
        """Download the ONNX model from HuggingFace if not present"""
        model_file = Path(model_path)

        if model_file.exists():
            return

        print(f"Model not found at {model_path}")
        print(f"Downloading from HuggingFace ({HF_MODEL_REPO})...")
        print("This will download ~249MB...")

        try:
            from huggingface_hub import hf_hub_download

            model_file.parent.mkdir(parents=True, exist_ok=True)

            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                local_dir=TOKENIZER_PATH,
            )

            print(f"✓ Model downloaded to {model_path}")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download the model. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    @staticmethod
    def load_session(model_path: str = MODEL_PATH) -> ort.InferenceSession:
        """Create an ONNX Runtime inference session"""
        ModelLoader.ensure_downloaded(model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        return session


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizer operations"""

    def __init__(self, tokenizer_path: str = TOKENIZER_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, text: str, max_length: int = MAX_LENGTH) -> dict[str, np.ndarray]:
        """Tokenize input text and return numpy arrays"""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

    def convert_ids_to_tokens(self, input_ids: np.ndarray) -> list[str]:
        """Convert token IDs to token strings"""
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert token list back to readable text"""
        return self.tokenizer.convert_tokens_to_string(tokens)


class EntityExtractor:
    """Extracts named entities from model outputs using BIO tagging"""

    def __init__(self, id2label: dict[int, str] = ID2LABEL):
        self.id2label = id2label

    def extract(
        self,
        tokens: list[str],
        labels: list[str],
        attention_mask: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        label_ids: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Extract entities from BIO-tagged tokens with word-level aggregation.

        Uses 'first' aggregation strategy: groups consecutive tokens into words
        (using ## subword markers), then merges consecutive same-type entities.

        Args:
            tokens: List of token strings
            labels: List of label strings (BIO tags)
            attention_mask: Attention mask array
            probabilities: Optional probability distribution over labels for each token
            label_ids: Optional predicted label IDs for each token
        """
        # Group tokens into words and aggregate labels
        words = self._group_tokens_into_words(
            tokens, labels, attention_mask, probabilities, label_ids
        )

        # Aggregate words into entities
        entities = self._aggregate_words_into_entities(words)

        return entities

    def _group_tokens_into_words(
        self,
        tokens: list[str],
        labels: list[str],
        attention_mask: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        label_ids: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Group subword tokens into complete words"""
        words = []
        current_word = None

        for idx, (token, label, mask) in enumerate(zip(tokens, labels, attention_mask)):
            # Skip padding and special tokens
            if mask == 0 or token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_word:
                    words.append(current_word)
                    current_word = None
                continue

            # Check if this is a subword token
            is_subword = token.startswith("##")

            if is_subword and current_word:
                # Continue the current word
                current_word["tokens"].append(token)
                current_word["labels"].append(label)
                current_word["end_token"] = idx
                if probabilities is not None and label_ids is not None:
                    current_word["confidences"].append(
                        probabilities[idx, label_ids[idx]]
                    )
            else:
                # Start a new word
                if current_word:
                    words.append(current_word)
                word_data = {
                    "tokens": [token],
                    "labels": [label],
                    "start_token": idx,
                    "end_token": idx,
                }
                if probabilities is not None and label_ids is not None:
                    word_data["confidences"] = [probabilities[idx, label_ids[idx]]]
                current_word = word_data

        # Don't forget last word
        if current_word:
            words.append(current_word)

        return words

    def _aggregate_words_into_entities(self, words: list[dict]) -> list[dict]:
        """Aggregate words into named entities using BIO tags"""
        entities = []
        current_entity = None

        for word in words:
            # Use 'first' strategy: take the label of the first token
            first_label = word["labels"][0]

            if first_label == "O":
                # Not an entity - save previous entity if exists
                if current_entity:
                    # Calculate mean confidence before saving
                    if "confidences" in current_entity:
                        current_entity["confidence"] = float(
                            np.mean(current_entity["confidences"])
                        )
                        del current_entity["confidences"]
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Extract entity type (remove B-/I- prefix)
            entity_type = (
                first_label[2:] if first_label.startswith(("B-", "I-")) else first_label
            )

            # Check if we should merge with previous entity
            # According to BIO scheme:
            # - B- prefix = Beginning of NEW entity (do NOT merge)
            # - I- prefix = Inside/continuation of entity (merge with previous)
            should_merge = (
                current_entity
                and current_entity["label"] == entity_type
                and first_label.startswith("I-")
            )

            if should_merge:
                # Extend current entity
                current_entity["end_token"] = word["end_token"]
                current_entity["tokens"].extend(word["tokens"])
                if "confidences" in word:
                    current_entity["confidences"].extend(word["confidences"])
            else:
                # Save previous entity and start new one
                if current_entity:
                    # Calculate mean confidence before saving
                    if "confidences" in current_entity:
                        current_entity["confidence"] = float(
                            np.mean(current_entity["confidences"])
                        )
                        del current_entity["confidences"]  # Remove intermediate list
                    entities.append(current_entity)

                entity_data = {
                    "label": entity_type,
                    "start_token": word["start_token"],
                    "end_token": word["end_token"],
                    "tokens": word["tokens"][:],
                }
                if "confidences" in word:
                    entity_data["confidences"] = word["confidences"][:]
                current_entity = entity_data

        # Don't forget last entity
        if current_entity:
            # Calculate mean confidence for the final entity
            if "confidences" in current_entity:
                current_entity["confidence"] = float(
                    np.mean(current_entity["confidences"])
                )
                del current_entity["confidences"]  # Remove intermediate list
            entities.append(current_entity)

        return entities


# Module-level singletons (lazy initialization)
_model_session: Optional[ort.InferenceSession] = None
_tokenizer: Optional[TokenizerWrapper] = None
_extractor: Optional[EntityExtractor] = None


def _get_model_session() -> ort.InferenceSession:
    """Get or create the ONNX model session (singleton)"""
    global _model_session
    if _model_session is None:
        _model_session = ModelLoader.load_session()
    return _model_session


def _get_tokenizer() -> TokenizerWrapper:
    """Get or create the tokenizer (singleton)"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = TokenizerWrapper()
    return _tokenizer


def _get_extractor() -> EntityExtractor:
    """Get or create the entity extractor (singleton)"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def predict_entities(text: str, max_length: int = MAX_LENGTH, stride: int = 256) -> list[dict]:
    """
    Run NER inference on input text with sliding window for long texts.

    Args:
        text: Input text to analyze
        max_length: Maximum tokens per chunk (default: 512)
        stride: Number of overlapping tokens between chunks (default: 256)

    Returns:
        List of detected entities with their labels and positions.
        Each entity is a dict with keys: 'text', 'label', 'start_token', 'end_token', 'tokens', 'confidence'
    """
    session = _get_model_session()
    tokenizer = _get_tokenizer()
    extractor = _get_extractor()

    # Tokenize with stride to get overlapping chunks
    encoded = tokenizer.tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="np",
    )

    all_entities = []

    # Process each chunk
    num_chunks = len(encoded["input_ids"])
    for chunk_idx in range(num_chunks):
        input_ids = encoded["input_ids"][chunk_idx : chunk_idx + 1].astype(np.int64)
        attention_mask = encoded["attention_mask"][chunk_idx : chunk_idx + 1].astype(np.int64)

        # Run ONNX inference
        outputs = session.run(
            None,  # Get all outputs
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        # Extract logits and convert to label predictions
        logits = outputs[0]  # Shape: (1, max_length, num_labels)
        label_ids = np.argmax(logits[0], axis=-1)  # Shape: (max_length,)

        # Convert logits to probabilities
        probabilities = softmax(logits[0], axis=-1)  # Shape: (max_length, num_labels)

        # Decode tokens and labels
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [ID2LABEL[label_id] for label_id in label_ids]

        # Extract entities from BIO tags with confidence scores
        chunk_entities = extractor.extract(
            tokens, labels, attention_mask[0], probabilities, label_ids
        )

        # Reconstruct entity text using tokenizer
        for entity in chunk_entities:
            entity["text"] = tokenizer.convert_tokens_to_string(entity["tokens"])
            entity["chunk_idx"] = chunk_idx

        all_entities.extend(chunk_entities)

    # Deduplicate entities across chunk boundaries
    if num_chunks > 1:
        all_entities = _deduplicate_chunk_entities(all_entities)

    # Remove chunk metadata
    for entity in all_entities:
        entity.pop("chunk_idx", None)

    return all_entities


def _deduplicate_chunk_entities(entities: list[dict]) -> list[dict]:
    """
    Deduplicate entities from overlapping chunks.

    When processing text with sliding windows, the same entity may appear in multiple
    chunks. This function keeps the entity with the highest confidence score.

    Args:
        entities: List of entity dicts with 'chunk_idx' field

    Returns:
        Deduplicated list with highest-confidence version of each entity
    """
    # Group entities by normalized text and label
    entity_groups = {}

    for entity in entities:
        # Create key based on normalized text and label
        key = (entity["text"].strip().lower(), entity["label"])

        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    # For each group, keep the entity with highest confidence
    deduplicated = []
    for group in entity_groups.values():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Multiple instances - keep the one with highest confidence
            best_entity = max(
                group,
                key=lambda e: e.get("confidence", 0.0)
            )
            deduplicated.append(best_entity)

    return deduplicated


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Deduplicate entities using case-insensitive matching and confidence averaging.

    Args:
        entities: List of entity dicts from predict_entities()

    Returns:
        Deduplicated list where duplicate entities (same text + label, case-insensitive)
        have their confidence scores averaged.
    """
    # Group entities by (lowercase_text, label) key
    entity_groups = {}

    for entity in entities:
        # Create case-insensitive key
        key = (entity["text"].lower(), entity["label"])

        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    # Merge duplicates by averaging confidence
    deduplicated = []
    for group in entity_groups.values():
        # Take first occurrence for base data
        merged = group[0].copy()

        # Average confidence scores if present
        if "confidence" in group[0]:
            confidences = [e["confidence"] for e in group if "confidence" in e]
            if confidences:
                merged["confidence"] = float(np.mean(confidences))

        deduplicated.append(merged)

    return deduplicated


def format_entities(
    entities: list[dict],
    include_labels: bool = False,
    include_confidence: bool = False,
    deduplicate: bool = True
) -> list[str]:
    """
    Format entities as a list of strings.

    Args:
        entities: List of entity dicts from predict_entities()
        include_labels: If True, append label in parentheses like "Microsoft (Organization)"
        include_confidence: If True, append confidence as decimal (requires include_labels=True)
        deduplicate: If True, remove duplicate entities (case-insensitive, averaging confidence)

    Returns:
        List of entity text strings, optionally with labels and confidence scores.
        With confidence enabled, format is: "SpaceX (entity_type:Organization, conf:0.99)"
        MISC entities are always returned as plain text (no metadata) to save tokens.
    """
    # Deduplicate if requested
    if deduplicate:
        entities = _deduplicate_entities(entities)

    result = []
    for entity in entities:
        text = entity["text"]
        label = entity["label"]

        # MISC entities: always return plain text (metadata not useful)
        if label == "MISC":
            result.append(text)
            continue

        # PER/ORG/LOC: add metadata if requested
        if include_labels:
            expanded_label = LABEL_EXPANSION.get(label, label)
            if include_confidence and "confidence" in entity:
                confidence_val = entity["confidence"]
                text = (
                    f"{text} (entity_type:{expanded_label}, conf:{confidence_val:.2f})"
                )
            else:
                text = f"{text} ({expanded_label})"
        result.append(text)
    return result


def main():
    # Load model at startup
    _get_model_session()
    _get_tokenizer()

    try:
        while True:
            text = input("\nEnter text to analyze: ").strip()
            if not text:
                continue

            entities = predict_entities(text)

            # Display results using format_entities
            formatted = format_entities(
                entities, include_labels=True, include_confidence=True
            )
            print(f"\nDetected {len(entities)} entities:")
            print("-" * 60)
            if formatted:
                for entity_str in formatted:
                    print(f"  {entity_str}")
            else:
                print("No entities detected")
            print()

    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
