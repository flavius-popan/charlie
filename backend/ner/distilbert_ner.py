#!/usr/bin/env python3
"""
DistilBERT NER - ONNX Inference Module

Lightweight NER inference using ONNX Runtime (no PyTorch required).
Dependencies: onnxruntime (~10MB), transformers (tokenizer only), numpy

Uses distilbert-base-uncased fine-tuned for NER - detects entities regardless of
capitalization while preserving the original text casing in results.

Auto-downloads the ONNX model from HuggingFace if not present (~255MB).
Tokenizer files are included in the repository.

Usage:
    from backend.ner import predict_entities, format_entities

    # Works with any capitalization, preserves original casing
    entities = predict_entities("charlie was a cool guy")
    texts = format_entities(entities)  # ["charlie"] - lowercase preserved

    entities = predict_entities("Charlie was a cool guy")
    texts = format_entities(entities)  # ["Charlie"] - proper case preserved

    labeled = format_entities(entities, include_labels=True)  # ["Charlie [PER]", ...]
    with_conf = format_entities(entities, include_labels=True, include_confidence=True)
    # ["Charlie [PER:99%]", ...]
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


# Configuration - paths relative to this module
_MODULE_DIR = Path(__file__).parent
MODEL_PATH = _MODULE_DIR / "distilbert-ner-uncased-onnx" / "onnx" / "model.onnx"
TOKENIZER_PATH = _MODULE_DIR / "distilbert-ner-uncased-onnx"
MAX_LENGTH = 512  # DistilBERT's max_position_embeddings limit

# HuggingFace model repository
HF_MODEL_REPO = "andi611/distilbert-base-uncased-ner-conll2003"
HF_MODEL_FILE = "onnx/model.onnx"
HF_REVISION = "refs/pr/4"  # PR with ONNX export

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


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax probabilities from logits"""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


class ModelLoader:
    """Handles ONNX model downloading and loading"""

    @staticmethod
    def ensure_downloaded(model_path: Path = MODEL_PATH) -> None:
        """Download the ONNX model from HuggingFace if not present"""
        if model_path.exists():
            return

        print(f"Model not found at {model_path}")
        print(f"Downloading from HuggingFace ({HF_MODEL_REPO})...")
        print("This will download ~255MB...")

        try:
            from huggingface_hub import hf_hub_download

            model_path.parent.mkdir(parents=True, exist_ok=True)

            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                local_dir=str(TOKENIZER_PATH),
                revision=HF_REVISION,
            )

            print(f"Model downloaded to {model_path}")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download the model. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    @staticmethod
    def load_session(model_path: Path = MODEL_PATH) -> ort.InferenceSession:
        """Create an ONNX Runtime inference session"""
        ModelLoader.ensure_downloaded(model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        return session


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizer operations"""

    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

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
        offset_mapping: Optional[list[tuple[int, int]]] = None,
        original_text: str | None = None,
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
            offset_mapping: Optional list of (start_char, end_char) tuples for each token
            original_text: Original input text (used to detect sentence boundaries)
        """
        words = self._group_tokens_into_words(
            tokens, labels, attention_mask, probabilities, label_ids
        )
        entities = self._aggregate_words_into_entities(words)
        entities = self._merge_fragmented_entities(
            entities, tokens, original_text, offset_mapping
        )

        if offset_mapping:
            for entity in entities:
                start_token = entity["start_token"]
                end_token = entity["end_token"]
                start_char = offset_mapping[start_token][0]
                end_char = offset_mapping[end_token][1]
                entity["start_char"] = start_char
                entity["end_char"] = end_char

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
            if mask == 0 or token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_word:
                    words.append(current_word)
                    current_word = None
                continue

            is_subword = token.startswith("##")

            if is_subword and current_word:
                current_word["tokens"].append(token)
                current_word["labels"].append(label)
                current_word["end_token"] = idx
                if probabilities is not None and label_ids is not None:
                    current_word["confidences"].append(
                        probabilities[idx, label_ids[idx]]
                    )
            else:
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

        if current_word:
            words.append(current_word)

        return words

    def _aggregate_words_into_entities(self, words: list[dict]) -> list[dict]:
        """Aggregate words into named entities using BIO tags"""
        entities = []
        current_entity = None

        for word in words:
            first_label = word["labels"][0]

            if first_label == "O":
                if current_entity:
                    if "confidences" in current_entity:
                        current_entity["confidence"] = float(
                            np.mean(current_entity["confidences"])
                        )
                        del current_entity["confidences"]
                    entities.append(current_entity)
                    current_entity = None
                continue

            entity_type = (
                first_label[2:] if first_label.startswith(("B-", "I-")) else first_label
            )

            # BIO scheme: B- = new entity, I- = continuation
            should_merge = (
                current_entity
                and current_entity["label"] == entity_type
                and first_label.startswith("I-")
            )

            if should_merge:
                current_entity["end_token"] = word["end_token"]
                current_entity["tokens"].extend(word["tokens"])
                if "confidences" in word:
                    current_entity["confidences"].extend(word["confidences"])
            else:
                if current_entity:
                    if "confidences" in current_entity:
                        current_entity["confidence"] = float(
                            np.mean(current_entity["confidences"])
                        )
                        del current_entity["confidences"]
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

        if current_entity:
            if "confidences" in current_entity:
                current_entity["confidence"] = float(
                    np.mean(current_entity["confidences"])
                )
                del current_entity["confidences"]
            entities.append(current_entity)

        return entities

    def _merge_fragmented_entities(
        self,
        entities: list[dict],
        tokens: list[str],
        original_text: str | None = None,
        offset_mapping: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """
        Merge consecutive entities of the same type that are fragmented by punctuation.

        This fixes cases like "G.I. Joe" being split into ["G", ".", "I. Joe"] as
        three separate PER entities. When consecutive same-type entities are separated
        by only punctuation tokens, they should be merged into one entity.

        IMPORTANT: Only merges when there's clear evidence of fragmentation:
        - Entities contain very short tokens (1-2 chars like "G", ".", "I")
        - OR entities are separated by only periods/dots

        IMPORTANT: Does NOT merge across sentence boundaries:
        - If original text between entities contains newlines -> don't merge
        - This prevents "Riley. \n\nXANDER" from merging while allowing "G.I. Joe"

        Args:
            entities: List of entity dicts from _aggregate_words_into_entities()
            tokens: Original token list to check what's between entities
            original_text: Original input text (used to detect newlines/sentence boundaries)
            offset_mapping: Token-to-character offset mapping (needed to find gap text)

        Returns:
            List of entities with fragmented same-type entities merged
        """
        if len(entities) <= 1:
            return entities

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]
            j = i + 1

            while j < len(entities):
                next_entity = entities[j]
                gap = next_entity["start_token"] - current["end_token"]
                if gap > 2:
                    break

                should_merge = False
                current_has_periods = "." in "".join(current["tokens"])

                if current_has_periods and gap <= 1:
                    should_merge = True
                elif current["label"] == next_entity["label"]:
                    current_has_short_tokens = any(
                        len(token.strip("##")) <= 2 for token in current["tokens"]
                    )
                    if current_has_short_tokens:
                        should_merge = True

                if gap > 0:
                    gap_start = current["end_token"] + 1
                    gap_end = next_entity["start_token"]
                    gap_tokens = [
                        t
                        for t in tokens[gap_start:gap_end]
                        if t not in ["[CLS]", "[SEP]", "[PAD]"]
                    ]

                    if gap_tokens:
                        has_period_gap = all(
                            token.strip("##") == "." for token in gap_tokens
                        )

                        if has_period_gap:
                            has_sentence_boundary = False

                            if original_text and offset_mapping:
                                current_end_char = offset_mapping[current["end_token"]][
                                    1
                                ]
                                next_start_char = offset_mapping[
                                    next_entity["start_token"]
                                ][0]
                                gap_text = original_text[
                                    current_end_char:next_start_char
                                ]

                                if "\n" in gap_text:
                                    has_sentence_boundary = True
                                elif next_start_char < len(original_text):
                                    next_char = original_text[next_start_char]
                                    if next_char.isupper() and ". " in gap_text:
                                        has_sentence_boundary = True

                            if not has_sentence_boundary:
                                should_merge = True

                if not should_merge:
                    break

                current["end_token"] = next_entity["end_token"]
                current["tokens"].extend(next_entity["tokens"])

                # Prefer non-MISC labels when merging across types
                if current["label"] != next_entity["label"]:
                    if current["label"] == "MISC":
                        current["label"] = next_entity["label"]

                if "confidence" in current and "confidence" in next_entity:
                    current["confidence"] = (
                        current["confidence"] + next_entity["confidence"]
                    ) / 2

                j += 1

            i = j
            merged.append(current)

        return merged


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


def predict_entities(
    text: str, max_length: int = MAX_LENGTH, stride: int = 256
) -> list[dict]:
    """
    Run NER inference on input text with sliding window for long texts.

    Args:
        text: Input text to analyze
        max_length: Maximum tokens per chunk (default: 512)
        stride: Number of overlapping tokens between chunks (default: 256)

    Returns:
        List of detected entities with their labels and positions.
        Each entity is a dict with keys: 'text', 'label', 'start_char', 'end_char',
        'start_token', 'end_token', 'tokens', 'confidence'
        The 'text' field preserves the original capitalization from the input.
    """
    session = _get_model_session()
    tokenizer = _get_tokenizer()
    extractor = _get_extractor()

    # Tokenize with stride to get overlapping chunks and character offsets
    encoded = tokenizer.tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )

    all_entities = []
    num_chunks = len(encoded["input_ids"])

    for chunk_idx in range(num_chunks):
        input_ids = encoded["input_ids"][chunk_idx : chunk_idx + 1].astype(np.int64)
        attention_mask = encoded["attention_mask"][chunk_idx : chunk_idx + 1].astype(
            np.int64
        )
        offset_mapping = encoded["offset_mapping"][chunk_idx].tolist()

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        logits = outputs[0]
        label_ids = np.argmax(logits[0], axis=-1)
        probabilities = softmax(logits[0], axis=-1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [ID2LABEL[label_id] for label_id in label_ids]

        chunk_entities = extractor.extract(
            tokens,
            labels,
            attention_mask[0],
            probabilities,
            label_ids,
            offset_mapping,
            text,
        )

        for entity in chunk_entities:
            if "start_char" in entity and "end_char" in entity:
                entity["text"] = text[entity["start_char"] : entity["end_char"]]
            else:
                entity["text"] = tokenizer.convert_tokens_to_string(entity["tokens"])
            entity["chunk_idx"] = chunk_idx

        all_entities.extend(chunk_entities)

    if num_chunks > 1:
        all_entities = _deduplicate_chunk_entities(all_entities)

    for entity in all_entities:
        entity.pop("chunk_idx", None)

    filtered = []
    for e in all_entities:
        text = e["text"].strip()
        if len(text) <= 1:
            continue
        if text.startswith("."):
            continue
        filtered.append(e)

    return filtered


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
    entity_groups: dict[tuple[str, str], list[dict]] = {}

    for entity in entities:
        key = (entity["text"].strip().lower(), entity["label"])
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    deduplicated = []
    for group in entity_groups.values():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            best_entity = max(group, key=lambda e: e.get("confidence", 0.0))
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
    entity_groups: dict[tuple[str, str], list[dict]] = {}

    for entity in entities:
        key = (entity["text"].lower(), entity["label"])
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    deduplicated = []
    for group in entity_groups.values():
        merged = group[0].copy()
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
    deduplicate: bool = True,
) -> list[str]:
    """
    Format entities as a list of strings.

    Args:
        entities: List of entity dicts from predict_entities()
        include_labels: If True, append label in brackets like "Microsoft [ORG]"
        include_confidence: If True, append confidence as percentage (requires include_labels=True)
        deduplicate: If True, remove duplicate entities (case-insensitive, averaging confidence)

    Returns:
        List of entity text strings, optionally with labels and confidence scores.
        Format examples:
        - Plain: "Microsoft"
        - With labels: "Microsoft [ORG]", "iPhone 15 [MISC]"
        - With labels+confidence: "Microsoft [ORG:99%]", "iPhone 15 [MISC:85%]"
    """
    if deduplicate:
        entities = _deduplicate_entities(entities)

    result = []
    for entity in entities:
        text = entity["text"]
        label = entity["label"]

        if include_labels:
            if include_confidence and "confidence" in entity:
                confidence_pct = int(entity["confidence"] * 100)
                text = f"{text} [{label}:{confidence_pct}%]"
            else:
                text = f"{text} [{label}]"
        result.append(text)
    return result


__all__ = [
    "predict_entities",
    "format_entities",
    "ID2LABEL",
    "MAX_LENGTH",
]
