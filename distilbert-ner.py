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

    # As a script
    python distilbert-ner.py  # Starts interactive mode
"""

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# Configuration
MODEL_PATH = "distilbert-ner-onnx/onnx/model.onnx"
TOKENIZER_PATH = "distilbert-ner-onnx"
MAX_LENGTH = 128

# HuggingFace model repository
HF_MODEL_REPO = "onnx-community/distilbert-NER-ONNX"
HF_MODEL_FILE = "onnx/model.onnx"

# Label mapping from model config (https://huggingface.co/dslim/distilbert-NER)
ID2LABEL = {
    0: "O",  # Outside any entity
    1: "B-PER",  # Beginning of person name
    2: "I-PER",  # Inside person name
    3: "B-ORG",  # Beginning of organization
    4: "I-ORG",  # Inside organization
    5: "B-LOC",  # Beginning of location
    6: "I-LOC",  # Inside location
    7: "B-MISC",  # Beginning of miscellaneous entity
    8: "I-MISC",  # Inside miscellaneous entity
}

# Label expansion for human-readable output
LABEL_EXPANSION = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}


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
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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
    ) -> list[dict]:
        """
        Extract entities from BIO-tagged tokens with word-level aggregation.

        Uses 'first' aggregation strategy: groups consecutive tokens into words
        (using ## subword markers), then merges consecutive same-type entities.
        """
        # Group tokens into words and aggregate labels
        words = self._group_tokens_into_words(tokens, labels, attention_mask)

        # Aggregate words into entities
        entities = self._aggregate_words_into_entities(words)

        return entities

    def _group_tokens_into_words(
        self,
        tokens: list[str],
        labels: list[str],
        attention_mask: np.ndarray,
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
            else:
                # Start a new word
                if current_word:
                    words.append(current_word)
                current_word = {
                    "tokens": [token],
                    "labels": [label],
                    "start_token": idx,
                    "end_token": idx,
                }

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
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Extract entity type (remove B-/I- prefix)
            entity_type = (
                first_label[2:] if first_label.startswith(("B-", "I-")) else first_label
            )

            # Check if we should merge with previous entity
            should_merge = (
                current_entity
                and current_entity["label"] == entity_type
                and (
                    first_label.startswith("I-")
                    or all(
                        l.startswith(f"B-{entity_type}") or l.startswith(f"I-{entity_type}")
                        for l in word["labels"]
                    )
                )
            )

            if should_merge:
                # Extend current entity
                current_entity["end_token"] = word["end_token"]
                current_entity["tokens"].extend(word["tokens"])
            else:
                # Save previous entity and start new one
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": entity_type,
                    "start_token": word["start_token"],
                    "end_token": word["end_token"],
                    "tokens": word["tokens"][:],
                }

        # Don't forget last entity
        if current_entity:
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
        print("Loading ONNX model...")
        _model_session = ModelLoader.load_session()
        print("✓ Model loaded")
    return _model_session


def _get_tokenizer() -> TokenizerWrapper:
    """Get or create the tokenizer (singleton)"""
    global _tokenizer
    if _tokenizer is None:
        print("Loading tokenizer...")
        _tokenizer = TokenizerWrapper()
        print("✓ Tokenizer loaded")
    return _tokenizer


def _get_extractor() -> EntityExtractor:
    """Get or create the entity extractor (singleton)"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def predict_entities(text: str) -> list[dict]:
    """
    Run NER inference on input text.

    Args:
        text: Input text to analyze

    Returns:
        List of detected entities with their labels and positions.
        Each entity is a dict with keys: 'text', 'label', 'start_token', 'end_token', 'tokens'
    """
    session = _get_model_session()
    tokenizer = _get_tokenizer()
    extractor = _get_extractor()

    # Tokenize input
    encoded = tokenizer.encode(text)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Run ONNX inference
    outputs = session.run(
        None,  # Get all outputs
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    # Extract logits and convert to label predictions
    logits = outputs[0]  # Shape: (1, MAX_LENGTH, num_labels)
    label_ids = np.argmax(logits[0], axis=-1)  # Shape: (MAX_LENGTH,)

    # Decode tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [ID2LABEL[label_id] for label_id in label_ids]

    # Extract entities from BIO tags
    entities = extractor.extract(tokens, labels, attention_mask[0])

    # Reconstruct entity text using tokenizer
    for entity in entities:
        entity["text"] = tokenizer.convert_tokens_to_string(entity["tokens"])

    return entities


def format_entities(entities: list[dict], include_labels: bool = False) -> list[str]:
    """
    Format entities as a list of strings.

    Args:
        entities: List of entity dicts from predict_entities()
        include_labels: If True, append label in parentheses like "Microsoft (Organization)"

    Returns:
        List of entity text strings, optionally with labels
    """
    result = []
    for entity in entities:
        text = entity["text"]
        if include_labels:
            label = entity["label"]
            expanded_label = LABEL_EXPANSION.get(label, label)
            text = f"{text} ({expanded_label})"
        result.append(text)
    return result


def print_results(text: str, entities: list[dict]) -> None:
    """Pretty print the inference results"""
    print(f"Input: {text}")
    print(f"\nDetected {len(entities)} entities:")
    print("-" * 60)

    if not entities:
        print("No entities detected")
    else:
        for entity in entities:
            label = LABEL_EXPANSION.get(entity["label"], entity["label"])
            print(f"  [{label}] {entity['text']}")

    print()


def main():
    """Run example inferences and enter interactive mode"""
    print("=" * 60)
    print("ONNX DistilBERT-NER Inference Demo")
    print("=" * 60)
    print()

    # Example 1: Standard entities
    print("=" * 60)
    print("Example 1: Standard Entities")
    print("=" * 60)
    text1 = "Apple Inc. is located in Cupertino, California."
    entities1 = predict_entities(text1)
    print_results(text1, entities1)

    # Example 2: Novel names (not in vocabulary)
    print("=" * 60)
    print("Example 2: Novel Names (Testing Subword Handling)")
    print("=" * 60)
    text2 = "Xylophus Thrandor works at Quantum Synergetics in Neo Tokyo."
    entities2 = predict_entities(text2)
    print_results(text2, entities2)

    # Example 3: Mixed entities
    print("=" * 60)
    print("Example 3: Multiple Entity Types")
    print("=" * 60)
    text3 = "Microsoft CEO Satya Nadella announced a partnership with OpenAI in Seattle."
    entities3 = predict_entities(text3)
    print_results(text3, entities3)

    # Interactive mode
    print("=" * 60)
    print("Interactive Mode (Ctrl+C to exit)")
    print("=" * 60)

    try:
        while True:
            text = input("\nEnter text to analyze: ").strip()
            if not text:
                continue

            entities = predict_entities(text)
            print_results(text, entities)

    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
