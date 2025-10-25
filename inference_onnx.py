#!/usr/bin/env python3
"""
ONNX Inference for DistilBERT-NER
Lightweight Python inference using ONNX Runtime (no PyTorch required).
Dependencies: onnxruntime (~10MB), transformers (tokenizer only), numpy
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# Configuration
MODEL_PATH = "distilbert-ner-onnx/model.onnx"
TOKENIZER_PATH = "distilbert-ner-onnx"
MAX_LENGTH = 128

# Label mapping from model config (https://huggingface.co/dslim/distilbert-NER)
ID2LABEL = {
    0: "O",       # Outside any entity
    1: "B-PER",   # Beginning of person name
    2: "I-PER",   # Inside person name
    3: "B-ORG",   # Beginning of organization
    4: "I-ORG",   # Inside organization
    5: "B-LOC",   # Beginning of location
    6: "I-LOC",   # Inside location
    7: "B-MISC",  # Beginning of miscellaneous entity
    8: "I-MISC",  # Inside miscellaneous entity
}


class ONNXNERInference:
    """Lightweight NER inference using ONNX Runtime"""

    def __init__(self, model_path: str, tokenizer_path: str):
        print("Loading ONNX model...")
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]  # CPU only, very portable
        )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        print(f"✓ Model and tokenizer loaded")
        print(f"  - Runtime: ONNX Runtime (CPU)")
        print(f"  - Input names: {[inp.name for inp in self.session.get_inputs()]}")
        print(f"  - Output names: {[out.name for out in self.session.get_outputs()]}\n")

    def predict(self, text: str) -> list[dict]:
        """
        Run NER inference on input text

        Args:
            text: Input text to analyze

        Returns:
            List of detected entities with their labels and positions
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="np",  # Return numpy arrays
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Run ONNX inference
        outputs = self.session.run(
            None,  # Get all outputs
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        # Extract logits and convert to label predictions
        logits = outputs[0]  # Shape: (1, MAX_LENGTH, num_labels)
        label_ids = np.argmax(logits[0], axis=-1)  # Shape: (MAX_LENGTH,)

        # Decode tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [ID2LABEL[label_id] for label_id in label_ids]

        # Debug: Print first 20 token-label pairs (uncomment to debug)
        # print("\nToken-Label pairs (first 20):")
        # for i, (tok, lab) in enumerate(zip(tokens[:20], labels[:20])):
        #     if attention_mask[0][i] == 1:
        #         print(f"  {i:3d}: {tok:15s} -> {lab}")

        # Extract entities from BIO tags
        entities = self._extract_entities(tokens, labels, attention_mask[0])

        return entities

    def _extract_entities(self, tokens: list[str], labels: list[str], attention_mask: np.ndarray) -> list[dict]:
        """
        Extract entities from BIO-tagged tokens with word-level aggregation.

        Uses a 'first' aggregation strategy similar to HuggingFace transformers:
        Groups consecutive tokens into words (using ## subword markers), then
        merges consecutive same-type entities that are part of the same word.

        Args:
            tokens: List of tokenized words
            labels: List of BIO labels for each token
            attention_mask: Mask indicating real tokens vs padding

        Returns:
            List of entities with text, label, and token positions
        """
        # First, group tokens into words and aggregate labels
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

        # Now aggregate words into entities
        # Use 'first' strategy: take the label of the first token in each word
        entities = []
        current_entity = None

        for word in words:
            # Get the entity label from the first token (first strategy)
            first_label = word["labels"][0]

            if first_label == "O":
                # Not an entity - save previous entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Extract entity type (remove B-/I- prefix)
            entity_type = first_label[2:] if first_label.startswith(("B-", "I-")) else first_label

            # Check if we should merge with previous entity
            # Merge if: previous exists, same type, and current is continuation (I-) OR same type subword (B-)
            should_merge = (
                current_entity
                and current_entity["label"] == entity_type
                and (first_label.startswith("I-") or all(l.startswith(f"B-{entity_type}") or l.startswith(f"I-{entity_type}") for l in word["labels"]))
            )

            if should_merge:
                # Extend current entity
                current_entity["end_token"] = word["end_token"]
            else:
                # Save previous entity and start new one
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": entity_type,
                    "start_token": word["start_token"],
                    "end_token": word["end_token"],
                }

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

        # Reconstruct entity text using tokenizer
        for entity in entities:
            entity_tokens = tokens[entity["start_token"]:entity["end_token"] + 1]
            entity["text"] = self.tokenizer.convert_tokens_to_string(entity_tokens)

        return entities

    def print_results(self, text: str, entities: list[dict]):
        """Pretty print the inference results"""
        print(f"Input: {text}")
        print(f"\nDetected {len(entities)} entities:")
        print("-" * 60)

        if not entities:
            print("No entities detected")
        else:
            for entity in entities:
                print(f"  [{entity['label']:4s}] {entity['text']}")

        print()


def main():
    """Run example inferences"""
    print("=" * 60)
    print("ONNX DistilBERT-NER Inference Demo")
    print("=" * 60)
    print()

    # Initialize inference
    ner = ONNXNERInference(MODEL_PATH, TOKENIZER_PATH)

    # Example 1: Standard entities
    print("=" * 60)
    print("Example 1: Standard Entities")
    print("=" * 60)
    text1 = "Apple Inc. is located in Cupertino, California."
    entities1 = ner.predict(text1)
    ner.print_results(text1, entities1)

    # Example 2: Novel names (not in vocabulary)
    print("=" * 60)
    print("Example 2: Novel Names (Testing Subword Handling)")
    print("=" * 60)
    text2 = "Xylophus Thrandor works at Quantum Synergetics in Neo Tokyo."
    entities2 = ner.predict(text2)
    ner.print_results(text2, entities2)

    # Example 3: Mixed entities
    print("=" * 60)
    print("Example 3: Multiple Entity Types")
    print("=" * 60)
    text3 = "Microsoft CEO Satya Nadella announced a partnership with OpenAI in Seattle."
    entities3 = ner.predict(text3)
    ner.print_results(text3, entities3)

    # Example 4: Interactive mode
    print("=" * 60)
    print("Interactive Mode (Ctrl+C to exit)")
    print("=" * 60)

    try:
        while True:
            text = input("\nEnter text to analyze: ").strip()
            if not text:
                continue

            entities = ner.predict(text)
            ner.print_results(text, entities)

    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
