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

# Label mapping from model training
ID2LABEL = {
    0: "O",       # Outside any entity
    1: "B-MISC",  # Beginning of miscellaneous entity
    2: "I-MISC",  # Inside miscellaneous entity
    3: "B-PER",   # Beginning of person name
    4: "I-PER",   # Inside person name
    5: "B-ORG",   # Beginning of organization
    6: "I-ORG",   # Inside organization
    7: "B-LOC",   # Beginning of location
    8: "I-LOC",   # Inside location
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

        # Extract entities from BIO tags
        entities = self._extract_entities(tokens, labels, attention_mask[0])

        return entities

    def _extract_entities(self, tokens: list[str], labels: list[str], attention_mask: np.ndarray) -> list[dict]:
        """
        Extract entities from BIO-tagged tokens

        Args:
            tokens: List of tokenized words
            labels: List of BIO labels for each token
            attention_mask: Mask indicating real tokens vs padding

        Returns:
            List of entities with text, label, and token positions
        """
        entities = []
        current_entity = None

        for idx, (token, label, mask) in enumerate(zip(tokens, labels, attention_mask)):
            # Skip padding tokens and special tokens
            if mask == 0 or token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Handle BIO tagging
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "text": token.replace("##", ""),  # Remove subword marker
                    "label": entity_type,
                    "start_token": idx,
                    "end_token": idx,
                }

            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    # Append to current entity text
                    token_text = token.replace("##", "")
                    current_entity["text"] += token_text
                    current_entity["end_token"] = idx

            else:  # "O" label
                # Outside any entity - save previous if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

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
