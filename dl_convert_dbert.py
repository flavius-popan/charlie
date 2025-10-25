#!/usr/bin/env python3
"""
Convert DistilBERT-NER to Core ML
Downloads the model from HuggingFace and converts it to CoreML format.
Requirements: pip install transformers torch coremltools numpy
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import coremltools as ct

# Configuration
MODEL_NAME = "dslim/distilbert-NER"
OUTPUT_DIR = "distilbert-ner-coreml"
MAX_LENGTH = 128


class ModelWrapper(torch.nn.Module):
    """Wrapper to extract logits from model output dict"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def download_and_convert():
    """Download model from HuggingFace and convert to Core ML"""
    print("=" * 60)
    print("Step 1: Downloading model from Hugging Face...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"✓ Model downloaded: {MODEL_NAME}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  - Labels: {model.config.num_labels}")

    print("\n" + "=" * 60)
    print("Step 2: Converting to Core ML...")
    print("=" * 60)

    # Create example input
    example_text = "Apple Inc. is located in Cupertino, California."
    example_input = tokenizer(
        example_text,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    # Wrap model to return only logits (not dict)
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # Trace the wrapped model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model, (example_input["input_ids"], example_input["attention_mask"])
        )

    # Convert to Core ML
    # Note: Using neuralnetwork format due to Python 3.13 compatibility issues with mlprogram
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_LENGTH), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_LENGTH), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        convert_to="neuralnetwork",  # Using neuralnetwork format for Python 3.13 compatibility
    )

    # Save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "DistilBERT_NER.mlmodel")
    coreml_model.save(model_path)

    # Save tokenizer for later use
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Get number of labels from model config
    num_labels = model.config.num_labels

    print(f"✓ Model converted and saved to: {model_path}")
    print("  - Format: Neural Network (compatible with Python 3.13)")
    print(f"  - Input: input_ids [{MAX_LENGTH}], attention_mask [{MAX_LENGTH}]")
    print(f"  - Output: logits [{MAX_LENGTH}, {num_labels}]")
    print(f"\nModel size: ~260MB")

    return model_path, tokenizer


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("DistilBERT-NER Model Download & Conversion")
        print("=" * 60)

        # Download and convert
        model_path, tokenizer = download_and_convert()

        print("\n" + "=" * 60)
        print("✓ Conversion complete!")
        print("=" * 60)
        print(f"\nModel saved at: {model_path}")
        print(f"Tokenizer saved at: {OUTPUT_DIR}/")
        print("\nReady for inference with PyTorch or CoreML.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
