#!/usr/bin/env python3
"""
Convert DistilBERT-NER to ONNX
Downloads the model from HuggingFace and converts it to ONNX format.
ONNX Runtime is lightweight (~10MB) and cross-platform.
"""

import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configuration
MODEL_NAME = "dslim/distilbert-NER"
OUTPUT_DIR = "distilbert-ner-onnx"
MAX_LENGTH = 128


def download_and_convert():
    """Download model from HuggingFace and convert to ONNX"""
    print("=" * 60)
    print("Step 1: Downloading model from Hugging Face...")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"✓ Model downloaded: {MODEL_NAME}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  - Labels: {model.config.num_labels}")

    print("\n" + "=" * 60)
    print("Step 2: Converting to ONNX...")
    print("=" * 60)

    # Create example input for tracing
    example_text = "Apple Inc. is located in Cupertino, California."
    example_input = tokenizer(
        example_text,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    # Export to ONNX
    model_path = Path(OUTPUT_DIR) / "model.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example_input["input_ids"], example_input["attention_mask"]),
            str(model_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Tokenizer saved to: {OUTPUT_DIR}/")

    # Get model info
    file_size = model_path.stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"\nModel size: ~{file_size:.1f}MB")
    print(f"Labels: {model.config.num_labels}")

    return model_path, tokenizer


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("DistilBERT-NER Model Download & ONNX Conversion")
        print("=" * 60)

        # Download and convert
        model_path, tokenizer = download_and_convert()

        print("\n" + "=" * 60)
        print("✓ Conversion complete!")
        print("=" * 60)
        print(f"\nModel saved at: {model_path}")
        print(f"Tokenizer saved at: {OUTPUT_DIR}/")
        print("\nReady for lightweight inference with ONNX Runtime.")
        print("No PyTorch required for inference!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
