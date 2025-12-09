"""NER extraction module using DistilBERT ONNX."""

from backend.ner.distilbert_ner import predict_entities, format_entities

__all__ = ["predict_entities", "format_entities"]
