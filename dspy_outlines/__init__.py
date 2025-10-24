"""DSPy + Outlines integration."""

from .base_lm import PassthroughLM
from .schema_extractor import extract_output_schema
from .hybrid_lm import OutlinesDSPyLM

__all__ = ["PassthroughLM", "extract_output_schema", "OutlinesDSPyLM"]
