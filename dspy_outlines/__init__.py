"""DSPy + Outlines integration."""

from .adapter import OutlinesAdapter
from .schema_extractor import extract_output_schema
from .lm import OutlinesLM

__all__ = ["OutlinesLM", "OutlinesAdapter", "extract_output_schema"]
