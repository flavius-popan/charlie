"""DSPy + Outlines integration."""

from .adapter import OutlinesAdapter
from .kg_extraction import KGExtractionModule
from .lm import OutlinesLM

__all__ = ["OutlinesLM", "OutlinesAdapter", "KGExtractionModule"]
