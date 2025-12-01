"""Shared inference runtime utilities."""

from __future__ import annotations

from .dspy_lm import DspyLM
from .loader import load_model

__all__ = ["DspyLM", "load_model"]
