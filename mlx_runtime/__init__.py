"""Shared MLX runtime utilities."""

from __future__ import annotations

import threading
from importlib import import_module

from .loader import load_mlx_model

# MLX/Metal is not thread-safe; reuse a single lock everywhere.
MLX_LOCK = threading.Lock()

__all__ = ["MLX_LOCK", "MLXDspyLM", "load_mlx_model"]


def __getattr__(name: str):
    if name == "MLXDspyLM":
        return import_module(".dspy_lm", __name__).MLXDspyLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
