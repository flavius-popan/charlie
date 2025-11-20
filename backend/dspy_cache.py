"""Centralized DSPy cache location for the backend."""

from __future__ import annotations

import os
from pathlib import Path


def set_dspy_cache_dir() -> Path:
    """Force DSPy disk cache into backend/prompts/.dspy_cache."""
    cache_dir = Path(__file__).resolve().parent / "prompts" / ".dspy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for env in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
        os.environ[env] = str(cache_dir)
    return cache_dir


# Execute on import so any caller that imports backend.dspy_cache gets the path set
CACHE_DIR = set_dspy_cache_dir()
