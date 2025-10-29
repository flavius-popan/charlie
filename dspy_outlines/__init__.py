"""DSPy + Outlines integration."""

import os
from pathlib import Path

_DEFAULT_CACHE_DIR = Path("prompts/.dspy_cache")
_cache_dir = Path(os.environ.get("DSPY_CACHE_DIR", _DEFAULT_CACHE_DIR)).expanduser().resolve()
os.environ["DSPY_CACHE_DIR"] = str(_cache_dir)
_cache_dir.mkdir(parents=True, exist_ok=True)
for shard in range(16):
    (_cache_dir / f"{shard:03d}").mkdir(parents=True, exist_ok=True)

from .adapter import OutlinesAdapter
from .kg_extraction import KGExtractionModule
from .lm import OutlinesLM

__all__ = ["OutlinesLM", "OutlinesAdapter", "KGExtractionModule"]
