"""Ensure DSPy environment variables are configured before importing DSPy.

Prefer `settings.py` if available; otherwise fall back to pipeline-local cache.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import settings as _settings  # noqa: F401
except ImportError:
    # Minimal fallback so DSPy never defaults to ~/.dspy_cache.
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = prompts_dir / ".dspy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for env_var in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
        os.environ.setdefault(env_var, str(cache_dir))
