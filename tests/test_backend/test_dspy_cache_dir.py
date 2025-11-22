from __future__ import annotations

import importlib
import os
from pathlib import Path

import backend.dspy_cache as dspy_cache


def test_dspy_cache_dir_is_forced(monkeypatch):
    expected = (Path(__file__).resolve().parents[2] / "backend" / "prompts" / ".dspy_cache").resolve()

    # Simulate wrong env vars, then reload cache module to enforce the path.
    for key in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
        monkeypatch.setenv(key, "/tmp/not-the-cache")

    importlib.reload(dspy_cache)

    # Reload dspy.clients so it re-reads env vars for DISK_CACHE_DIR.
    import sys

    sys.modules.pop("dspy.clients", None)
    import dspy.clients as clients  # type: ignore

    assert all(os.environ[key] == str(expected) for key in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"))
    assert Path(clients.DISK_CACHE_DIR).resolve() == expected
    assert expected.is_dir()
