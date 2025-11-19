"""Backend configuration for Charlie."""

from __future__ import annotations

from pathlib import Path

DB_PATH = Path("data/charlie.db")
DEFAULT_JOURNAL = "default"

# TCP server for debugging (disabled by default)
ENABLE_TCP_SERVER = False
TCP_HOST = "127.0.0.1"
TCP_PORT = 6379
TCP_PASSWORD = None
