"""Hyphenated CLI entrypoint wrapper for Graphiti agents."""
from __future__ import annotations

import sys

from graphiti_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
