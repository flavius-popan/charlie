"""Frontend configuration for Charlie."""

from __future__ import annotations

from pathlib import Path

DEFAULT_THEME = "catppuccin-mocha"

# Theme is stored in a file (not Redis) because it must be available immediately
# on app startup, before the database is initialized. Reading from a file is
# synchronous and fast, preventing the UI from flashing between themes.
# Use absolute path anchored to project root, not current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
THEME_FILE = _PROJECT_ROOT / "data" / "theme.txt"


def get_theme() -> str:
    """Load theme from file, or return default if anything goes wrong."""
    try:
        theme = THEME_FILE.read_text().strip()
        return theme if theme else DEFAULT_THEME
    except Exception:
        return DEFAULT_THEME


def set_theme(theme: str) -> None:
    """Save theme to file. Never raises - logs errors silently."""
    try:
        THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
        THEME_FILE.write_text(theme)
    except Exception:
        pass  # Theme persistence is non-critical, never crash
