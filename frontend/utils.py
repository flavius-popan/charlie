"""Utility functions for Charlie TUI frontend."""


def get_display_title(episode: dict, max_chars: int = 50) -> str:
    """Get a short, display-ready title without inspecting full content."""

    source = (episode.get("preview") or episode.get("name") or "Untitled").strip()
    if len(source) <= max_chars:
        return source

    trimmed = source[:max_chars].rstrip()
    if not trimmed.endswith("..."):
        trimmed = trimmed.rstrip(".!? ") + "..."
    return trimmed
