"""Utility functions for Charlie TUI frontend."""


def extract_title(content: str) -> str | None:
    """Extract first # header from markdown content.

    Args:
        content: Markdown text

    Returns:
        Title text without # prefix, or None if no header found

    Examples:
        >>> extract_title("# Hello World\\nContent")
        'Hello World'
        >>> extract_title("No header here")

        >>> extract_title("  # Trimmed  \\n")
        'Trimmed'
    """
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def get_display_title(episode: dict, max_chars: int = 50) -> str:
    """Get title for display in list view.

    Args:
        episode: Episode dict with 'content' and 'name' fields
        max_chars: Maximum characters to display

    Returns:
        Title string, truncated to max_chars

    Examples:
        >>> get_display_title({'content': '# My Title\\nBody', 'name': 'default'})
        'My Title'
        >>> get_display_title({'content': 'Plain text', 'name': 'default'})
        'Plain text'
    """
    content = episode.get("content", "")

    title = extract_title(content)
    if title:
        return title[:max_chars]

    preview = content.replace("\n", " ").strip()
    return preview[:max_chars] if preview else episode.get("name", "Untitled")
