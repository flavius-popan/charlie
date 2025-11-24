"""Tests for Charlie TUI utility functions."""

import pytest

from frontend.utils import extract_title, get_display_title


class TestExtractTitle:
    """Tests for extract_title function."""

    def test_extract_title_with_header(self):
        """Should extract title from markdown header."""
        content = "# My Title\nContent here"
        assert extract_title(content) == "My Title"

    def test_extract_title_no_header(self):
        """Should return None when no header present."""
        content = "Plain text without header"
        assert extract_title(content) is None

    def test_extract_title_trimmed(self):
        """Should trim whitespace from extracted title."""
        content = "  # Trimmed Title  \nContent"
        assert extract_title(content) == "Trimmed Title"

    def test_extract_title_empty_header(self):
        """Should return None for header without space after #."""
        content = "#\nContent"
        assert extract_title(content) is None

    def test_extract_title_multiple_hashes(self):
        """Should only extract level 1 headers."""
        content = "## Level 2\nContent"
        assert extract_title(content) is None

    def test_extract_title_header_in_middle(self):
        """Should extract first header even if not at start."""
        content = "Some text\n# First Header\nMore content"
        assert extract_title(content) == "First Header"

    def test_extract_title_empty_content(self):
        """Should handle empty content."""
        assert extract_title("") is None

    def test_extract_title_only_whitespace(self):
        """Should handle whitespace-only content."""
        assert extract_title("   \n  \n  ") is None


class TestGetDisplayTitle:
    """Tests for get_display_title function."""

    def test_get_display_title_uses_extracted_title(self):
        """Should use extracted title from content."""
        episode = {"content": "# My Title\nBody text", "name": "default"}
        assert get_display_title(episode) == "My Title"

    def test_get_display_title_falls_back_to_content_preview(self):
        """Should use content preview when no header."""
        episode = {"content": "No header here", "name": "episode_name"}
        assert get_display_title(episode) == "No header here"

    def test_get_display_title_truncates_long_title(self):
        """Should truncate title to max_chars."""
        long_title = "a" * 100
        episode = {"content": f"# {long_title}\nBody", "name": "default"}
        result = get_display_title(episode, max_chars=50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_get_display_title_no_truncation_when_short(self):
        """Should not truncate when title is short."""
        episode = {"content": "# Short\nBody", "name": "default"}
        result = get_display_title(episode, max_chars=50)
        assert result == "Short"

    def test_get_display_title_respects_custom_max_chars(self):
        """Should respect custom max_chars parameter."""
        long_title = "a" * 50
        episode = {"content": f"# {long_title}\nBody", "name": "default"}
        result = get_display_title(episode, max_chars=20)
        assert len(result) == 20
        assert result == "a" * 20

    def test_get_display_title_empty_content(self):
        """Should handle empty content."""
        episode = {"content": "", "name": "fallback_name"}
        assert get_display_title(episode) == "fallback_name"

    def test_get_display_title_uses_content_preview_over_none_name(self):
        """Should use content preview instead of None name."""
        episode = {"content": "No header content", "name": None}
        result = get_display_title(episode)
        assert result == "No header content"
