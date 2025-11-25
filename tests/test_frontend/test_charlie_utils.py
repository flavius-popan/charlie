"""Tests for Charlie TUI utility functions."""

import pytest

from frontend.utils import get_display_title


class TestGetDisplayTitle:
    """Tests for get_display_title function."""

    def test_get_display_title_uses_preview(self):
        """Should prefer preview when provided."""
        episode = {"preview": "Preview text", "name": "fallback"}
        assert get_display_title(episode) == "Preview text"

    def test_get_display_title_falls_back_to_name(self):
        """Should use name when preview missing."""
        episode = {"name": "Episode Name"}
        assert get_display_title(episode) == "Episode Name"

    def test_get_display_title_truncates_with_ellipsis(self):
        """Should truncate long preview and add ellipsis."""
        long_preview = "a" * 100
        episode = {"preview": long_preview, "name": "fallback"}
        result = get_display_title(episode, max_chars=50)
        assert result.endswith("...")
        assert len(result) <= 53  # 50 chars plus ellipsis adjustments

    def test_get_display_title_no_truncation_when_short(self):
        """Should not truncate when text fits."""
        episode = {"preview": "Short", "name": "fallback"}
        assert get_display_title(episode, max_chars=50) == "Short"

    def test_get_display_title_handles_missing_fields(self):
        """Should fall back to default when fields absent."""
        episode = {}
        assert get_display_title(episode) == "Untitled"
