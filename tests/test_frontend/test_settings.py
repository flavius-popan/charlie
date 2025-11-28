"""Tests for frontend settings - theme file handling.

IMPORTANT: All tests MUST use tmp_path or mocks. Never touch the real THEME_FILE.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from frontend.settings import DEFAULT_THEME
import frontend.settings as settings_module


class TestGetTheme:
    """Tests for get_theme - must be bulletproof and NEVER crash."""

    def test_returns_theme_from_file(self, tmp_path):
        """Normal case: file exists with valid theme."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_text("textual-dark")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == "textual-dark"

    def test_file_not_found_returns_default(self, tmp_path):
        """Missing file should return default, no errors."""
        theme_file = tmp_path / "nonexistent" / "theme.txt"

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == DEFAULT_THEME

    def test_permission_denied_returns_default(self, tmp_path):
        """Unreadable file should return default, no errors."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_text("some-theme")
        theme_file.chmod(0o000)

        try:
            with patch.object(settings_module, "THEME_FILE", theme_file):
                assert settings_module.get_theme() == DEFAULT_THEME
        finally:
            theme_file.chmod(0o644)

    def test_empty_file_returns_default(self, tmp_path):
        """Empty file should return default."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_text("")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == DEFAULT_THEME

    def test_whitespace_only_returns_default(self, tmp_path):
        """Whitespace-only file should return default."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_text("   \n\t  \n  ")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == DEFAULT_THEME

    def test_strips_whitespace(self, tmp_path):
        """Theme with surrounding whitespace should be stripped."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_text("  textual-dark  \n")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == "textual-dark"

    def test_binary_garbage_returns_default(self, tmp_path):
        """Binary file should return default, not crash."""
        theme_file = tmp_path / "theme.txt"
        theme_file.write_bytes(b"\x80\x81\x82\xff\xfe")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            assert settings_module.get_theme() == DEFAULT_THEME


class TestSetTheme:
    """Tests for set_theme - must NEVER crash."""

    def test_writes_theme_to_file(self, tmp_path):
        """Should write theme to file."""
        theme_file = tmp_path / "data" / "theme.txt"

        with patch.object(settings_module, "THEME_FILE", theme_file):
            settings_module.set_theme("textual-dark")

        assert theme_file.read_text() == "textual-dark"

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directories if needed."""
        theme_file = tmp_path / "nested" / "dir" / "theme.txt"

        with patch.object(settings_module, "THEME_FILE", theme_file):
            settings_module.set_theme("catppuccin-mocha")

        assert theme_file.exists()
        assert theme_file.read_text() == "catppuccin-mocha"

    def test_overwrites_existing_file(self, tmp_path):
        """Should overwrite existing theme."""
        theme_file = tmp_path / "theme.txt"
        theme_file.parent.mkdir(parents=True, exist_ok=True)
        theme_file.write_text("old-theme")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            settings_module.set_theme("new-theme")

        assert theme_file.read_text() == "new-theme"

    def test_permission_denied_does_not_crash(self, tmp_path):
        """Write failure should NOT crash - just silently fail."""
        # Use a path we definitely can't write to
        theme_file = Path("/root/cant_write_here/theme.txt")

        with patch.object(settings_module, "THEME_FILE", theme_file):
            # This should NOT raise - just silently fail
            settings_module.set_theme("new-theme")
        # If we got here without exception, test passed

    def test_any_exception_does_not_crash(self, tmp_path):
        """Any exception during write should be swallowed."""
        # Create a directory where the file should be - can't write a file over a dir
        theme_file = tmp_path / "theme.txt"
        theme_file.mkdir()  # Make it a directory, not a file

        with patch.object(settings_module, "THEME_FILE", theme_file):
            # This should NOT raise - trying to write to a directory fails silently
            settings_module.set_theme("whatever")
        # If we got here without exception, test passed
