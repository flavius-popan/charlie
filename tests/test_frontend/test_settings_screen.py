"""Unit tests for SettingsScreen widget.

These tests use a lightweight test app that mounts SettingsScreen directly,
avoiding the overhead of full CharlieApp initialization.
"""

import pytest
from unittest.mock import patch, MagicMock
from textual.app import App
from textual.widgets import Switch, Footer, Label

from frontend.screens.settings_screen import SettingsScreen


class SettingsScreenTestApp(App):
    """Lightweight test app for SettingsScreen unit tests."""

    def __init__(self, inference_enabled=False):
        super().__init__()
        self.initial_inference = inference_enabled

    def on_mount(self):
        with patch("frontend.screens.settings_screen.get_inference_enabled",
                   return_value=self.initial_inference):
            self.push_screen(SettingsScreen())


@pytest.fixture
def mock_settings_db():
    """Mock database operations for SettingsScreen tests."""
    with patch("frontend.screens.settings_screen.get_inference_enabled", return_value=False) as mock_get, \
         patch("frontend.screens.settings_screen.set_inference_enabled") as mock_set:
        yield {
            "get_inference_enabled": mock_get,
            "set_inference_enabled": mock_set,
        }


@pytest.mark.asyncio
async def test_settings_screen_has_switch(mock_settings_db):
    """Should display inference toggle switch."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        switches = app.screen.query(Switch)
        assert len(switches) == 1

        toggle = app.screen.query_one("#inference-toggle", Switch)
        assert toggle is not None


@pytest.mark.asyncio
async def test_settings_screen_has_footer(mock_settings_db):
    """Should display Footer with key bindings."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        footers = app.screen.query(Footer)
        assert len(footers) > 0


@pytest.mark.asyncio
async def test_settings_screen_has_title(mock_settings_db):
    """Should display settings title."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        title = app.screen.query_one("#settings-title", Label)
        assert title is not None
        assert "Settings" in title.render().plain


@pytest.mark.asyncio
async def test_settings_screen_toggle_initial_off(mock_settings_db):
    """Toggle should start off when inference is disabled."""
    mock_settings_db["get_inference_enabled"].return_value = False

    with patch("frontend.screens.settings_screen.get_inference_enabled", return_value=False):
        app = SettingsScreenTestApp(inference_enabled=False)
        async with app.run_test() as pilot:
            await pilot.pause()

            toggle = app.screen.query_one("#inference-toggle", Switch)
            assert toggle.value is False


@pytest.mark.asyncio
async def test_settings_screen_toggle_initial_on(mock_settings_db):
    """Toggle should start on when inference is enabled."""
    mock_settings_db["get_inference_enabled"].return_value = True

    with patch("frontend.screens.settings_screen.get_inference_enabled", return_value=True):
        app = SettingsScreenTestApp(inference_enabled=True)
        async with app.run_test() as pilot:
            await pilot.pause()

            toggle = app.screen.query_one("#inference-toggle", Switch)
            assert toggle.value is True


@pytest.mark.asyncio
async def test_settings_screen_toggle_focus(mock_settings_db):
    """Toggle switch should be focusable."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        toggle = app.screen.query_one("#inference-toggle", Switch)
        assert toggle.can_focus is True


@pytest.mark.asyncio
async def test_settings_screen_toggle_changes_value(mock_settings_db):
    """Space on toggle should change its value and call setter."""
    app = SettingsScreenTestApp(inference_enabled=False)
    async with app.run_test() as pilot:
        await pilot.pause()

        toggle = app.screen.query_one("#inference-toggle", Switch)
        toggle.focus()

        # Toggle should start off
        assert toggle.value is False

        await pilot.press("space")
        await pilot.pause()

        # Toggle should now be on
        assert toggle.value is True
        mock_settings_db["set_inference_enabled"].assert_called_once_with(True)


@pytest.mark.asyncio
async def test_settings_screen_escape_dismisses(mock_settings_db):
    """Escape should dismiss the settings modal."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("escape")
        await pilot.pause()

        # Should have dismissed (no longer on SettingsScreen)
        assert not isinstance(app.screen, SettingsScreen)


@pytest.mark.asyncio
async def test_settings_screen_s_key_dismisses(mock_settings_db):
    """Pressing 's' should dismiss the settings modal."""
    app = SettingsScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("s")
        await pilot.pause()

        # Should have dismissed
        assert not isinstance(app.screen, SettingsScreen)
