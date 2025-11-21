"""Tests for Charlie TUI application screens and interactions.

These tests use Textual's headless testing mode to validate UI functionality
without requiring a terminal. All database operations are mocked.

Testing patterns demonstrated here:
- Use app.query_one() to assert widget state (not stdout)
- Use pilot.pause() after initialization for async operations
- Use pytest-textual-snapshot for visual testing (generates readable SVG files)

Workers are automatically managed by Textual during cleanup.

Known issue: Textual 6.6.0 raises CancelledError during test shutdown when
cleaning up screens. This is a framework issue during test cleanup and does
not affect test correctness. The app_test_context helper suppresses this error.

## Snapshot Testing Commands

Generate/update snapshot baselines (run after intentional UI changes):
    pytest --snapshot-update

Update specific test snapshot:
    pytest --snapshot-update -k test_home_empty_state_snapshot

Compare against baselines (normal test run):
    pytest tests/test_frontend/test_charlie.py
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from datetime import datetime
import subprocess
from unittest.mock import AsyncMock, Mock, patch, mock_open

import pytest

from charlie import CharlieApp, HomeScreen, ViewScreen, EditScreen, SettingsScreen


@asynccontextmanager
async def app_test_context(app):
    """Context manager that wraps app.run_test() and suppresses shutdown CancelledError.

    Textual 6.6.0 raises CancelledError during test cleanup when removing screens.
    This is a framework issue that doesn't affect test correctness - the test logic
    completes successfully before the error occurs during cleanup.
    """
    try:
        async with app.run_test() as pilot:
            yield pilot
            # Ensure orderly shutdown to avoid unawaited coroutine warnings
            await pilot.exit()
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_database():
    """Mock all database operations for testing."""
    with patch('charlie.ensure_database_ready', new_callable=AsyncMock) as mock_ensure, \
         patch('charlie.get_all_episodes', new_callable=AsyncMock) as mock_get_all, \
         patch('charlie.get_episode', new_callable=AsyncMock) as mock_get, \
         patch('charlie.add_journal_entry', new_callable=AsyncMock) as mock_add, \
         patch('charlie.update_episode', new_callable=AsyncMock) as mock_update, \
         patch('charlie.delete_episode', new_callable=AsyncMock) as mock_delete, \
         patch('charlie.shutdown_database') as mock_shutdown, \
         patch('charlie.get_inference_enabled') as mock_get_inference_enabled, \
         patch('charlie.set_inference_enabled') as mock_set_inference_enabled:

        mock_ensure.return_value = None
        mock_get_all.return_value = []
        mock_get_inference_enabled.return_value = False

        yield {
            'ensure': mock_ensure,
            'get_all': mock_get_all,
            'get': mock_get,
            'add': mock_add,
            'update': mock_update,
            'delete': mock_delete,
            'shutdown': mock_shutdown,
            'get_inference_enabled': mock_get_inference_enabled,
            'set_inference_enabled': mock_set_inference_enabled,
        }


class TestCharlieApp:
    """Tests for main CharlieApp class."""

    @pytest.mark.asyncio
    async def test_app_starts_and_configures_properly(self, mock_database):
        """Should start with proper configuration."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            # Wait for pending messages to be processed
            await pilot.pause()

            # Verify app configuration
            assert app.title == "Charlie"
            assert app.theme == "catppuccin-mocha"
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_app_starts_worker_always(self, mock_database):
        """Huey worker starts regardless of inference toggle; tasks enforce it."""
        mock_process = Mock()
        mock_process.wait.return_value = 0
        with patch('charlie.subprocess.Popen', return_value=mock_process) as mock_popen:
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.pause()  # allow call_after_refresh to run

            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert '-k' in args and '-w' in args
            # Shutdown should be invoked without error
            mock_process.send_signal.assert_called_with(signal.SIGINT)
            # stderr redirected to log file, stdout suppressed
            assert mock_popen.call_args[1]['stdout'] is subprocess.DEVNULL
            assert mock_popen.call_args[1]['stderr'] is not None


class TestHomeScreen:
    """Tests for HomeScreen functionality."""

    @pytest.mark.asyncio
    async def test_home_screen_shows_empty_state(self, mock_database):
        """Should display empty state when no episodes exist."""
        mock_database['get_all'].return_value = []

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            # Wait for initialization worker to complete
            await pilot.pause()

            # Check for empty state element
            empty_state = app.query_one("#empty-state")
            assert empty_state is not None
            assert "No entries yet!" in empty_state.renderable

    @pytest.mark.asyncio
    async def test_home_screen_displays_header(self, mock_database):
        """Should display Header with Charlie title."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            from textual.widgets import Header
            headers = app.query(Header)
            assert len(headers) > 0
            assert app.title == "Charlie"

    @pytest.mark.asyncio
    async def test_home_screen_has_footer(self, mock_database):
        """Should display Footer with key bindings."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            from textual.widgets import Footer
            footers = app.query(Footer)
            assert len(footers) > 0

    @pytest.mark.asyncio
    async def test_n_key_opens_new_entry_screen(self, mock_database):
        """Should open EditScreen when 'n' is pressed."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            assert isinstance(app.screen, EditScreen)
            assert app.screen.is_new_entry is True

    @pytest.mark.asyncio
    async def test_s_key_opens_settings(self, mock_database):
        """Should open SettingsScreen when 's' is pressed."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()

            assert isinstance(app.screen, SettingsScreen)

    @pytest.mark.asyncio
    async def test_home_loads_episodes(self, mock_database):
        """Should load and display episodes from database."""
        mock_episodes = [
            {
                "uuid": "123",
                "content": "# Test Entry\nContent",
                "name": "Test Entry",
                "valid_at": datetime(2025, 11, 19, 10, 0, 0)
            }
        ]
        mock_database['get_all'].return_value = mock_episodes

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            # Wait for initialization worker to complete
            await pilot.pause()

            home_screen = app.screen
            assert isinstance(home_screen, HomeScreen)
            assert len(home_screen.episodes) == 1

    def test_home_empty_state_snapshot(self, snap_compare, mock_database):
        """Visual regression test: empty home screen with no entries."""
        mock_database['get_all'].return_value = []
        assert snap_compare(CharlieApp())


class TestViewScreen:
    """Tests for ViewScreen functionality."""

    @pytest.mark.asyncio
    async def test_view_screen_displays_content(self, mock_database):
        """Should display episode content."""
        mock_episode = {
            "uuid": "123",
            "content": "# Test Title\nTest content",
            "name": "Test Entry",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, ViewScreen)

            from textual.widgets import Markdown
            markdown = app.query_one("#content", Markdown)
            assert markdown is not None

    @pytest.mark.asyncio
    async def test_view_screen_has_header(self, mock_database):
        """Should display Header."""
        mock_episode = {
            "uuid": "123",
            "content": "# Test\nContent",
            "name": "Test",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            from textual.widgets import Header
            headers = app.query(Header)
            assert len(headers) > 0

    @pytest.mark.asyncio
    async def test_e_key_opens_edit_from_view(self, mock_database):
        """Should open EditScreen when 'e' is pressed."""
        mock_episode = {
            "uuid": "123",
            "content": "# Test\nContent",
            "name": "Test",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            assert isinstance(app.screen, EditScreen)
            assert app.screen.episode_uuid == "123"

    @pytest.mark.asyncio
    async def test_escape_returns_to_home(self, mock_database):
        """Should return to HomeScreen when escape is pressed."""
        mock_episode = {
            "uuid": "123",
            "content": "# Test\nContent",
            "name": "Test",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)


class TestEditScreen:
    """Tests for EditScreen functionality."""

    @pytest.mark.asyncio
    async def test_edit_screen_has_text_area(self, mock_database):
        """Should display TextArea for editing."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.query_one("#editor", TextArea)
            assert editor is not None

    @pytest.mark.asyncio
    async def test_edit_screen_has_header(self, mock_database):
        """Should display Header."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import Header
            headers = app.query(Header)
            assert len(headers) > 0

    @pytest.mark.asyncio
    async def test_edit_saves_new_entry(self, mock_database):
        """Should save new entry when escape is pressed."""
        mock_database['add'].return_value = "new-uuid"

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.query_one("#editor", TextArea)
            editor.text = "# New Entry\nContent"

            await pilot.press("escape")
            await pilot.pause()

            # Should have saved
            mock_database['add'].assert_called_once()


class TestSettingsScreen:
    """Tests for SettingsScreen functionality."""

    @pytest.mark.asyncio
    async def test_settings_opens_as_modal(self, mock_database):
        """Should open as modal overlay."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()

            assert isinstance(app.screen, SettingsScreen)

    @pytest.mark.asyncio
    async def test_settings_shows_inference_toggle(self, mock_database):
        """Should display inference toggle switch."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            from textual.widgets import Switch
            switches = app.query(Switch)
            assert len(switches) == 1

            toggle = app.query_one("#inference-toggle", Switch)
            assert toggle is not None

    @pytest.mark.asyncio
    async def test_settings_toggles_initial_state(self, mock_database):
        """Should load initial state from backend."""
        mock_database['get_inference_enabled'].return_value = True
        with patch('charlie.subprocess.Popen'):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.press("s")
                await pilot.pause()

                from textual.widgets import Switch
                toggle = app.query_one("#inference-toggle", Switch)

                assert toggle.value is True

        # Reset for other tests
        mock_database['get_inference_enabled'].return_value = False

    @pytest.mark.asyncio
    async def test_settings_closes_with_s_key(self, mock_database):
        """Should close when 's' is pressed."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_settings_closes_with_escape(self, mock_database):
        """Should close when escape is pressed."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_settings_tab_navigation(self, mock_database):
        """Should navigate with tab key."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            # Tab navigation should work
            await pilot.press("tab")
            await pilot.pause()

            from textual.widgets import Switch
            toggle = app.query_one("#inference-toggle", Switch)
            initial = toggle.value

            await pilot.press("space")
            await pilot.pause()

            # Toggle should have changed and persisted
            assert toggle.value != initial
            mock_database['set_inference_enabled'].assert_called_once_with(toggle.value)

    @pytest.mark.asyncio
    async def test_settings_vim_navigation(self, mock_database):
        """Should support j/k navigation keys."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            # Vim keys should work for navigation
            await pilot.press("j")
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()

            # Basic test - bindings exist and don't crash
            assert isinstance(app.screen, SettingsScreen)

    @pytest.mark.asyncio
    async def test_settings_toggle_disables_worker(self, mock_database):
        """Disabling inference should NOT stop the Huey worker (tasks honor toggle)."""
        mock_database['get_inference_enabled'].return_value = True

        with patch('charlie.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.press("s")
                await pilot.pause()

                with patch.object(app, "stop_huey_worker") as mock_stop:
                    from textual.widgets import Switch
                    toggle = app.query_one("#inference-toggle", Switch)
                    toggle.value = True  # starting state enabled
                    await pilot.press("space")  # toggle off
                    await pilot.pause()

                    mock_stop.assert_not_called()
                    mock_database['set_inference_enabled'].assert_called_with(False)


class TestIntegration:
    """Integration tests for complete user workflows."""

    @pytest.mark.asyncio
    async def test_create_entry_workflow(self, mock_database):
        """Should create and save a new journal entry."""
        mock_database['add'].return_value = "new-uuid"

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Create new entry
            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.query_one("#editor", TextArea)
            editor.text = "# My Entry\nSome content"

            # Save
            await pilot.press("escape")
            await pilot.pause()

            # Verify saved
            mock_database['add'].assert_called_once()
            mock_database['update'].assert_called_once()

    @pytest.mark.asyncio
    async def test_settings_workflow(self, mock_database):
        """Should open settings, toggle, and return."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open settings
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)

            # Toggle a switch
            await pilot.press("tab")
            await pilot.pause()

            from textual.widgets import Switch
            toggle = app.query_one("#inference-toggle", Switch)
            initial = toggle.value

            await pilot.press("space")
            await pilot.pause()
            assert toggle.value != initial
            mock_database['set_inference_enabled'].assert_called_once_with(toggle.value)

            # Close settings
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)


class TestWorkerManagement:
    """Tests for Huey worker lifecycle management."""

    @pytest.mark.asyncio
    async def test_worker_startup_closes_file_on_popen_failure(self, mock_database):
        """Should close log file handle if subprocess.Popen raises exception."""
        from pathlib import Path

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            mock_file = Mock()
            with patch('charlie.open', mock_open()) as mock_open_func, \
                 patch('charlie.subprocess.Popen', side_effect=FileNotFoundError("huey_consumer not found")), \
                 patch.object(Path, 'mkdir'):

                mock_open_func.return_value = mock_file

                app._ensure_huey_worker_running()

                mock_file.close.assert_called_once()
                assert app.huey_log_file is None
                assert app.huey_process is None
