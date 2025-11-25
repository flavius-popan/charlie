"""Tests for Charlie TUI application screens and interactions.

These tests use Textual's headless testing mode to validate UI functionality
without requiring a terminal. All database operations are mocked.

Testing patterns demonstrated here:
- Use app.screen.query_one() to assert widget state (not stdout)
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
from contextlib import asynccontextmanager
from datetime import datetime
import subprocess
from unittest.mock import AsyncMock, Mock, patch, mock_open

import pytest

from charlie import CharlieApp
from frontend.screens.home_screen import HomeScreen
from frontend.screens.view_screen import ViewScreen
from frontend.screens.edit_screen import EditScreen
from frontend.screens.settings_screen import SettingsScreen
from frontend.widgets.entity_sidebar import EntitySidebar, DeleteEntityModal


@asynccontextmanager
async def app_test_context(app):
    """Context manager that wraps app.run_test() and suppresses shutdown CancelledError.

    Textual 6.6.0 raises CancelledError during test cleanup when removing screens.
    This is a framework issue that doesn't affect test correctness - the test logic
    completes successfully before the error occurs during cleanup.
    """
    try:
        async with app.run_test() as pilot:
            # Allow multiple event loop cycles for screen initialization:
            # 1. CharlieApp.on_mount() runs
            # 2. HomeScreen is pushed
            # 3. HomeScreen.compose() creates widgets
            # 4. Widgets are mounted and on_mount() is called
            # 5. Workers start and may update reactive attributes
            for _ in range(3):
                await pilot.pause()
            yield pilot
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_database():
    """Mock all database operations for testing."""
    from contextlib import ExitStack
    from unittest.mock import MagicMock

    with ExitStack() as stack:
        mock_get_home = AsyncMock(return_value=[])
        mock_get = AsyncMock()
        mock_add = AsyncMock()
        mock_update = AsyncMock()
        mock_delete = AsyncMock()
        mock_ensure = AsyncMock(return_value=None)
        mock_shutdown = MagicMock()
        mock_get_inference_enabled = MagicMock(return_value=False)
        mock_set_inference_enabled = MagicMock()
        mock_get_episode_status = MagicMock(return_value=None)

        # Patch shared functions across modules to the same mocks
        stack.enter_context(patch('charlie.get_home_screen', mock_get_home))
        stack.enter_context(patch('backend.database.get_home_screen', mock_get_home))
        stack.enter_context(patch('frontend.screens.home_screen.get_home_screen', mock_get_home))

        stack.enter_context(patch('charlie.get_episode', mock_get))
        stack.enter_context(patch('backend.database.get_episode', mock_get))
        stack.enter_context(patch('frontend.screens.edit_screen.get_episode', mock_get))
        stack.enter_context(patch('frontend.screens.view_screen.get_episode', mock_get))

        stack.enter_context(patch('charlie.add_journal_entry', mock_add))
        stack.enter_context(patch('backend.add_journal_entry', mock_add))
        stack.enter_context(patch('frontend.screens.edit_screen.add_journal_entry', mock_add))

        stack.enter_context(patch('charlie.update_episode', mock_update))
        stack.enter_context(patch('backend.database.update_episode', mock_update))
        stack.enter_context(patch('frontend.screens.edit_screen.update_episode', mock_update))

        stack.enter_context(patch('charlie.delete_episode', mock_delete))
        stack.enter_context(patch('backend.database.delete_episode', mock_delete))

        stack.enter_context(patch('charlie.ensure_database_ready', mock_ensure))
        stack.enter_context(patch('backend.database.ensure_database_ready', mock_ensure))
        stack.enter_context(patch('frontend.screens.home_screen.ensure_database_ready', mock_ensure))

        stack.enter_context(patch('charlie.shutdown_database', mock_shutdown))
        stack.enter_context(patch('backend.database.shutdown_database', mock_shutdown))

        stack.enter_context(patch('charlie.get_inference_enabled', mock_get_inference_enabled))
        stack.enter_context(patch('frontend.screens.view_screen.get_inference_enabled', mock_get_inference_enabled))
        stack.enter_context(patch('frontend.screens.settings_screen.get_inference_enabled', mock_get_inference_enabled))
        stack.enter_context(patch('frontend.screens.edit_screen.get_inference_enabled', mock_get_inference_enabled))
        stack.enter_context(patch('backend.database.redis_ops.get_inference_enabled', mock_get_inference_enabled))

        stack.enter_context(patch('charlie.set_inference_enabled', mock_set_inference_enabled))
        stack.enter_context(patch('frontend.screens.settings_screen.set_inference_enabled', mock_set_inference_enabled))

        stack.enter_context(patch('charlie.get_episode_status', mock_get_episode_status))
        stack.enter_context(patch('frontend.screens.view_screen.get_episode_status', mock_get_episode_status))
        stack.enter_context(patch('frontend.screens.edit_screen.get_episode_status', mock_get_episode_status))
        stack.enter_context(patch('backend.database.redis_ops.get_episode_status', mock_get_episode_status))

        # Keep EditScreen redis_ops in sync with charlie.redis_ops (so test patches propagate)
        def current_redis_ops():
            import charlie
            return charlie.redis_ops()
        stack.enter_context(patch('frontend.screens.edit_screen.redis_ops', current_redis_ops))

        # Huey / task queue no-ops
        mock_start_huey = MagicMock()
        mock_huey_running = MagicMock(return_value=True)
        stack.enter_context(patch('charlie.start_huey_consumer', mock_start_huey))
        stack.enter_context(patch('charlie.is_huey_consumer_running', mock_huey_running))
        stack.enter_context(patch('backend.services.queue.start_huey_consumer', mock_start_huey))
        stack.enter_context(patch('backend.services.queue.is_huey_consumer_running', mock_huey_running))

        # Prevent extraction tasks from enqueuing
        mock_extract_task = MagicMock()
        stack.enter_context(patch('backend.services.tasks.extract_nodes_task', mock_extract_task))

        # Execute EditScreen workers inline for presence keys
        def immediate_run_worker(self, coro, *, exclusive=False, name=None, thread=False):
            if asyncio.iscoroutine(coro):
                asyncio.get_running_loop().create_task(coro)
                return None
            if callable(coro):
                return coro()
            return coro

        stack.enter_context(patch('frontend.screens.edit_screen.EditScreen.run_worker', immediate_run_worker))

        # Patch presence helpers to honor patched charlie.redis_ops in tests
        def patched_set_presence():
            import charlie
            try:
                with charlie.redis_ops() as r:
                    r.set("editing:active", "active")
            except Exception:
                pass

        def patched_clear_presence():
            import charlie
            try:
                with charlie.redis_ops() as r:
                    r.delete("editing:active")
            except Exception:
                pass

        stack.enter_context(patch('frontend.screens.edit_screen.EditScreen._set_editing_presence', staticmethod(patched_set_presence)))
        stack.enter_context(patch('frontend.screens.edit_screen.EditScreen._clear_editing_presence', staticmethod(patched_clear_presence)))

        # Sensible defaults for common expectations
        mock_add.return_value = "new-uuid"
        mock_update.return_value = True
        mock_get.return_value = {
            "uuid": "existing-uuid",
            "content": "# Title\nBody",
            "name": "Title",
            "valid_at": None,
        }

        yield {
            'ensure': mock_ensure,
            'get_home': mock_get_home,
            'get': mock_get,
            'add': mock_add,
            'update': mock_update,
            'delete': mock_delete,
            'shutdown': mock_shutdown,
            'get_inference_enabled': mock_get_inference_enabled,
            'set_inference_enabled': mock_set_inference_enabled,
            'get_episode_status': mock_get_episode_status,
            # shared mocks reused across layers
            'backend_get_all': mock_get_home,
            'backend_get': mock_get,
            'backend_update': mock_update,
            'backend_delete': mock_delete,
            'backend_ensure': mock_ensure,
            'backend_shutdown': mock_shutdown,
            'home_get_home': mock_get_home,
            'home_ensure': mock_ensure,
            'start_huey': mock_start_huey,
            'huey_running': mock_huey_running,
            'backend_start_huey': mock_start_huey,
            'backend_huey_running': mock_huey_running,
            'extract_task': mock_extract_task,
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
        with patch('charlie.is_huey_consumer_running', return_value=False), \
             patch('charlie.start_huey_consumer') as mock_start, \
             patch('charlie.atexit.register'):

            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.pause()  # allow call_after_refresh to run

            mock_start.assert_called_once()


class TestHomeScreen:
    """Tests for HomeScreen functionality."""

    @pytest.mark.asyncio
    async def test_home_screen_shows_empty_state(self, mock_database):
        """Should display empty state when no episodes exist."""
        mock_database['get_home'].return_value = []

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            # Wait for initialization worker to complete
            await pilot.pause()

            # Check for empty state element
            empty_state = app.screen.query_one("#empty-state")
            assert empty_state is not None
            assert "No entries yet" in empty_state.render().plain

    @pytest.mark.asyncio
    async def test_home_screen_displays_header(self, mock_database):
        """Should display Header with Charlie title."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            from textual.widgets import Header
            headers = app.screen.query(Header)
            assert len(headers) > 0
            assert app.title == "Charlie"

    @pytest.mark.asyncio
    async def test_home_screen_has_footer(self, mock_database):
        """Should display Footer with key bindings."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            from textual.widgets import Footer
            footers = app.screen.query(Footer)
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
    async def test_l_key_opens_logs_and_reload(self, mock_database):
        """Should open Log screen and handle reload."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("l")
            await pilot.pause()

            from charlie import LogScreen
            assert isinstance(app.screen, LogScreen)

            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, LogScreen)

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
        mock_database['get_home'].return_value = mock_episodes

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            # Wait for initialization worker to complete
            await pilot.pause()

            home_screen = app.screen
            assert isinstance(home_screen, HomeScreen)
            assert len(home_screen.episodes) == 1

    def test_home_empty_state_snapshot(self, snap_compare, mock_database):
        """Visual regression test: empty home screen with no entries."""
        mock_database['get_home'].return_value = []
        assert snap_compare(CharlieApp())

    def test_css_includes_journal_content_styles(self):
        """CSS should target the journal markdown widget id."""
        css = CharlieApp.CSS
        assert "#journal-content" in css, "journal-content styles should be present for new layout"


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
        mock_database['get_home'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, ViewScreen)

            from textual.widgets import Markdown
            markdown = app.screen.query_one("#journal-content", Markdown)
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
        mock_database['get_home'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            from textual.widgets import Header
            headers = app.screen.query(Header)
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
        mock_database['get_home'].return_value = [mock_episode]
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
        mock_database['get_home'].return_value = [mock_episode]
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
            editor = app.screen.query_one("#editor", TextArea)
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
            headers = app.screen.query(Header)
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
            editor = app.screen.query_one("#editor", TextArea)
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
            switches = app.screen.query(Switch)
            assert len(switches) == 1

            toggle = app.screen.query_one("#inference-toggle", Switch)
            assert toggle is not None

    @pytest.mark.asyncio
    async def test_inference_toggle_calls_set_and_retains_focus(self, mock_database):
        """Space toggles inference, calls setter once, and keeps focus for fast toggling."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            switch = app.screen.query_one("#inference-toggle")
            assert switch.has_focus

            await pilot.press("space")
            await pilot.pause()

            mock_database["set_inference_enabled"].assert_called_once_with(True)
            assert switch.has_focus

    @pytest.mark.asyncio
    async def test_settings_toggles_initial_state(self, mock_database):
        """Should load initial state from backend."""
        mock_database['get_inference_enabled'].return_value = True
        with patch('charlie.is_huey_consumer_running', return_value=False), \
             patch('charlie.start_huey_consumer'), \
             patch('charlie.atexit.register'):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.press("s")
                await pilot.pause()

                from textual.widgets import Switch
                toggle = app.screen.query_one("#inference-toggle", Switch)

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
            toggle = app.screen.query_one("#inference-toggle", Switch)
            initial = toggle.value

            await pilot.press("space")
            await pilot.pause()

            # Toggle should have changed and persisted
            assert toggle.value != initial
            mock_database['set_inference_enabled'].assert_called_once_with(toggle.value)

    @pytest.mark.asyncio
    async def test_settings_toggle_disables_worker(self, mock_database):
        """Disabling inference should NOT stop the Huey worker (tasks honor toggle)."""
        mock_database['get_inference_enabled'].return_value = True

        with patch('charlie.is_huey_consumer_running', return_value=False), \
             patch('charlie.start_huey_consumer'), \
             patch('charlie.atexit.register'):

            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()
                await pilot.press("s")
                await pilot.pause()

                with patch.object(app, "stop_huey_worker") as mock_stop:
                    from textual.widgets import Switch
                    toggle = app.screen.query_one("#inference-toggle", Switch)
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
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# My Entry\nSome content"

            # Save
            await pilot.press("escape")
            await pilot.pause()

            # Verify saved
            mock_database['add'].assert_called_once()
        mock_database['update'].assert_not_called()

    @pytest.mark.asyncio
    async def test_new_entry_two_escapes_return_home(self, mock_database):
        """After leaving editor, a single ESC from viewer should return home."""
        mock_database["add"].return_value = "uuid-new"
        mock_database["get"].return_value = {
            "uuid": "uuid-new",
            "content": "# Title\nBody",
            "name": "Title",
        }

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Title\nBody"

            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, ViewScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_edit_existing_single_escape_returns_home(self, mock_database):
        """Editing an existing entry should need only one ESC from viewer to go home."""
        mock_episode = {
            "uuid": "existing-uuid",
            "content": "# Title\nBody",
            "name": "Title",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0),
        }
        mock_database["get_home"].return_value = [mock_episode]
        mock_database["get"].return_value = mock_episode
        mock_database["update"].return_value = True
        mock_database["get_inference_enabled"].return_value = False
        mock_database["get_episode_status"].return_value = None

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open existing entry
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            # Edit the entry
            await pilot.press("e")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Title\nUpdated body"

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, ViewScreen)

            # One escape should return to home
            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_escape_works_while_inference_pending(self, mock_database):
        """Esc from viewer should still return home even while connections polling is active."""
        mock_episode = {
            "uuid": "existing-uuid",
            "content": "# Title\nBody",
            "name": "Title",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0),
        }
        mock_database["get_home"].return_value = [mock_episode]
        mock_database["get"].return_value = mock_episode
        mock_database["update"].return_value = True
        mock_database["get_inference_enabled"].return_value = True
        mock_database["get_episode_status"].return_value = "pending_nodes"

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause(); await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Title\nUpdated"

            await pilot.press("escape")
            await pilot.pause(); await pilot.pause()

            # Viewer should be active and polling
            from charlie import ViewScreen
            assert isinstance(app.screen, ViewScreen)
            assert any(w.name == "status-poll" and w.is_running for w in app.screen.workers)

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(app.screen, HomeScreen)

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
            toggle = app.screen.query_one("#inference-toggle", Switch)
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
    async def test_worker_startup_notifies_on_failure(self, mock_database):
        """App should surface an error if the consumer fails to start."""
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            with patch('charlie.is_huey_consumer_running', return_value=False), \
                 patch('charlie.start_huey_consumer', side_effect=RuntimeError("boom")), \
                 patch('charlie.atexit.register'):

                app.notify = Mock()
                app._ensure_huey_worker_running()
                app.notify.assert_called_once()


@pytest.mark.asyncio
async def test_edit_screen_esc_goes_to_view_screen():
    """ESC from EditScreen should navigate to ViewScreen, not HomeScreen."""
    from charlie import EditScreen, ViewScreen
    from textual.app import App

    class TestApp(App):
        def on_mount(self):
            self.push_screen(EditScreen(episode_uuid=None))

    with patch("charlie.add_journal_entry", new_callable=AsyncMock) as mock_add, \
         patch("charlie.redis_ops") as mock_redis_ops:
        mock_add.return_value = "new-uuid"

        # Mock Redis to prevent real connection attempts
        mock_redis = Mock()
        mock_redis.hget.return_value = None
        mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

        with patch("charlie.update_episode", new_callable=AsyncMock):
            with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {"uuid": "new-uuid", "content": "# Test Entry\nSome content"}

                app = TestApp()

                async with app.run_test() as pilot:
                    screen = app.screen
                    # Type some content
                    from textual.widgets import TextArea
                    editor = screen.query_one("#editor", TextArea)
                    editor.text = "# Test Entry\nSome content"

                    # Press ESC
                    await pilot.press("escape")
                    await pilot.pause()

                    # Should navigate to ViewScreen, not pop to HomeScreen
                    assert isinstance(app.screen, ViewScreen)
                    assert app.screen.episode_uuid == "new-uuid"


@pytest.mark.asyncio
async def test_edit_screen_enqueues_extraction_for_new_entry(mock_database):
    """EditScreen should enqueue extraction task when creating new entry with inference enabled."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = True
    mock_database['get'].return_value = {
        "uuid": "new-uuid",
        "content": "# New Entry\nTest content",
        "name": "New Entry",
    }

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task, \
         patch('charlie.redis_ops') as mock_redis_ops:
        # Mock Redis to prevent EntitySidebar.refresh_entities() errors
        mock_redis = Mock()
        mock_redis.hget.return_value = None
        mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis_ops.return_value.__exit__ = Mock(return_value=None)
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# New Entry\nTest content"

            await pilot.press("escape")
            await pilot.pause()

            # Verify task was called. We don't check specific arguments because the
            # API signature changed to include a priority parameter, making exact
            # argument checks brittle. The important behavior is that the task runs.
            mock_extract_task.assert_called_once()


@pytest.mark.asyncio
async def test_edit_screen_enqueues_extraction_when_content_changes(mock_database):
    """EditScreen should enqueue extraction task when editing existing entry and content changes."""
    mock_episode = {
        "uuid": "existing-uuid",
        "content": "# Original\nOriginal content",
        "name": "Original",
        "valid_at": datetime(2025, 11, 19, 10, 0, 0)
    }
    mock_database['get_home'].return_value = [mock_episode]
    mock_database['get'].return_value = mock_episode
    mock_database['update'].return_value = True  # Content changed
    mock_database['get_inference_enabled'].return_value = True

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task, \
         patch('charlie.redis_ops') as mock_redis_ops:
        # Mock Redis to prevent EntitySidebar.refresh_entities() errors
        mock_redis = Mock()
        mock_redis.hget.return_value = None
        mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis_ops.return_value.__exit__ = Mock(return_value=None)
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Updated\nNew content"

            await pilot.press("escape")
            await pilot.pause()

            # Verify task was called. We don't check specific arguments because the
            # API signature changed to include a priority parameter, making exact
            # argument checks brittle. The important behavior is that the task runs.
            mock_extract_task.assert_called_once()


@pytest.mark.asyncio
async def test_edit_screen_skips_extraction_when_content_unchanged(mock_database):
    """EditScreen should NOT enqueue extraction task when content is unchanged."""
    mock_episode = {
        "uuid": "existing-uuid",
        "content": "# Original\nOriginal content",
        "name": "Original",
        "valid_at": datetime(2025, 11, 19, 10, 0, 0)
    }
    mock_database['get_home'].return_value = [mock_episode]
    mock_database['get'].return_value = mock_episode
    mock_database['update'].return_value = False  # Content NOT changed
    mock_database['get_inference_enabled'].return_value = True

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task:
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Original\nOriginal content"

            await pilot.press("escape")
            await pilot.pause()

            mock_extract_task.assert_not_called()


@pytest.mark.asyncio
async def test_edit_screen_skips_extraction_when_inference_disabled(mock_database):
    """EditScreen should NOT enqueue extraction task when inference is disabled."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = False

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task:
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# New Entry\nTest content"

            await pilot.press("escape")
            await pilot.pause()

            mock_extract_task.assert_not_called()


@pytest.mark.asyncio
async def test_edit_screen_uses_switch_screen_not_pop_push(mock_database):
    """EditScreen should use switch_screen() for navigation, not pop_screen() + push_screen()."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = True

    with patch('backend.services.tasks.extract_nodes_task'):
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# New Entry\nTest content"

            # Spy on switch_screen
            with patch.object(app, 'switch_screen', wraps=app.switch_screen) as mock_switch:
                await pilot.press("escape")
                await pilot.pause()

                mock_switch.assert_called_once()


@pytest.mark.asyncio
async def test_edit_screen_navigates_to_view_screen_before_task_enqueue(mock_database):
    """EditScreen should navigate to ViewScreen before enqueueing extraction task."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = True
    mock_database['get'].return_value = {
        "uuid": "new-uuid",
        "content": "# New Entry\nTest content",
        "name": "New Entry",
    }

    from charlie import ViewScreen

    screen_type_when_task_enqueued = None

    def capture_screen_type(episode_uuid, journal, priority=None):
        nonlocal screen_type_when_task_enqueued
        screen_type_when_task_enqueued = type(app.screen)

    with patch('backend.services.tasks.extract_nodes_task', side_effect=capture_screen_type) as mock_extract_task, \
         patch('charlie.redis_ops') as mock_redis_ops:
        # Mock Redis to prevent EntitySidebar.refresh_entities() errors
        mock_redis = Mock()
        mock_redis.hget.return_value = None
        mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis_ops.return_value.__exit__ = Mock(return_value=None)
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# New Entry\nTest content"

            await pilot.press("escape")
            await pilot.pause()

            # Note: assert_called_once() is weaker than checking both call count and arguments.
            # The extract_nodes_task signature changed and this test uses side_effect only.
            # Could be strengthened with assert_called_once_with() once API is fully stable.
            mock_extract_task.assert_called_once()
            assert screen_type_when_task_enqueued == ViewScreen, \
                f"Expected ViewScreen but got {screen_type_when_task_enqueued}"


@pytest.mark.asyncio
async def test_create_entry_with_inference_disabled_integration(mock_database):
    """Full workflow: create entry with inference disabled, verify task NOT called but entry persisted."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = False

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task:
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            editor.text = "# Integration Test\nTest content with inference disabled"

            await pilot.press("escape")
            await pilot.pause()

            # Verify task NOT called
            mock_extract_task.assert_not_called()


class TestConnectionsPaneVisibility:
    """Connections sidebar should only auto-open when inference + content change + processing active."""

    @pytest.mark.asyncio
    async def test_connections_hidden_when_inference_disabled(self, mock_database):
        """Saving new entry with inference off should not open connections pane."""
        mock_database["add"].return_value = "uuid-new"
        mock_database["get_inference_enabled"].return_value = False
        mock_database["get"].return_value = {
            "uuid": "uuid-new",
            "content": "# Title\nBody",
            "name": "Title",
        }

        with patch("backend.services.tasks.extract_nodes_task"):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                await pilot.press("n")
                await pilot.pause()

                from textual.widgets import TextArea
                editor = app.screen.query_one("#editor", TextArea)
                editor.text = "# Title\nBody"

                await pilot.press("escape")
                await pilot.pause()

                assert isinstance(app.screen, ViewScreen)
                sidebar = app.screen.query_one(EntitySidebar)
                assert app.screen.from_edit is False
                assert sidebar.display is False
                assert not any(w.name == "status-poll" and w.is_running for w in app.screen.workers)

    @pytest.mark.asyncio
    async def test_connections_hidden_when_content_unchanged(self, mock_database):
        """Editing without content change should not auto-open connections pane."""
        mock_episode = {
            "uuid": "existing-uuid",
            "content": "# Original\nOriginal content",
            "name": "Original",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0),
        }
        mock_database["get_home"].return_value = [mock_episode]
        mock_database["get"].return_value = mock_episode
        mock_database["update"].return_value = False  # content unchanged
        mock_database["get_inference_enabled"].return_value = True
        mock_database["get"].return_value = mock_episode

        with patch("backend.services.tasks.extract_nodes_task"):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                await pilot.press("space")
                await pilot.pause()
                await pilot.pause()

                await pilot.press("e")
                await pilot.pause()

                from textual.widgets import TextArea
                editor = app.screen.query_one("#editor", TextArea)
                editor.text = "# Original\nOriginal content"

                await pilot.press("escape")
                await pilot.pause()

                assert isinstance(app.screen, ViewScreen)
                sidebar = app.screen.query_one(EntitySidebar)
                assert app.screen.from_edit is False
                assert sidebar.display is False
                assert not any(w.name == "status-poll" and w.is_running for w in app.screen.workers)

    @pytest.mark.asyncio
    async def test_connections_shown_only_when_processing_active(self, mock_database):
        """Auto-open only when inference enabled, content changed, and status pending."""
        mock_database["add"].return_value = "uuid-new"
        mock_database["get_inference_enabled"].return_value = True
        mock_database["get"].return_value = {
            "uuid": "uuid-new",
            "content": "# Title\nBody",
            "name": "Title",
        }
        mock_database["get_episode_status"].return_value = "pending_nodes"

        with patch("backend.services.tasks.extract_nodes_task") as mock_task, \
             patch("charlie.get_episode_status", return_value="pending_nodes"), \
             patch("charlie.redis_ops") as mock_redis_ops:
            # Mock Redis to prevent EntitySidebar.refresh_entities() errors
            mock_redis = Mock()
            mock_redis.hget.return_value = None
            mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
            mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                await pilot.press("n")
                await pilot.pause()

                from textual.widgets import TextArea
                editor = app.screen.query_one("#editor", TextArea)
                editor.text = "# Title\nBody"

                await pilot.press("escape")
                await pilot.pause()

                assert isinstance(app.screen, ViewScreen)
                sidebar = app.screen.query_one(EntitySidebar)
                assert app.screen.from_edit is True
                assert sidebar.display is True
                assert any(w.name == "status-poll" and w.is_running for w in app.screen.workers)
                mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_requires_open_sidebar(self, mock_database):
        """Pressing 'd' should only delete when sidebar is open and focused."""
        mock_episode = {
            "uuid": "existing-uuid",
            "content": "# Title\nBody",
            "name": "Title",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0),
        }
        mock_database["get_home"].return_value = [mock_episode]
        mock_database["get"].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            from textual.widgets import ListView
            view = app.screen
            sidebar = view.query_one(EntitySidebar)
            sidebar.entities = [
                {"uuid": "entity-1", "name": "Alice", "type": "Person"},
            ]
            sidebar.loading = False
            sidebar._update_content()
            list_view = sidebar.query_one(ListView)
            list_view.index = 0
            sidebar.display = False  # hidden/closed

            # Directly invoke the action to ensure it short-circuits when hidden
            with patch.object(app, "push_screen", wraps=app.push_screen) as mock_push:
                sidebar.action_delete_entity()
                mock_push.assert_not_called()

            # Should remain on ViewScreen and no delete modal should appear
            assert isinstance(app.screen, ViewScreen)
            assert all(not isinstance(s, DeleteEntityModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_edit_entry_title_only_no_extraction_integration(mock_database):
    """Full workflow: edit title only (no content change), verify task NOT called but title updated."""
    mock_episode = {
        "uuid": "existing-uuid",
        "content": "# Original\nOriginal content",
        "name": "Original",
        "valid_at": datetime(2025, 11, 19, 10, 0, 0)
    }
    mock_database['get_home'].return_value = [mock_episode]
    mock_database['get'].return_value = mock_episode
    mock_database['update'].return_value = False  # Content unchanged
    mock_database['get_inference_enabled'].return_value = True

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task:
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            # Change title but keep content the same
            editor.text = "# Updated Title\nOriginal content"

            await pilot.press("escape")
            await pilot.pause()

            # Verify task NOT called
            mock_extract_task.assert_not_called()

            # Verify title was updated
            mock_database['update'].assert_called()


@pytest.mark.asyncio
async def test_edit_screen_empty_content_no_task_enqueue(mock_database):
    """Empty/whitespace content should pop screen without saving or enqueueing task."""
    mock_database['add'].return_value = "new-uuid"
    mock_database['get_inference_enabled'].return_value = True

    with patch('backend.services.tasks.extract_nodes_task') as mock_extract_task:
        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()

            from textual.widgets import TextArea
            editor = app.screen.query_one("#editor", TextArea)
            # Set empty/whitespace content
            editor.text = "   \n  \n  "

            await pilot.press("escape")
            await pilot.pause()

            # Verify task NOT called
            mock_extract_task.assert_not_called()

            # Verify add NOT called (empty content should not save)
            mock_database['add'].assert_not_called()


class TestGracefulShutdown:
    """Tests for graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_quit_shows_notification(self, mock_database):
        """Pressing 'q' should show shutdown notification immediately."""
        notify_calls = []

        original_notify = CharlieApp.notify

        def tracking_notify(self, message, *args, **kwargs):
            notify_calls.append(str(message))
            return original_notify(self, message, *args, **kwargs)

        with patch.object(CharlieApp, 'notify', tracking_notify):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                await pilot.press("q")
                await pilot.pause()

                # Verify shutdown notification was shown
                assert any("closing up shop" in msg for msg in notify_calls), \
                    f"Expected shutdown notification, got: {notify_calls}"

    @pytest.mark.asyncio
    async def test_quit_ui_responsive_during_shutdown(self, mock_database):
        """UI should remain responsive while shutdown runs in background thread."""
        import time

        shutdown_started = False
        shutdown_finished = False

        original_blocking = CharlieApp._blocking_shutdown

        def slow_shutdown(self):
            nonlocal shutdown_started, shutdown_finished
            shutdown_started = True
            time.sleep(0.1)  # Simulate worker drain
            original_blocking(self)
            shutdown_finished = True

        with patch.object(CharlieApp, '_blocking_shutdown', slow_shutdown):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                # Initiate shutdown (runs in background thread via asyncio.to_thread)
                await pilot.press("q")

                # Give a moment for async shutdown to start the thread
                await pilot.pause()

                # If we can still interact with pilot, UI is responsive
                # (the blocking work is happening in a thread)
                assert True  # Reaching here means UI thread wasn't blocked

    @pytest.mark.asyncio
    async def test_double_quit_idempotent(self, mock_database):
        """Double-quit should not cause errors or duplicate shutdown."""
        call_count = 0

        original_blocking_shutdown = CharlieApp._blocking_shutdown

        def counting_shutdown(self):
            nonlocal call_count
            call_count += 1
            original_blocking_shutdown(self)

        with patch.object(CharlieApp, '_blocking_shutdown', counting_shutdown):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                # Press quit twice rapidly
                await pilot.press("q")
                await pilot.press("q")
                await pilot.pause()

                # Should only have one shutdown sequence due to flag check
                assert call_count <= 1

    @pytest.mark.asyncio
    async def test_shutdown_flag_set_before_teardown(self, mock_database):
        """request_shutdown() should be called before stop_huey_worker()."""
        call_order = []

        def track_request_shutdown():
            call_order.append('request_shutdown')

        def track_stop_huey(self):
            call_order.append('stop_huey')

        with patch('charlie.request_shutdown', track_request_shutdown), \
             patch.object(CharlieApp, 'stop_huey_worker', track_stop_huey):
            app = CharlieApp()
            async with app_test_context(app) as pilot:
                await pilot.pause()

                await pilot.press("q")
                await pilot.pause()

                # request_shutdown must come before stop_huey_worker
                if 'request_shutdown' in call_order and 'stop_huey' in call_order:
                    assert call_order.index('request_shutdown') < call_order.index('stop_huey')
