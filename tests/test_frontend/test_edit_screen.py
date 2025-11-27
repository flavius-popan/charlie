"""Unit tests for EditScreen widget.

These tests use a lightweight test app that mounts EditScreen directly,
avoiding the overhead of full CharlieApp initialization.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock
from textual.app import App
from textual.widgets import Header, TextArea

from frontend.screens.edit_screen import EditScreen


class EditScreenTestApp(App):
    """Lightweight test app for EditScreen unit tests."""

    def __init__(self, episode_uuid=None, episode_data=None):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.episode_data = episode_data or {
            "uuid": episode_uuid or "test-uuid",
            "content": "# Test Entry\nSome content",
            "name": "Test Entry",
        }

    def on_mount(self):
        self.push_screen(EditScreen(episode_uuid=self.episode_uuid))


@pytest.fixture
def mock_edit_db():
    """Mock database operations for EditScreen tests."""
    with patch("frontend.screens.edit_screen.get_episode", new_callable=AsyncMock) as mock_get, \
         patch("frontend.screens.edit_screen.update_episode", new_callable=AsyncMock) as mock_update, \
         patch("frontend.screens.edit_screen.add_journal_entry", new_callable=AsyncMock) as mock_add, \
         patch("frontend.screens.edit_screen.get_inference_enabled", return_value=False), \
         patch("frontend.screens.edit_screen.get_episode_status", return_value=None), \
         patch("frontend.screens.edit_screen.redis_ops") as mock_redis_ops:

        # Mock Redis context manager
        mock_redis = Mock()
        mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

        mock_get.return_value = {
            "uuid": "existing-uuid",
            "content": "# Title\nBody",
            "name": "Title",
        }
        mock_add.return_value = "new-uuid"
        mock_update.return_value = True

        yield {
            "get": mock_get,
            "add": mock_add,
            "update": mock_update,
            "redis": mock_redis,
        }


@pytest.mark.asyncio
async def test_edit_screen_has_text_area(mock_edit_db):
    """Should display TextArea for editing."""
    app = EditScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        editor = app.screen.query_one("#editor", TextArea)
        assert editor is not None


@pytest.mark.asyncio
async def test_edit_screen_has_header(mock_edit_db):
    """Should display Header."""
    app = EditScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        headers = app.screen.query(Header)
        assert len(headers) > 0


@pytest.mark.asyncio
async def test_edit_screen_is_new_entry_flag(mock_edit_db):
    """New entry should have is_new_entry=True."""
    app = EditScreenTestApp(episode_uuid=None)
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.screen.is_new_entry is True


@pytest.mark.asyncio
async def test_edit_screen_existing_entry_flag(mock_edit_db):
    """Existing entry should have is_new_entry=False."""
    app = EditScreenTestApp(episode_uuid="existing-uuid")
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.screen.is_new_entry is False


@pytest.mark.asyncio
async def test_edit_screen_loads_existing_content(mock_edit_db):
    """Should load existing episode content into TextArea."""
    mock_edit_db["get"].return_value = {
        "uuid": "existing-uuid",
        "content": "# Existing Title\nExisting content here",
        "name": "Existing Title",
    }

    app = EditScreenTestApp(episode_uuid="existing-uuid")
    async with app.run_test() as pilot:
        await pilot.pause()

        editor = app.screen.query_one("#editor", TextArea)
        assert "Existing content here" in editor.text


@pytest.mark.asyncio
async def test_edit_screen_new_entry_empty(mock_edit_db):
    """New entry should start with empty TextArea."""
    app = EditScreenTestApp(episode_uuid=None)
    async with app.run_test() as pilot:
        await pilot.pause()

        editor = app.screen.query_one("#editor", TextArea)
        # New entry starts empty or with placeholder
        assert editor.text == "" or editor.text.startswith("#")


@pytest.mark.asyncio
async def test_edit_screen_text_area_focusable(mock_edit_db):
    """TextArea should be focusable for editing."""
    app = EditScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        editor = app.screen.query_one("#editor", TextArea)
        assert editor.can_focus is True
