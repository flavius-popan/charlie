"""Unit tests for HomeScreen widget.

These tests use a lightweight test app that mounts HomeScreen directly,
avoiding the overhead of full CharlieApp initialization.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
from textual.app import App
from textual.widgets import Header, Footer

from frontend.screens.home_screen import HomeScreen


class HomeScreenTestApp(App):
    """Lightweight test app for HomeScreen unit tests."""

    def __init__(self, episodes=None):
        super().__init__()
        self.test_episodes = episodes or []

    def on_mount(self):
        self.push_screen(HomeScreen())


@pytest.fixture
def mock_home_db():
    """Mock database operations for HomeScreen tests."""
    with patch("frontend.screens.home_screen.get_home_screen", new_callable=AsyncMock) as mock_get_home, \
         patch("frontend.screens.home_screen.ensure_database_ready", new_callable=AsyncMock) as mock_ensure:
        mock_get_home.return_value = []
        mock_ensure.return_value = None
        yield {
            "get_home": mock_get_home,
            "ensure": mock_ensure,
        }


@pytest.mark.asyncio
async def test_home_screen_shows_empty_state(mock_home_db):
    """Should display empty state when no episodes exist."""
    mock_home_db["get_home"].return_value = []

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        empty_state = app.screen.query_one("#empty-state")
        assert empty_state is not None
        assert "No entries yet" in empty_state.render().plain


@pytest.mark.asyncio
async def test_home_screen_displays_header(mock_home_db):
    """Should display Header."""
    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        headers = app.screen.query(Header)
        assert len(headers) > 0


@pytest.mark.asyncio
async def test_home_screen_has_footer(mock_home_db):
    """Should display Footer with key bindings."""
    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        footers = app.screen.query(Footer)
        assert len(footers) > 0


@pytest.mark.asyncio
async def test_home_screen_loads_episodes(mock_home_db):
    """Should load and display episodes from database."""
    mock_episodes = [
        {
            "uuid": "123",
            "content": "# Test Entry\nContent",
            "name": "Test Entry",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        home_screen = app.screen
        assert isinstance(home_screen, HomeScreen)
        assert len(home_screen.episodes) == 1


@pytest.mark.asyncio
async def test_home_screen_displays_episode_list(mock_home_db):
    """Should render episode entries in ListView."""
    mock_episodes = [
        {
            "uuid": "123",
            "content": "# First Entry\nContent",
            "name": "First Entry",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        },
        {
            "uuid": "456",
            "content": "# Second Entry\nMore content",
            "name": "Second Entry",
            "valid_at": datetime(2025, 11, 18, 9, 0, 0)
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        list_view = app.screen.query_one(ListView)
        assert list_view is not None
        # Should have 3 items: 1 period divider + 2 entries
        items = list(list_view.children)
        assert len(items) == 3
        # First item is divider (disabled), next two are entries
        assert items[0].disabled is True
        assert items[1].disabled is False
        assert items[2].disabled is False
