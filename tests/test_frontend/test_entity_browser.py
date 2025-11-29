"""Unit tests for EntityBrowserScreen widget."""

import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime
from textual.app import App
from textual.widgets import Header, Footer, ListView

from frontend.screens.entity_browser_screen import EntityBrowserScreen


class EntityBrowserTestApp(App):
    """Lightweight test app for EntityBrowserScreen unit tests."""

    def __init__(self, entity_uuid: str = "test-uuid"):
        super().__init__()
        self.entity_uuid = entity_uuid

    def on_mount(self):
        self.push_screen(EntityBrowserScreen(self.entity_uuid))


def make_mock_entity_data():
    """Create mock entity browser data with multiple entries."""
    return {
        "entity": {
            "uuid": "test-uuid",
            "name": "Suecia",
            "first_mention": datetime(2025, 9, 1),
            "last_mention": datetime(2025, 9, 25),
            "entry_count": 5,
        },
        "entries": [
            {
                "episode_uuid": "ep-1",
                "content": "Had a great video call with Suecia today.",
                "valid_at": datetime(2025, 9, 25, 21, 15),
            },
            {
                "episode_uuid": "ep-2",
                "content": "Suecia mentioned she's coming to visit next month.",
                "valid_at": datetime(2025, 9, 20, 14, 30),
            },
            {
                "episode_uuid": "ep-3",
                "content": "Talked to Suecia about the project deadline.",
                "valid_at": datetime(2025, 9, 15, 10, 0),
            },
            {
                "episode_uuid": "ep-4",
                "content": "Suecia and I went hiking in the mountains.",
                "valid_at": datetime(2025, 9, 10, 8, 0),
            },
            {
                "episode_uuid": "ep-5",
                "content": "First met Suecia at the conference.",
                "valid_at": datetime(2025, 9, 1, 16, 45),
            },
        ],
        "connections": [
            {"uuid": "conn-1", "name": "Ryan", "count": 3, "sample_content": "..."},
            {"uuid": "conn-2", "name": "Roci", "count": 2, "sample_content": "..."},
            {"uuid": "conn-3", "name": "Tom", "count": 1, "sample_content": "..."},
        ],
    }


@pytest.fixture
def mock_entity_db():
    """Mock database operations for EntityBrowserScreen tests."""
    with patch(
        "frontend.screens.entity_browser_screen.get_entity_browser_data",
        new_callable=AsyncMock
    ) as mock_get:
        mock_get.return_value = make_mock_entity_data()
        yield mock_get


@pytest.fixture
def mock_get_entry_entities():
    """Mock get_entry_entities_with_counts for reader content loading."""
    with patch(
        "frontend.screens.entity_browser_screen.get_entry_entities_with_counts",
        new_callable=AsyncMock
    ) as mock:
        mock.return_value = []
        yield mock


@pytest.fixture
def mock_home_db():
    """Mock database operations for HomeScreen."""
    with patch("frontend.screens.home_screen.get_home_screen", new_callable=AsyncMock) as mock_home, \
         patch("frontend.screens.home_screen.ensure_database_ready", new_callable=AsyncMock) as mock_ensure, \
         patch("frontend.screens.home_screen.get_processing_status") as mock_status:
        mock_home.return_value = []
        mock_ensure.return_value = None
        mock_status.return_value = {"status": "idle", "queue_length": 0, "processing": []}
        yield {"get_home": mock_home, "ensure": mock_ensure, "status": mock_status}


# =============================================================================
# RESPONSIVE BEHAVIOR TESTS - Wide vs Narrow screen behavior
# =============================================================================


@pytest.mark.asyncio
async def test_wide_screen_auto_opens_reader(mock_entity_db, mock_get_entry_entities):
    """Wide screen (>= 100 columns) should auto-open reader with first quote."""
    app = EntityBrowserTestApp()
    async with app.run_test(size=(120, 40)) as pilot:  # Wide screen
        await pilot.pause()

        # Reader should be open automatically on wide screen
        assert app.screen.reader_open, "Reader should auto-open on wide screen"


@pytest.mark.asyncio
async def test_narrow_screen_reader_closed_initially(mock_entity_db):
    """Narrow screen (< 100 columns) should not auto-open reader."""
    app = EntityBrowserTestApp()
    async with app.run_test(size=(80, 24)) as pilot:  # Narrow screen
        await pilot.pause()

        # Reader should be closed on narrow screen
        assert not app.screen.reader_open, "Reader should be closed on narrow screen"


@pytest.mark.asyncio
async def test_narrow_screen_enter_opens_reader(mock_entity_db, mock_get_entry_entities):
    """On narrow screen, Enter key should open the reader."""
    app = EntityBrowserTestApp()
    async with app.run_test(size=(80, 24)) as pilot:  # Narrow screen
        await pilot.pause()

        # Reader should be closed initially
        assert not app.screen.reader_open

        # Press Enter to select first quote
        await pilot.press("enter")
        await pilot.pause()

        # Reader should now be open
        assert app.screen.reader_open, "Reader should open when Enter is pressed"


@pytest.mark.asyncio
async def test_narrow_screen_escape_closes_reader_first(mock_entity_db, mock_get_entry_entities):
    """On narrow screen, Escape should close reader first, then pop screen."""
    app = EntityBrowserTestApp()
    async with app.run_test(size=(80, 24)) as pilot:  # Narrow screen
        await pilot.pause()

        # Open reader by pressing Enter
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.reader_open, "Reader should be open after Enter"

        # First escape should close reader, not pop screen
        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.reader_open, "Reader should close on first Escape"

        # Screen should still be EntityBrowserScreen
        assert isinstance(app.screen, EntityBrowserScreen), "Screen should remain EntityBrowserScreen"


@pytest.mark.asyncio
async def test_wide_screen_escape_pops_immediately(mock_entity_db, mock_get_entry_entities):
    """On wide screen, Escape should pop screen (no intermediate reader close)."""
    from textual.screen import Screen

    app = EntityBrowserTestApp()
    async with app.run_test(size=(120, 40)) as pilot:  # Wide screen
        await pilot.pause()

        # Reader should be open automatically
        assert app.screen.reader_open

        # Escape should pop screen directly
        await pilot.press("escape")
        await pilot.pause()

        # Should have popped back to default screen
        assert not isinstance(app.screen, EntityBrowserScreen), "Should have popped EntityBrowserScreen"


@pytest.mark.asyncio
async def test_go_home_returns_to_home_screen(mock_entity_db, mock_home_db):
    """Pressing 'h' should return to HomeScreen without UI corruption.

    Bug: action_go_home() pops screens in a tight loop without yielding
    to the event loop, causing lifecycle race conditions.
    """
    import asyncio
    from contextlib import asynccontextmanager
    from frontend.screens.home_screen import HomeScreen

    class TestAppWithHome(App):
        """Test app that starts with HomeScreen like the real app."""
        def on_mount(self):
            self.push_screen(HomeScreen())

    @asynccontextmanager
    async def home_test_context(app):
        """Context manager to clean up HomeScreen workers."""
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                yield pilot
                # Cancel workers before cleanup
                for screen in app.screen_stack:
                    if isinstance(screen, HomeScreen):
                        for group in ["processing_poll", "entities", "period_stats"]:
                            screen.workers.cancel_group(screen, group)
                        for _ in range(10):
                            await pilot.pause()
        except asyncio.CancelledError:
            pass

    app = TestAppWithHome()
    async with home_test_context(app) as pilot:
        # Should start on HomeScreen
        assert isinstance(app.screen, HomeScreen), "Should start on HomeScreen"
        initial_stack_len = len(app.screen_stack)

        # Push EntityBrowserScreen
        app.push_screen(EntityBrowserScreen("test-uuid"))
        await pilot.pause()

        # Should now be on EntityBrowserScreen
        assert isinstance(app.screen, EntityBrowserScreen), "Should be on EntityBrowserScreen"
        assert len(app.screen_stack) == initial_stack_len + 1

        # Press 'h' to go home
        await pilot.press("h")
        await pilot.pause()

        # Should be back on HomeScreen
        assert isinstance(app.screen, HomeScreen), "Should return to HomeScreen"
        assert len(app.screen_stack) == initial_stack_len, "Stack should be back to original size"


# =============================================================================
# CONTENT TESTS - Basic display verification
# =============================================================================


@pytest.mark.asyncio
async def test_entity_browser_displays_entity_name(mock_entity_db):
    """Should display the entity name in the header."""
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        name_widget = app.screen.query_one("#entity-name")
        assert "Suecia" in name_widget.render().plain


@pytest.mark.asyncio
async def test_entity_browser_displays_quotes(mock_entity_db):
    """Should display quote entries in the ListView."""
    from frontend.screens.entity_browser_screen import QuoteListItem
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        quotes_list = app.screen.query_one("#quotes-list", ListView)
        quote_items = [c for c in quotes_list.children if isinstance(c, QuoteListItem)]
        assert len(quote_items) == 5


@pytest.mark.asyncio
async def test_entity_browser_displays_connections(mock_entity_db):
    """Should display connection links in header."""
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        connections = app.screen.query_one("#quotes-footer")
        # Should have 3 links + 2 separators = 5 children
        assert len(connections.children) == 5
