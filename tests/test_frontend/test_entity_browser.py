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
        "sparkline_data": [0, 1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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


@pytest.mark.asyncio
async def test_entity_browser_displays_entity_name(mock_entity_db):
    """Should display the entity name in the header."""
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        name_widget = app.screen.query_one("#entity-name")
        assert "Suecia" in name_widget.render().plain


@pytest.mark.asyncio
async def test_entity_browser_displays_all_quotes(mock_entity_db):
    """Should display all quote entries in the ListView."""
    from frontend.screens.entity_browser_screen import QuoteListItem, DateListItem
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        quotes_list = app.screen.query_one("#quotes-list", ListView)
        # Should have 5 DateListItems + 5 QuoteListItems = 10 total
        quote_items = [c for c in quotes_list.children if isinstance(c, QuoteListItem)]
        date_items = [c for c in quotes_list.children if isinstance(c, DateListItem)]
        assert len(quote_items) == 5
        assert len(date_items) == 5


@pytest.mark.asyncio
async def test_entity_browser_displays_connections(mock_entity_db):
    """Should display connection chips."""
    app = EntityBrowserTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        chips = app.screen.query_one("#connections-chips")
        # Should have 3 connection chips
        assert len(chips.children) == 3


@pytest.mark.asyncio
async def test_entity_browser_layout_visible(mock_entity_db):
    """Debug test to see the actual layout."""
    from frontend.screens.entity_browser_screen import QuoteListItem, DateListItem
    app = EntityBrowserTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Print widget sizes for debugging
        print("\n" + "="*80)
        print("WIDGET SIZES:")
        print("="*80)

        body = app.screen.query_one("#body-container")
        print(f"body-container: size={body.size}, region={body.region}")

        quotes_container = app.screen.query_one("#quotes-container")
        print(f"quotes-container: size={quotes_container.size}, region={quotes_container.region}")

        quotes_list = app.screen.query_one("#quotes-list", ListView)
        print(f"quotes-list: size={quotes_list.size}, region={quotes_list.region}")

        footer = app.screen.query_one("#connections-footer")
        print(f"connections-footer: size={footer.size}, region={footer.region}")

        print(f"\nListView children count: {len(quotes_list.children)}")
        for i, child in enumerate(quotes_list.children):
            print(f"  Child {i}: {child}, size={child.size}, region={child.region}")

        print("="*80)

        # 5 dates + 5 quotes = 10 total items
        quote_items = [c for c in quotes_list.children if isinstance(c, QuoteListItem)]
        assert len(quote_items) == 5, f"Expected 5 quotes, got {len(quote_items)}"
