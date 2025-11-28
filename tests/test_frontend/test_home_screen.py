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
         patch("frontend.screens.home_screen.ensure_database_ready", new_callable=AsyncMock) as mock_ensure, \
         patch("frontend.screens.home_screen.get_entry_entities", new_callable=AsyncMock) as mock_entry_entities, \
         patch("frontend.screens.home_screen.get_period_entities", new_callable=AsyncMock) as mock_period_entities:
        mock_get_home.return_value = []
        mock_ensure.return_value = None
        mock_entry_entities.return_value = []
        mock_period_entities.return_value = {"entry_count": 0, "connection_count": 0, "top_entities": []}
        yield {
            "get_home": mock_get_home,
            "ensure": mock_ensure,
            "get_entry_entities": mock_entry_entities,
            "get_period_entities": mock_period_entities,
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


@pytest.mark.asyncio
async def test_navigating_entries_updates_period_on_boundary_crossing(mock_home_db):
    """Temporal pane should update when navigating across period boundaries."""
    from datetime import timedelta, timezone

    now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
    this_week_start = now - timedelta(days=now.weekday())
    last_week = this_week_start - timedelta(days=3)

    mock_episodes = [
        {
            "uuid": "this-week-entry",
            "name": "This Week Entry",
            "preview": "Entry from this week",
            "valid_at": now - timedelta(days=1),
        },
        {
            "uuid": "last-week-entry",
            "name": "Last Week Entry",
            "preview": "Entry from last week",
            "valid_at": last_week,
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        home_screen = app.screen
        assert isinstance(home_screen, HomeScreen)
        assert len(home_screen.periods) == 2
        assert home_screen.periods[0]["label"] == "This Week"
        assert home_screen.periods[1]["label"] == "Last Week"

        # Initially should be on first period (This Week)
        assert home_screen.selected_period_index == 0

        # Navigate down to Last Week entry (skip divider, skip This Week entry, land on Last Week)
        await pilot.press("down")  # Move to Last Week divider (skipped)
        await pilot.press("down")  # Move to Last Week entry
        await pilot.pause()

        # Period should have updated to Last Week
        assert home_screen.selected_period_index == 1


@pytest.mark.asyncio
async def test_pane_titles_show_key_hints(mock_home_db):
    """Pane titles should display numerical key hints."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_period_entities"].return_value = {
        "entry_count": 1,
        "connection_count": 0,
        "top_entities": [],
    }

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.containers import Container
        entries_pane = app.screen.query_one("#entries-pane", Container)
        connections_pane = app.screen.query_one("#connections-pane", Container)
        temporal_pane = app.screen.query_one("#temporal-pane", Container)

        assert "(1)" in entries_pane.border_title
        assert "(2)" in connections_pane.border_title
        assert "(3)" in temporal_pane.border_title


@pytest.mark.asyncio
async def test_numerical_key_focuses_entries_list(mock_home_db):
    """Pressing '1' should focus the entries list."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        entries_list = app.screen.query_one("#episodes-list", ListView)

        # Press 1 to focus entries
        await pilot.press("1")
        await pilot.pause()

        assert entries_list.has_focus
        # First item should be selected
        assert entries_list.index == 1  # Index 1 because index 0 is the divider


@pytest.mark.asyncio
async def test_numerical_key_focuses_connections_list(mock_home_db):
    """Pressing '2' should focus the connections list when it has items."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        connections_list = app.screen.query_one("#connections-list", ListView)

        # Press 2 to focus connections
        await pilot.press("2")
        await pilot.pause()

        assert connections_list.has_focus
        # First item should be selected
        assert connections_list.index == 0


@pytest.mark.asyncio
async def test_numerical_key_focuses_temporal_list(mock_home_db):
    """Pressing '3' should focus the temporal list when it has items."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_period_entities"].return_value = {
        "entry_count": 1,
        "connection_count": 2,
        "top_entities": [{"name": "Entity 1"}, {"name": "Entity 2"}],
    }

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        temporal_list = app.screen.query_one("#temporal-list", ListView)

        # Press 3 to focus temporal
        await pilot.press("3")
        await pilot.pause()

        assert temporal_list.has_focus
        # First item should be selected
        assert temporal_list.index == 0


@pytest.mark.asyncio
async def test_position_memory_preserved_on_focus_switch(mock_home_db):
    """List position should be remembered when switching focus between panes."""
    mock_episodes = [
        {
            "uuid": "entry-1",
            "name": "Entry 1",
            "preview": "Content 1",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        },
        {
            "uuid": "entry-2",
            "name": "Entry 2",
            "preview": "Content 2",
            "valid_at": datetime(2025, 11, 26, 10, 0, 0),
        },
        {
            "uuid": "entry-3",
            "name": "Entry 3",
            "preview": "Content 3",
            "valid_at": datetime(2025, 11, 25, 10, 0, 0),
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        entries_list = app.screen.query_one("#episodes-list", ListView)

        # Navigate to a specific entry (down twice from divider)
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        original_index = entries_list.index

        # Switch to connections pane
        await pilot.press("2")
        await pilot.pause()

        # Switch back to entries pane
        await pilot.press("1")
        await pilot.pause()

        # Position should be preserved
        assert entries_list.index == original_index


@pytest.mark.asyncio
async def test_entity_list_navigation_does_not_change_entry_selection(mock_home_db):
    """Navigating in connections/temporal lists should not change selected entry."""
    mock_episodes = [
        {
            "uuid": "entry-1",
            "name": "Entry 1",
            "preview": "Content 1",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
        {"name": "Entity 3"},
    ]

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        home_screen = app.screen
        original_entry_uuid = home_screen.selected_entry_uuid
        original_period_index = home_screen.selected_period_index

        # Focus connections list
        await pilot.press("2")
        await pilot.pause()

        # Navigate within connections list
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()

        # Entry selection should NOT have changed
        assert home_screen.selected_entry_uuid == original_entry_uuid
        assert home_screen.selected_period_index == original_period_index


@pytest.mark.asyncio
async def test_entity_list_selection_does_not_trigger_view(mock_home_db):
    """Pressing enter on entity list should not push ViewScreen."""
    mock_episodes = [
        {
            "uuid": "entry-1",
            "name": "Entry 1",
            "preview": "Content 1",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        # Focus connections list
        await pilot.press("2")
        await pilot.pause()

        # Press enter on entity
        await pilot.press("enter")
        await pilot.pause()

        # Should still be on HomeScreen, not ViewScreen
        assert isinstance(app.screen, HomeScreen)


@pytest.mark.asyncio
async def test_entry_formatting_uses_bold_date(mock_home_db):
    """Entry labels should use bold Rich markup for dates."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from frontend.screens.home_screen import EntryLabel
        entry_labels = app.screen.query(EntryLabel)
        assert len(entry_labels) == 1

        label = entry_labels[0]
        # The text should contain Rich markup for bold
        assert "[bold]" in label._text
        assert "[/bold]" in label._text


@pytest.mark.asyncio
async def test_entry_formatting_no_dot_separator(mock_home_db):
    """Entry labels should not use dot separator between date and preview."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from frontend.screens.home_screen import EntryLabel
        entry_labels = app.screen.query(EntryLabel)
        assert len(entry_labels) == 1

        label = entry_labels[0]
        # Should NOT contain the old dot separator
        assert " · " not in label._text


@pytest.mark.asyncio
async def test_focus_on_empty_connections_list_does_nothing(mock_home_db):
    """Pressing '2' when connections list is empty should not crash."""
    mock_episodes = [
        {
            "uuid": "123",
            "name": "Test Entry",
            "preview": "Test content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    mock_home_db["get_entry_entities"].return_value = []  # No entities

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        entries_list = app.screen.query_one("#episodes-list", ListView)
        connections_list = app.screen.query_one("#connections-list", ListView)

        # Press 2 to try to focus empty connections
        await pilot.press("2")
        await pilot.pause()

        # Connections list should NOT have focus (it's empty/hidden)
        assert not connections_list.has_focus
        # Entries list should retain focus
        assert entries_list.has_focus


@pytest.mark.asyncio
async def test_index_bounds_validation_after_list_shrinks(mock_home_db):
    """Saved index should be validated when list has fewer items."""
    mock_episodes = [
        {
            "uuid": "entry-1",
            "name": "Entry 1",
            "preview": "Content 1",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        },
        {
            "uuid": "entry-2",
            "name": "Entry 2",
            "preview": "Content 2",
            "valid_at": datetime(2025, 11, 26, 10, 0, 0),
        },
    ]
    mock_home_db["get_home"].return_value = mock_episodes
    # Start with many entities
    mock_home_db["get_entry_entities"].return_value = [
        {"name": f"Entity {i}"} for i in range(10)
    ]

    app = HomeScreenTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        from textual.widgets import ListView
        connections_list = app.screen.query_one("#connections-list", ListView)

        # Focus connections and navigate to high index
        await pilot.press("2")
        await pilot.pause()
        for _ in range(5):
            await pilot.press("down")
        await pilot.pause()

        saved_index = connections_list.index
        assert saved_index >= 5

        # Now reduce the entity list
        mock_home_db["get_entry_entities"].return_value = [{"name": "Only One"}]

        # Switch away and back
        await pilot.press("1")
        await pilot.pause()

        # Trigger connections refresh by navigating entries (which updates selected_entry_uuid)
        # Need to navigate to a different entry to trigger the refresh
        await pilot.press("down")
        await pilot.pause()

        # Now try to focus connections again
        await pilot.press("2")
        await pilot.pause()

        # Should not crash, and index should be valid (0 for single item list)
        assert connections_list.index == 0
