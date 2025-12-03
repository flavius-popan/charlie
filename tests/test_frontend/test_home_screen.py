"""Unit tests for HomeScreen widget.

These tests use a lightweight test app that mounts HomeScreen directly,
avoiding the overhead of full CharlieApp initialization.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
from textual.app import App
from textual.widgets import Header, Footer

from frontend.screens.home_screen import HomeScreen
from frontend.state.processing_state_machine import ProcessingOutput


def make_test_episodes(count: int, start_offset: int = 0) -> list[dict]:
    """Generate mock episodes for pagination tests."""
    from datetime import timedelta
    base = datetime(2025, 1, 1)
    return [
        {
            "uuid": f"ep-{start_offset + i}",
            "name": f"Episode {start_offset + i}",
            "preview": f"Content {start_offset + i}",
            "valid_at": base - timedelta(days=start_offset + i),
        }
        for i in range(count)
    ]


@asynccontextmanager
async def home_test_context(app):
    """Context manager that cancels HomeScreen workers before cleanup.

    Prevents unraisable exceptions from workers still running when test ends.
    """
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            yield pilot
            # Cancel HomeScreen workers before cleanup
            for screen in app.screen_stack:
                if isinstance(screen, HomeScreen):
                    for group in ["processing_poll", "entities", "period_stats", "load_more"]:
                        screen.workers.cancel_group(screen, group)
                    for _ in range(10):
                        await pilot.pause()
                        if not any(w.is_running for w in screen.workers):
                            break
    except asyncio.CancelledError:
        pass


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
         patch("frontend.screens.home_screen.get_home_screen_paginated", new_callable=AsyncMock) as mock_get_home_paginated, \
         patch("frontend.screens.home_screen.ensure_database_ready", new_callable=AsyncMock) as mock_ensure, \
         patch("frontend.screens.home_screen.get_entry_entities", new_callable=AsyncMock) as mock_entry_entities, \
         patch("frontend.screens.home_screen.get_period_entities", new_callable=AsyncMock) as mock_period_entities, \
         patch("frontend.screens.home_screen.get_episode_status") as mock_episode_status:
        mock_get_home.return_value = []
        mock_get_home_paginated.return_value = ([], False)
        mock_ensure.return_value = None
        mock_entry_entities.return_value = []
        mock_period_entities.return_value = {"entry_count": 0, "connection_count": 0, "top_entities": []}
        mock_episode_status.return_value = None
        yield {
            "get_home": mock_get_home,
            "get_home_paginated": mock_get_home_paginated,
            "ensure": mock_ensure,
            "get_entry_entities": mock_entry_entities,
            "get_period_entities": mock_period_entities,
            "get_episode_status": mock_episode_status,
        }


@pytest.mark.asyncio
async def test_home_screen_shows_empty_state(mock_home_db):
    """Should display empty state when no episodes exist."""
    mock_home_db["get_home_paginated"].return_value = ([], False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
        await pilot.pause()

        empty_state = app.screen.query_one("#empty-state")
        assert empty_state is not None
        assert "No entries yet" in empty_state.render().plain


@pytest.mark.asyncio
async def test_home_screen_displays_header(mock_home_db):
    """Should display Header."""
    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
        await pilot.pause()

        headers = app.screen.query(Header)
        assert len(headers) > 0


@pytest.mark.asyncio
async def test_home_screen_has_footer(mock_home_db):
    """Should display Footer with key bindings."""
    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    import datetime as dt_module

    # Fixed reference time - Thursday Nov 27, 2025
    fixed_now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
    this_week_start = fixed_now - timedelta(days=fixed_now.weekday())
    last_week = this_week_start - timedelta(days=3)

    mock_episodes = [
        {
            "uuid": "this-week-entry",
            "name": "This Week Entry",
            "preview": "Entry from this week",
            "valid_at": fixed_now - timedelta(days=1),
        },
        {
            "uuid": "last-week-entry",
            "name": "Last Week Entry",
            "preview": "Entry from last week",
            "valid_at": last_week,
        },
    ]
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    # Create a datetime wrapper that returns fixed_now for now() but works normally otherwise
    class FrozenDatetime(dt_module.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz:
                return fixed_now.astimezone(tz)
            return fixed_now.replace(tzinfo=None)

    # Patch datetime in frontend.utils so calculate_periods uses our fixed time
    with patch("frontend.utils.datetime", FrozenDatetime):
        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_period_entities"].return_value = {
        "entry_count": 1,
        "connection_count": 0,
        "top_entities": [],
    }

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_period_entities"].return_value = {
        "entry_count": 1,
        "connection_count": 2,
        "top_entities": [{"name": "Entity 1"}, {"name": "Entity 2"}],
    }

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
        {"name": "Entity 3"},
    ]

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = [
        {"name": "Entity 1"},
        {"name": "Entity 2"},
    ]

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = []  # No entities

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    # Start with many entities
    mock_home_db["get_entry_entities"].return_value = [
        {"name": f"Entity {i}"} for i in range(10)
    ]

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
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


@pytest.mark.asyncio
async def test_connections_pane_shows_awaiting_processing_for_pending_entry(mock_home_db):
    """Connections pane should show 'Awaiting processing...' for entries in queue."""
    mock_episodes = [
        {
            "uuid": "pending-entry",
            "name": "Pending Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = []
    mock_home_db["get_episode_status"].return_value = "pending_nodes"

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
        await pilot.pause()

        from textual.widgets import Static
        empty_msg = app.screen.query_one("#connections-empty", Static)
        assert "Awaiting processing" in empty_msg.render().plain


@pytest.mark.asyncio
async def test_connections_pane_shows_no_connections_for_processed_entry(mock_home_db):
    """Connections pane should show 'No connections' for processed entries without entities."""
    mock_episodes = [
        {
            "uuid": "done-entry",
            "name": "Done Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = []
    mock_home_db["get_episode_status"].return_value = "done"

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
        await pilot.pause()

        from textual.widgets import Static
        empty_msg = app.screen.query_one("#connections-empty", Static)
        assert "No connections" in empty_msg.render().plain


@pytest.mark.asyncio
async def test_connections_pane_shows_loading_for_actively_processing_entry(mock_home_db):
    """Connections pane should show LoadingIndicator when entry is being processed."""
    mock_episodes = [
        {
            "uuid": "active-entry",
            "name": "Active Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        }
    ]
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)
    mock_home_db["get_entry_entities"].return_value = []
    mock_home_db["get_episode_status"].return_value = "pending_nodes"

    # Mock get_processing_status to return inferring state
    with patch("frontend.screens.home_screen.get_processing_status") as mock_processing:
        mock_processing.return_value = {
            "model_state": "inferring",
            "active_uuid": "active-entry",
            "pending_count": 0,
            "inference_enabled": True,
        }

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            # Wait for polling loop to update processing_output
            await pilot.pause()

            from textual.widgets import LoadingIndicator
            indicators = app.screen.query(LoadingIndicator)
            assert len(indicators) > 0


@pytest.mark.asyncio
async def test_home_screen_selects_episode_on_resume(mock_home_db):
    """HomeScreen should select episode specified by _select_uuid_on_resume on load."""
    mock_episodes = [
        {
            "uuid": "first-entry",
            "name": "First Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 27, 10, 0, 0),
        },
        {
            "uuid": "second-entry",
            "name": "Second Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 26, 10, 0, 0),
        },
        {
            "uuid": "third-entry",
            "name": "Third Entry",
            "preview": "Content",
            "valid_at": datetime(2025, 11, 25, 10, 0, 0),
        },
    ]
    mock_home_db["get_home_paginated"].return_value = (mock_episodes, False)

    app = HomeScreenTestApp()
    async with home_test_context(app) as pilot:
        await pilot.pause()

        home_screen = app.screen

        # Simulate returning from ViewScreen with a different episode selected
        home_screen._select_uuid_on_resume = "third-entry"
        await home_screen.load_episodes()
        await pilot.pause()

        # Should have selected the third entry
        assert home_screen.selected_entry_uuid == "third-entry"
        # UUID should be cleared after use
        assert home_screen._select_uuid_on_resume is None


# Pagination Tests
class TestHomeScreenPagination:
    """Tests for home screen pagination behavior."""

    @pytest.mark.asyncio
    async def test_initial_load_uses_pagination(self, mock_home_db):
        """Should call get_home_screen_paginated with limit=100, offset=0 on initial load."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(10), False)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            mock_home_db["get_home_paginated"].assert_called()
            call_kwargs = mock_home_db["get_home_paginated"].call_args
            # Initial call should have offset=0 and limit=100
            assert call_kwargs.kwargs.get("offset", 0) == 0

    @pytest.mark.asyncio
    async def test_initial_load_less_than_page_size(self, mock_home_db):
        """Should handle fewer episodes than page size."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(50), False)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            assert len(home_screen.episodes) == 50
            assert home_screen._has_more is False

    @pytest.mark.asyncio
    async def test_has_more_true_allows_loading(self, mock_home_db):
        """When has_more=True, pagination state should indicate more available."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(100), True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            assert home_screen._has_more is True

    @pytest.mark.asyncio
    async def test_has_more_false_stops_loading(self, mock_home_db):
        """When has_more=False, no more loading should occur."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(50), False)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            assert home_screen._has_more is False

    @pytest.mark.asyncio
    async def test_cursor_near_end_triggers_load_more(self, mock_home_db):
        """Cursor within threshold of end should trigger load more."""
        # Initial load: 100 episodes, more available
        initial_episodes = make_test_episodes(100)
        mock_home_db["get_home_paginated"].return_value = (initial_episodes, True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            assert len(home_screen.episodes) == 100

            # Setup for load more call
            more_episodes = make_test_episodes(50, start_offset=100)
            mock_home_db["get_home_paginated"].return_value = (more_episodes, False)

            # Navigate near the end (within threshold of 20)
            home_screen._check_load_more(85)
            await pilot.pause()
            await pilot.pause()

            # Should have called get_home_paginated again
            assert mock_home_db["get_home_paginated"].call_count >= 2

    @pytest.mark.asyncio
    async def test_cursor_away_from_end_no_load(self, mock_home_db):
        """Cursor far from end should not trigger load more."""
        initial_episodes = make_test_episodes(100)
        mock_home_db["get_home_paginated"].return_value = (initial_episodes, True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            initial_call_count = mock_home_db["get_home_paginated"].call_count

            # Navigate to middle of list (far from end)
            home_screen._check_load_more(50)
            await pilot.pause()

            # Should NOT have called get_home_paginated again
            assert mock_home_db["get_home_paginated"].call_count == initial_call_count

    @pytest.mark.asyncio
    async def test_screen_resume_resets_pagination(self, mock_home_db):
        """Screen resume should reset pagination state."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(50), True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            # Simulate some pagination state
            home_screen._current_offset = 100
            home_screen._has_more = False

            # Simulate screen resume
            await home_screen.on_screen_resume()
            await pilot.pause()

            # State should be reset
            assert home_screen._current_offset == len(home_screen.episodes)
            # has_more is set from the return value
            assert home_screen._has_more is True

    @pytest.mark.asyncio
    async def test_load_more_blocked_while_loading(self, mock_home_db):
        """Second load more should be blocked while first is in progress."""
        initial_episodes = make_test_episodes(100)
        mock_home_db["get_home_paginated"].return_value = (initial_episodes, True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen

            # Track call count before setting the flag
            calls_before = mock_home_db["get_home_paginated"].call_count

            # Simulate loading in progress
            home_screen._is_loading_more = True

            # Try to trigger load more
            home_screen._check_load_more(95)
            await pilot.pause()

            # Should not have triggered another load
            assert mock_home_db["get_home_paginated"].call_count == calls_before

    @pytest.mark.asyncio
    async def test_empty_response_stops_loading(self, mock_home_db):
        """Empty response from backend should stop further loading."""
        initial_episodes = make_test_episodes(100)
        mock_home_db["get_home_paginated"].return_value = (initial_episodes, True)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen
            assert home_screen._has_more is True

            # Setup empty response for next call
            mock_home_db["get_home_paginated"].return_value = ([], False)

            # Trigger load more
            home_screen._check_load_more(95)
            await pilot.pause()
            await pilot.pause()

            # has_more should now be False
            assert home_screen._has_more is False

    @pytest.mark.asyncio
    async def test_pagination_state_initialized(self, mock_home_db):
        """HomeScreen should initialize pagination state variables."""
        mock_home_db["get_home_paginated"].return_value = (make_test_episodes(10), False)

        app = HomeScreenTestApp()
        async with home_test_context(app) as pilot:
            await pilot.pause()

            home_screen = app.screen

            # Check pagination state variables exist
            assert hasattr(home_screen, "_has_more")
            assert hasattr(home_screen, "_is_loading_more")
            assert hasattr(home_screen, "_current_offset")
            assert hasattr(home_screen, "_page_size")
            assert hasattr(home_screen, "_pagination_generation")
