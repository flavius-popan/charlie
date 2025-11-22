"""Tests for EntitySidebar widget."""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListView, LoadingIndicator
from charlie import EntitySidebar


class EntitySidebarTestApp(App):
    """Test app for EntitySidebar."""

    def __init__(self, episode_uuid: str = "test-uuid", journal: str = "test"):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal

    def compose(self) -> ComposeResult:
        yield EntitySidebar(episode_uuid=self.episode_uuid, journal=self.journal)


@pytest.mark.asyncio
async def test_entity_sidebar_shows_loading_initially():
    """Should show loading state on mount."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        # Should be in loading state
        assert sidebar.loading is True


@pytest.mark.asyncio
async def test_entity_sidebar_shows_loading_indicator_when_loading():
    """Should show LoadingIndicator when loading."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.loading = True
        sidebar._render_content()

        content = sidebar.query_one("#entity-content")
        loading_indicator = content.query_one(LoadingIndicator)

        assert loading_indicator is not None


@pytest.mark.asyncio
async def test_entity_sidebar_displays_entities():
    """Should display entities in ListView when loaded."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            {"uuid": "uuid-2", "name": "Central Park", "type": "Place"},
        ]
        sidebar.loading = False

        list_view = sidebar.query_one(ListView)
        assert list_view is not None

        items = list(list_view.children)
        assert len(items) == 2


@pytest.mark.asyncio
async def test_entity_sidebar_formats_entity_labels():
    """Should format entities as 'Name [Type]'."""
    app = EntitySidebarTestApp()

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            {"uuid": "uuid-2", "name": "Park", "type": "Entity"},
        ]
        sidebar.loading = False

        await pilot.pause()

        list_view = sidebar.query_one(ListView)
        items = list(list_view.children)

        from charlie import EntityListItem
        item1_label = items[0].label_text if isinstance(items[0], EntityListItem) else ""
        item2_label = items[1].label_text if isinstance(items[1], EntityListItem) else ""

        assert "Sarah" in item1_label
        assert "[Person]" in item1_label

        assert "Park" in item2_label
        assert "[Entity]" in item2_label


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_entities():
    """Should fetch entities from Redis and update state."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_entities = [
        {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
    ]

    mock_redis = MagicMock()
    mock_redis.hget.side_effect = lambda key, field: (
        json.dumps(mock_entities).encode() if field == "nodes" else None
    )

    with patch("charlie.redis_ops") as mock_redis_ops:
        mock_redis_ops.return_value.__enter__.return_value = mock_redis
        mock_redis_ops.return_value.__exit__.return_value = None

        async with app.run_test():
            sidebar = app.query_one(EntitySidebar)
            await sidebar.refresh_entities()

            assert sidebar.loading is False
            assert sidebar.entities == mock_entities


@pytest.mark.asyncio
async def test_entity_sidebar_shows_delete_confirmation():
    """Pressing 'd' on entity should show confirmation modal."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
        ]
        sidebar.loading = False

        list_view = sidebar.query_one(ListView)
        list_view.focus()

        await pilot.press("d")

        modal = app.screen
        assert isinstance(modal, ModalScreen)


@pytest.mark.asyncio
async def test_entity_sidebar_deletes_entity():
    """Confirming deletion should remove entity from list."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    with patch("charlie.delete_entity_mention", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = False

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
                {"uuid": "uuid-2", "name": "Park", "type": "Place"},
            ]
            sidebar.loading = False

            list_view = sidebar.query_one(ListView)
            list_view.focus()
            list_view.index = 0

            await pilot.press("d")
            modal = app.screen
            remove_button = modal.query_one("#remove", Button)
            remove_button.press()

            await asyncio.sleep(0.1)

            mock_delete.assert_called_once_with("test-uuid", "uuid-1", "test")

            assert len(sidebar.entities) == 1
            assert sidebar.entities[0]["name"] == "Park"


@pytest.mark.asyncio
async def test_entity_sidebar_auto_selects_first_item():
    """Should auto-select first entity when entities load."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)

        # Verify index is None initially (no selection)
        await pilot.pause()

        # Now set entities
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            {"uuid": "uuid-2", "name": "Park", "type": "Place"},
        ]
        sidebar.loading = False

        await pilot.pause()

        list_view = sidebar.query_one(ListView)
        # The bug is that index might still be None - user has to manually press arrow keys
        # After fix, index should be 0 automatically
        assert list_view.index == 0, f"Expected index 0 but got {list_view.index}"
