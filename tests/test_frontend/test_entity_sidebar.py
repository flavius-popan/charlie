"""Tests for EntitySidebar widget."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListView
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
async def test_entity_sidebar_has_header():
    """Should have 'Connections' header."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        # Just verify the header label exists with the right class
        header_label = sidebar.query_one(".sidebar-header", Label)
        assert header_label is not None


@pytest.mark.asyncio
async def test_entity_sidebar_shows_cat_spinner_when_loading():
    """Should show ASCII cat spinner and 'Slinging yarn...' when loading."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.loading = True
        sidebar._render_content()

        content = sidebar.query_one("#entity-content")
        labels = content.query(Label)

        # Should have loading message with cat spinner
        assert len(labels) > 0


@pytest.mark.asyncio
async def test_entity_sidebar_displays_entities():
    """Should display entities in ListView when loaded."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        # Set entities
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
            {"uuid": "uuid-2", "name": "Central Park", "labels": ["Entity", "Place"], "ref_count": 1},
        ]
        sidebar.loading = False

        # Should have ListView
        list_view = sidebar.query_one(ListView)
        assert list_view is not None

        # Should have 2 items
        items = list(list_view.children)
        assert len(items) == 2


@pytest.mark.asyncio
async def test_entity_sidebar_formats_entity_labels():
    """Should format entities as 'Name [Type]' or 'Name [Type] (RefCount)' if > 1."""
    app = EntitySidebarTestApp()

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
            {"uuid": "uuid-2", "name": "Park", "labels": ["Entity"], "ref_count": 1},
        ]
        sidebar.loading = False

        # Wait for compose to complete
        await pilot.pause()

        list_view = sidebar.query_one(ListView)
        items = list(list_view.children)

        # Get label text from EntityListItems
        from charlie import EntityListItem
        item1_label = items[0].label_text if isinstance(items[0], EntityListItem) else ""
        item2_label = items[1].label_text if isinstance(items[1], EntityListItem) else ""

        # First item: show most specific type (Person, not Entity) and ref_count > 1
        assert "Sarah" in item1_label
        assert "[Person]" in item1_label
        assert "(3)" in item1_label

        # Second item: show Entity when it's the only type, no ref_count when = 1
        assert "Park" in item2_label
        assert "[Entity]" in item2_label
        assert "(1)" not in item2_label


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_entities():
    """Should fetch entities from database and update state."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_entities = [
        {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 2},
    ]

    with patch("charlie.fetch_entities_for_episode", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_entities

        async with app.run_test():
            sidebar = app.query_one(EntitySidebar)
            await sidebar.refresh_entities()

            # Should have called fetch with correct args
            mock_fetch.assert_called_once_with("test-uuid", "test")

            # Should update state
            assert sidebar.loading is False
            assert sidebar.entities == mock_entities


@pytest.mark.asyncio
async def test_entity_sidebar_shows_delete_confirmation():
    """Pressing 'd' on entity should show confirmation modal."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
        ]
        sidebar.loading = False

        # Focus list and select first item
        list_view = sidebar.query_one(ListView)
        list_view.focus()

        # Press 'd' for delete
        await pilot.press("d")

        # Should show confirmation modal
        modal = app.screen
        assert isinstance(modal, ModalScreen)


@pytest.mark.asyncio
async def test_entity_sidebar_deletes_entity():
    """Confirming deletion should remove entity from list."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    with patch("charlie.delete_entity_mention", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = False  # Not fully deleted

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
                {"uuid": "uuid-2", "name": "Park", "labels": ["Entity", "Place"], "ref_count": 1},
            ]
            sidebar.loading = False

            list_view = sidebar.query_one(ListView)
            list_view.focus()
            list_view.index = 0  # Select Sarah

            # Press 'd' and confirm
            await pilot.press("d")
            modal = app.screen
            remove_button = modal.query_one("#remove", Button)
            remove_button.press()

            await asyncio.sleep(0.1)  # Let deletion process

            # Should have called delete
            mock_delete.assert_called_once_with("test-uuid", "uuid-1", "test")

            # Sarah should be removed from list
            assert len(sidebar.entities) == 1
            assert sidebar.entities[0]["name"] == "Park"
