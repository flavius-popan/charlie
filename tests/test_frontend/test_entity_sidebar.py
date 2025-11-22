"""Tests for EntitySidebar widget."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label, ListView
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
