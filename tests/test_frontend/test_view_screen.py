"""Tests for ViewScreen with entity sidebar."""

import pytest
from unittest.mock import patch, AsyncMock
from textual.app import App, ComposeResult
from textual.widgets import Markdown
from charlie import ViewScreen, EntitySidebar


class ViewScreenTestApp(App):
    """Test app for ViewScreen."""

    def __init__(self, episode_uuid: str = "test-uuid", journal: str = "test"):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal

    def on_mount(self) -> None:
        self.push_screen(ViewScreen(episode_uuid=self.episode_uuid, journal=self.journal))


@pytest.mark.asyncio
async def test_view_screen_has_sidebar():
    """Should contain EntitySidebar in horizontal layout."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test():
            screen = app.screen
            # Should have EntitySidebar
            sidebar = screen.query_one(EntitySidebar)
            assert sidebar is not None
            assert sidebar.episode_uuid == "test-uuid"
            assert sidebar.journal == "test"


@pytest.mark.asyncio
async def test_view_screen_has_markdown_viewer():
    """Should still have Markdown widget for journal content."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test():
            screen = app.screen
            markdown = screen.query_one(Markdown)
            assert markdown is not None
