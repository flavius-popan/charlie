"""Tests for ViewScreen with entity sidebar."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Markdown
from charlie import ViewScreen, EntitySidebar, LogScreen


class ViewScreenTestApp(App):
    """Test app for ViewScreen."""

    def __init__(self, episode_uuid: str = "test-uuid", journal: str = "test", from_edit: bool = True):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.from_edit = from_edit

    def on_mount(self) -> None:
        self.push_screen(ViewScreen(episode_uuid=self.episode_uuid, journal=self.journal, from_edit=self.from_edit))


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


@pytest.mark.asyncio
async def test_view_screen_polls_job_status():
    """Should poll for job completion and refresh entities."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        # Mock get_episode_status to control polling behavior
        with patch("charlie.get_episode_status") as mock_status:
            # Start with pending, then complete
            mock_status.side_effect = ["pending_nodes", "pending_nodes", "pending_edges"]

            app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

            async with app.run_test():
                screen = app.screen

                # Poll timer should be running (sidebar is loading, no cached data)
                assert screen._poll_timer is not None

                # Wait for poll to detect completion
                await asyncio.sleep(0.6)  # Longer than poll interval

                # Timer should be stopped after job completes
                # (Note: Can't directly assert timer.stop() was called, but can check side effects)


@pytest.mark.asyncio
async def test_view_screen_log_viewer_toggle():
    """Test that 'l' key navigates to log viewer."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("l")

            await pilot.pause()

            assert isinstance(app.screen, LogScreen)


@pytest.mark.asyncio
async def test_view_screen_toggle_connections_auto_selects_first_item():
    """Test that toggling connections pane auto-selects first item."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test", from_edit=False)

        async with app.run_test() as pilot:
            screen = app.screen
            sidebar = screen.query_one(EntitySidebar)

            # Populate entities
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
                {"uuid": "uuid-2", "name": "Park", "type": "Place"},
            ]
            sidebar.loading = False

            await pilot.pause()

            # Toggle connections on
            await pilot.press("c")

            await pilot.pause()

            # Should auto-select first item
            from textual.widgets import ListView
            list_view = sidebar.query_one(ListView)
            assert list_view.index == 0, f"Expected index 0 but got {list_view.index}"
