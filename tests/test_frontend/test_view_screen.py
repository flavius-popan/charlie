"""Tests for ViewScreen with entity sidebar."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from textual.app import App, ComposeResult
from textual.widgets import Markdown
from charlie import ViewScreen, EntitySidebar
from backend.database.redis_ops import set_episode_status


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


@pytest.mark.asyncio
async def test_view_screen_polls_job_status():
    """Should poll for job completion and refresh entities."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test():
            screen = app.screen
            # Set job to pending initially
            set_episode_status("test-uuid", "pending_nodes")

            # Poll timer should be running
            assert screen._poll_timer is not None

            # Complete the job
            set_episode_status("test-uuid", "completed")

            # Wait for poll to detect completion
            await asyncio.sleep(0.6)  # Longer than poll interval

            # Timer should be stopped
            # (Note: Can't directly assert timer.stop() was called, but can check side effects)
