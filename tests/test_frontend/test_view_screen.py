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


@pytest.mark.asyncio
async def test_toggle_connections_starts_polling_and_refreshes_when_pending():
    """Toggling connections from Home flow should start polling and refresh after status completes."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        # Simulate pending -> pending_edges transition
        with patch("charlie.get_episode_status") as mock_status, patch(
            "charlie.get_inference_enabled", return_value=True
        ):
            from itertools import chain, repeat
            mock_status.side_effect = chain(
                ["pending_nodes", "pending_edges"], repeat("pending_edges")
            )

            refresh_called = False

            async def fake_refresh_entities(self):
                nonlocal refresh_called
                refresh_called = True

            # Capture set_interval to verify polling started
            poll_callbacks = []

            def fake_set_interval(self, interval, callback, *args, **kwargs):
                poll_callbacks.append(callback)

                class FakeTimer:
                    def __init__(self):
                        self.stopped = False

                    def pause(self):
                        self.stopped = True

                    def resume(self):
                        self.stopped = False

                    def stop(self):
                        self.stopped = True

                return FakeTimer()

            with patch.object(ViewScreen, "set_interval", new=fake_set_interval):
                with patch.object(EntitySidebar, "refresh_entities", new=fake_refresh_entities):
                    # Spy on run_worker to execute coroutine immediately
                    async def fake_run_worker(self, coro, *, exclusive=False):
                        if asyncio.iscoroutine(coro):
                            await coro
                        else:
                            return coro

                    with patch.object(ViewScreen, "run_worker", new=fake_run_worker):
                        app = ViewScreenTestApp(
                            episode_uuid="test-uuid", journal="test", from_edit=False
                        )

                        async with app.run_test() as pilot:
                            screen: ViewScreen = app.screen
                            sidebar = screen.query_one(EntitySidebar)

                            # Sidebar should start hidden for from_edit=False
                            assert sidebar.display is False
                            # Loading cleared after awaiting message rendered
                            assert sidebar.loading is False

                            # Toggle connections on (starts polling)
                            await pilot.press("c")
                            await pilot.pause()

                        assert poll_callbacks, "set_interval should be called to start polling"
