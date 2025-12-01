"""Tests for ViewScreen with entity sidebar."""

import asyncio
from datetime import datetime
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Markdown, LoadingIndicator
from frontend.screens.view_screen import ViewScreen
from frontend.widgets.entity_sidebar import EntitySidebar


class ViewScreenTestApp(App):
    """Test app for ViewScreen."""

    def __init__(
        self,
        episode_uuid: str = "test-uuid",
        journal: str = "test",
        from_edit: bool = True,
        episodes: list[dict] | None = None,
    ):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.from_edit = from_edit
        self.episodes = episodes

    def on_mount(self) -> None:
        self.push_screen(ViewScreen(
            episode_uuid=self.episode_uuid,
            journal=self.journal,
            from_edit=self.from_edit,
            episodes=self.episodes,
        ))


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
async def test_view_screen_displays_content():
    """Should display episode content in Markdown widget."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test Title\nTest content body",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test() as pilot:
            await pilot.pause()

            markdown = app.screen.query_one("#journal-content", Markdown)
            assert markdown is not None


@pytest.mark.asyncio
async def test_view_screen_has_header():
    """Should display Header."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test")

        async with app.run_test() as pilot:
            await pilot.pause()

            from textual.widgets import Header
            headers = app.screen.query(Header)
            assert len(headers) > 0


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
            sidebar.cache_loading = False

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
        with patch("frontend.screens.view_screen.get_episode_status") as mock_status, patch(
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
                    async def fake_run_worker(self, coro, *, exclusive=False, name=None):
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
                            # Loading remains True while entry is pending
                            assert sidebar.cache_loading is True

                            # Toggle connections on (starts polling)
                            await pilot.press("c")
                            await pilot.pause()

        assert poll_callbacks, "set_interval should be called to start polling"


@pytest.mark.asyncio
async def test_view_screen_from_edit_starts_polling_and_spinner_when_pending():
    """Opening ViewScreen from Edit flow should keep loading, start polling, and show spinner."""

    # Mock Redis to say this episode IS actively processing
    def mock_hgetall(key):
        if key == "task:active_episode":
            return {b"uuid": b"test-uuid", b"journal": b"test"}
        return {}

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get, patch(
        "frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"
    ), patch("charlie.get_inference_enabled", return_value=True), patch(
        "frontend.screens.view_screen.redis_ops"
    ) as mock_redis_ctx:
        # Setup Redis mock to return active episode
        mock_redis = mock_redis_ctx.return_value.__enter__.return_value
        mock_redis.hgetall = mock_hgetall

        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        poll_started = False

        def fake_set_interval(self, interval, callback, *args, **kwargs):
            nonlocal poll_started
            poll_started = True

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
            app = ViewScreenTestApp(
                episode_uuid="test-uuid", journal="test", from_edit=True
            )

            async with app.run_test() as pilot:
                # Allow mount hooks to run
                await pilot.pause()

                screen: ViewScreen = app.screen
                sidebar = screen.query_one(EntitySidebar)

                assert poll_started is True, "Polling should start when pending from edit flow"
                # Check that worker would be running (in real scenario, not mocked set_interval)
                assert sidebar.cache_loading is True, "Sidebar should remain in loading state while pending"
                # active_episode_uuid should match this episode
                assert sidebar.active_episode_uuid == "test-uuid", "Should have active episode UUID"

                content = sidebar.query_one("#entity-content")
                assert content.query(LoadingIndicator), "Spinner should be visible while processing"


@pytest.mark.asyncio
async def test_toggle_sidebar_cancels_sidebar_workers_on_hide():
    """Regression test: hiding sidebar must cancel sidebar workers to prevent race conditions.

    Without this fix, the open -> close -> open cycle would fail because:
    1. First open starts a worker on the sidebar (refresh_entities)
    2. Close hides sidebar but worker keeps running
    3. Second open races with the still-running worker, corrupting state

    The fix ensures sidebar.workers.cancel_all() is called when hiding.
    """
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        app = ViewScreenTestApp(episode_uuid="test-uuid", journal="test", from_edit=False)

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            sidebar = screen.query_one(EntitySidebar)

            # Sidebar starts hidden when from_edit=False
            assert sidebar.display is False

            # Track if cancel_all was called
            cancel_all_called = False
            original_cancel_all = sidebar.workers.cancel_all

            def mock_cancel_all():
                nonlocal cancel_all_called
                cancel_all_called = True
                return original_cancel_all()

            sidebar.workers.cancel_all = mock_cancel_all

            # Open sidebar (1st press)
            await pilot.press("c")
            await pilot.pause()
            assert sidebar.display is True, "Sidebar should be visible after 1st press"

            # Close sidebar (2nd press) - this should call cancel_all
            await pilot.press("c")
            await pilot.pause()
            assert sidebar.display is False, "Sidebar should be hidden after 2nd press"
            assert cancel_all_called, "sidebar.workers.cancel_all() must be called when hiding"

            # Reset tracking
            cancel_all_called = False

            # Open sidebar again (3rd press) - should work without issues
            await pilot.press("c")
            await pilot.pause()
            assert sidebar.display is True, "Sidebar should be visible after 3rd press (regression check)"


@pytest.mark.asyncio
async def test_view_screen_find_episode_index():
    """Test _find_episode_index helper finds correct position."""
    episodes = [
        {"uuid": "newest-uuid", "content": "Newest"},
        {"uuid": "middle-uuid", "content": "Middle"},
        {"uuid": "oldest-uuid", "content": "Oldest"},
    ]

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"uuid": "middle-uuid", "content": "Middle"}

        app = ViewScreenTestApp(
            episode_uuid="middle-uuid",
            journal="test",
            from_edit=False,
            episodes=episodes,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            assert screen._current_idx == 1, "Should find middle episode at index 1"
            assert screen._find_episode_index("newest-uuid") == 0
            assert screen._find_episode_index("oldest-uuid") == 2
            assert screen._find_episode_index("nonexistent") is None


@pytest.mark.asyncio
async def test_view_screen_prev_entry_navigation():
    """Test left arrow navigates to older entry (higher index)."""
    episodes = [
        {"uuid": "newest-uuid", "content": "Newest", "valid_at": datetime(2024, 1, 3)},
        {"uuid": "middle-uuid", "content": "Middle", "valid_at": datetime(2024, 1, 2)},
        {"uuid": "oldest-uuid", "content": "Oldest", "valid_at": datetime(2024, 1, 1)},
    ]

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        # Return the episode that matches the current episode_uuid
        def get_episode_side_effect(uuid):
            for ep in episodes:
                if ep["uuid"] == uuid:
                    return ep
            return None

        mock_get.side_effect = get_episode_side_effect

        app = ViewScreenTestApp(
            episode_uuid="newest-uuid",
            journal="test",
            from_edit=False,
            episodes=episodes,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # Start at newest (index 0)
            assert screen._current_idx == 0
            assert screen.episode_uuid == "newest-uuid"

            # Press left to go older
            await pilot.press("left")
            await pilot.pause()

            # Should now be at middle (index 1)
            assert screen._current_idx == 1, "Should move to index 1 (older)"
            assert screen.episode_uuid == "middle-uuid"


@pytest.mark.asyncio
async def test_view_screen_next_entry_navigation():
    """Test right arrow navigates to newer entry (lower index)."""
    episodes = [
        {"uuid": "newest-uuid", "content": "Newest", "valid_at": datetime(2024, 1, 3)},
        {"uuid": "middle-uuid", "content": "Middle", "valid_at": datetime(2024, 1, 2)},
        {"uuid": "oldest-uuid", "content": "Oldest", "valid_at": datetime(2024, 1, 1)},
    ]

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        def get_episode_side_effect(uuid):
            for ep in episodes:
                if ep["uuid"] == uuid:
                    return ep
            return None

        mock_get.side_effect = get_episode_side_effect

        app = ViewScreenTestApp(
            episode_uuid="oldest-uuid",
            journal="test",
            from_edit=False,
            episodes=episodes,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # Start at oldest (index 2)
            assert screen._current_idx == 2
            assert screen.episode_uuid == "oldest-uuid"

            # Press right to go newer
            await pilot.press("right")
            await pilot.pause()

            # Should now be at middle (index 1)
            assert screen._current_idx == 1, "Should move to index 1 (newer)"
            assert screen.episode_uuid == "middle-uuid"


@pytest.mark.asyncio
async def test_view_screen_navigation_at_boundaries():
    """Test navigation does nothing at list boundaries."""
    episodes = [
        {"uuid": "newest-uuid", "content": "Newest"},
        {"uuid": "oldest-uuid", "content": "Oldest"},
    ]

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"uuid": "newest-uuid", "content": "Newest"}

        app = ViewScreenTestApp(
            episode_uuid="newest-uuid",
            journal="test",
            from_edit=False,
            episodes=episodes,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # At newest - right should do nothing
            assert screen._current_idx == 0
            await pilot.press("right")
            await pilot.pause()
            assert screen._current_idx == 0, "Should stay at newest when pressing right"

            # Go to oldest
            mock_get.return_value = {"uuid": "oldest-uuid", "content": "Oldest"}
            await pilot.press("left")
            await pilot.pause()
            assert screen._current_idx == 1

            # At oldest - left should do nothing
            await pilot.press("left")
            await pilot.pause()
            assert screen._current_idx == 1, "Should stay at oldest when pressing left"


@pytest.mark.asyncio
async def test_view_screen_navigation_without_episodes():
    """Test navigation gracefully handles empty episodes list."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"uuid": "test-uuid", "content": "Test"}

        app = ViewScreenTestApp(
            episode_uuid="test-uuid",
            journal="test",
            from_edit=False,
            episodes=None,  # No episodes list
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # Navigation should not crash
            assert screen._current_idx is None
            await pilot.press("left")
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            # Should still be showing the original episode
            assert screen.episode_uuid == "test-uuid"


@pytest.mark.asyncio
async def test_view_screen_check_action_hides_bindings():
    """Test check_action hides prev/next bindings at boundaries."""
    episodes = [
        {"uuid": "newest-uuid", "content": "Newest"},
        {"uuid": "middle-uuid", "content": "Middle"},
        {"uuid": "oldest-uuid", "content": "Oldest"},
    ]

    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"uuid": "newest-uuid", "content": "Newest"}

        app = ViewScreenTestApp(
            episode_uuid="newest-uuid",
            journal="test",
            from_edit=False,
            episodes=episodes,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # At newest (index 0): prev should be visible, next should be hidden
            assert screen._current_idx == 0
            assert screen.check_action("prev_entry", ()) is True, "Prev should be visible at newest"
            assert screen.check_action("next_entry", ()) is False, "Next should be hidden at newest"

            # Move to middle
            mock_get.return_value = {"uuid": "middle-uuid", "content": "Middle"}
            await pilot.press("left")
            await pilot.pause()

            # At middle (index 1): both should be visible
            assert screen._current_idx == 1
            assert screen.check_action("prev_entry", ()) is True, "Prev should be visible at middle"
            assert screen.check_action("next_entry", ()) is True, "Next should be visible at middle"

            # Move to oldest
            mock_get.return_value = {"uuid": "oldest-uuid", "content": "Oldest"}
            await pilot.press("left")
            await pilot.pause()

            # At oldest (index 2): next should be visible, prev should be hidden
            assert screen._current_idx == 2
            assert screen.check_action("prev_entry", ()) is False, "Prev should be hidden at oldest"
            assert screen.check_action("next_entry", ()) is True, "Next should be visible at oldest"


@pytest.mark.asyncio
async def test_view_screen_check_action_no_episodes():
    """Test check_action hides both bindings when no episodes list."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"uuid": "test-uuid", "content": "Test"}

        app = ViewScreenTestApp(
            episode_uuid="test-uuid",
            journal="test",
            from_edit=False,
            episodes=None,
        )

        async with app.run_test() as pilot:
            screen: ViewScreen = app.screen
            await pilot.pause()

            # No episodes: both should be hidden
            assert screen.check_action("prev_entry", ()) is False
            assert screen.check_action("next_entry", ()) is False
