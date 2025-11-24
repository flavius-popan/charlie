"""Integration tests for ViewScreen and SidebarStateMachine coordination.

These tests verify:
- Machine instantiation with correct initial state
- Event routing from ViewScreen to machine (show, hide, status, cache, inference)
- Reactive property synchronization between machine outputs and EntitySidebar
- Worker lifecycle management (start/stop based on should_poll flag)
- Proper handling of inference toggle changes

Uses async Textual test harness with mocked database/Redis operations.
All tests focus on state transitions and event flow, not UI rendering.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from textual.app import App

from frontend.screens.view_screen import ViewScreen
from frontend.state.sidebar_state_machine import SidebarStateMachine


class ViewScreenMachineTestApp(App):
    """Test app for ViewScreen machine integration."""

    def __init__(
        self,
        episode_uuid: str = "test-uuid",
        journal: str = "test",
        from_edit: bool = True,
        initial_status: str | None = None,
        inference_enabled: bool = True,
    ):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.from_edit = from_edit
        self.initial_status = initial_status
        self.initial_inference = inference_enabled

    def on_mount(self) -> None:
        self.push_screen(
            ViewScreen(
                episode_uuid=self.episode_uuid,
                journal=self.journal,
                from_edit=self.from_edit,
                status=self.initial_status,
                inference_enabled=self.initial_inference,
            )
        )


@pytest.mark.asyncio
async def test_machine_instantiation_from_edit():
    """Machine should seed to appropriate state when from_edit=True."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    episode_uuid="test-uuid",
                    from_edit=True,
                    initial_status="pending_nodes",
                    inference_enabled=True,
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen
                    assert isinstance(screen, ViewScreen)

                    # Machine should be instantiated
                    assert screen.sidebar_machine is not None

                    # Machine should be in processing state (from_edit=True seeds to appropriate state)
                    assert screen.sidebar_machine.output.visible is True
                    assert screen.sidebar_machine.output.active_processing is True
                    assert screen.sidebar_machine.output.should_poll is True


@pytest.mark.asyncio
async def test_machine_instantiation_not_from_edit():
    """Machine should stay hidden when from_edit=False."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value=None):
                app = ViewScreenMachineTestApp(
                    episode_uuid="test-uuid",
                    from_edit=False,
                    initial_status=None,
                    inference_enabled=True,
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Machine should exist but be hidden after on_mount
                    assert screen.sidebar_machine is not None
                    assert screen.sidebar_machine.output.visible is False
                    assert screen.sidebar_machine.output.should_poll is False


@pytest.mark.asyncio
async def test_toggle_connections_routes_show_event():
    """Toggling connections should route show event and sync outputs."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value=None):
                app = ViewScreenMachineTestApp(
                    from_edit=False,
                    initial_status=None,
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Initially hidden
                    assert screen.sidebar_machine.output.visible is False

                    # Toggle on
                    screen.action_toggle_connections()
                    await pilot.pause()

                    # Should be visible (but not necessarily should_poll since status is None/done)
                    assert screen.sidebar_machine.output.visible is True


@pytest.mark.asyncio
async def test_toggle_connections_routes_hide_event():
    """Toggling connections off should route hide event."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status="pending_nodes",
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Initially visible (from_edit=True)
                    assert screen.sidebar_machine.output.visible is True

                    # Toggle off
                    screen.action_toggle_connections()
                    await pilot.pause()

                    # Should be hidden
                    assert screen.sidebar_machine.output.visible is False
                    assert screen.sidebar_machine.output.should_poll is False


@pytest.mark.asyncio
async def test_poll_routes_status_pending_nodes():
    """Polling with pending_nodes status should route event to machine."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            # Mock status polling to return pending_nodes
            with patch(
                "backend.database.redis_ops.get_episode_status",
                side_effect=["pending_nodes", "done"],  # Return pending_nodes first, then done
            ):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status=None,  # Start with unknown status
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Initial state should be processing (from_edit seeded)
                    # Now simulate polling - the machine should transition
                    # Note: We can't directly test the poll worker without a full async test,
                    # but we can verify that the machine routes events correctly
                    screen.sidebar_machine.send("status_pending_nodes", status="pending_nodes")
                    screen._sync_machine_output()

                    assert screen.sidebar_machine.output.status == "pending_nodes"
                    assert screen.sidebar_machine.output.active_processing is True


@pytest.mark.asyncio
async def test_poll_routes_status_done():
    """Polling with done status should update machine and stop processing."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status="pending_nodes",
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Transition to done status
                    screen.sidebar_machine.send("status_pending_edges_or_done", status="done")
                    screen._sync_machine_output()

                    # Should no longer be actively processing
                    assert screen.sidebar_machine.output.status == "done"
                    assert screen.sidebar_machine.output.active_processing is False


@pytest.mark.asyncio
async def test_worker_lifecycle_based_on_should_poll():
    """Worker should be controlled by should_poll flag."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status="pending_nodes",
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # With pending_nodes, should_poll should be True
                    assert screen.sidebar_machine.output.should_poll is True

                    # Transition to ready (no entities)
                    screen.sidebar_machine.send("cache_empty")
                    screen._sync_machine_output()

                    # Now should_poll should be False
                    assert screen.sidebar_machine.output.should_poll is False


@pytest.mark.asyncio
async def test_inference_disabled_routes_event():
    """Disabling inference should route event to machine and show disabled message."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status="pending_nodes",
                    inference_enabled=True,
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # Initially in processing_nodes state with inference enabled
                    assert screen.sidebar_machine.current_state == screen.sidebar_machine.processing_nodes

                    # Manually send inference_disabled event to simulate detection
                    screen.sidebar_machine.send("inference_disabled")
                    screen._sync_machine_output()

                    # Machine should be in disabled state with disabled message
                    assert screen.sidebar_machine.current_state == screen.sidebar_machine.disabled
                    assert screen.sidebar_machine.output.message is not None
                    assert "Inference disabled" in screen.sidebar_machine.output.message


@pytest.mark.asyncio
async def test_sync_machine_output_updates_reactives():
    """_sync_machine_output should update ViewScreen reactive properties."""
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value="pending_nodes"):
                app = ViewScreenMachineTestApp(
                    from_edit=True,
                    initial_status="pending_nodes",
                )

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen

                    # from_edit=True should seed the machine to processing_nodes
                    assert screen.status == "pending_nodes"
                    # processing_nodes state should have active_processing=True
                    assert screen.active_processing is True

                    # Transition to ready state and verify sync
                    screen.sidebar_machine.send("cache_entities_found")
                    screen._sync_machine_output()

                    # Now in ready_entities state, so active_processing should be False
                    assert screen.active_processing is False


@pytest.mark.asyncio
async def test_action_back_with_hidden_sidebar():
    """Test that pressing back with hidden sidebar doesn't raise TransitionNotAllowed.

    Regression test for bug where episode_closed event was sent even when
    sidebar already hidden, causing TransitionNotAllowed exception.
    """
    with patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test content",
        }

        with patch("frontend.screens.view_screen.get_inference_enabled", return_value=True):
            with patch("frontend.screens.view_screen.get_episode_status", return_value=None):
                # Create app with from_edit=False so sidebar starts hidden
                app = ViewScreenMachineTestApp(from_edit=False)

                async with app.run_test() as pilot:
                    await pilot.pause()
                    screen = app.screen
                    assert isinstance(screen, ViewScreen)

                    # Sidebar should be hidden when from_edit=False
                    assert not screen.sidebar_machine.output.visible
                    assert screen.sidebar_machine.current_state == screen.sidebar_machine.hidden

                    # Pressing back should NOT raise TransitionNotAllowed
                    # (this would fail before the fix)
                    screen.action_back()

                    # Should have successfully popped the screen
                    # (we're back at the home screen now)
                    await pilot.pause()
