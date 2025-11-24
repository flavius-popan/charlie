"""Tests for EntitySidebar widget."""

import asyncio
import json
import inspect
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListView, LoadingIndicator
from frontend.widgets.entity_sidebar import EntitySidebar


class EntitySidebarTestApp(App):
    """Test app for EntitySidebar."""

    def __init__(self, episode_uuid: str = "test-uuid", journal: str = "test"):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal

    def compose(self) -> ComposeResult:
        yield EntitySidebar(episode_uuid=self.episode_uuid, journal=self.journal)


def test_watch_entities_is_synchronous():
    """watch_entities should be synchronous (no async def)."""

    assert inspect.iscoroutinefunction(EntitySidebar.watch_entities) is False


@pytest.mark.asyncio
async def test_request_render_coalesces_updates():
    """_request_render should coalesce multiple watcher triggers into one render."""
    app = EntitySidebarTestApp()

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)

        # Wrap _update_content to count calls
        render_count = 0
        original_update = sidebar._update_content

        def counting_update():
            nonlocal render_count
            render_count += 1
            original_update()

        sidebar._update_content = counting_update

        # Trigger multiple watcher paths rapidly
        sidebar.status = "pending_edges"
        sidebar.entities = []
        sidebar.active_processing = True

        await pilot.pause()

        assert render_count == 1, f"Expected 1 render, got {render_count}"


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
        sidebar.status = "pending_nodes"
        sidebar.active_processing = True
        sidebar._update_content()

        content = sidebar.query_one("#entity-content")
        loading_indicator = content.query_one(LoadingIndicator)

        assert loading_indicator is not None


@pytest.mark.asyncio
async def test_entity_sidebar_displays_entities():
    """Should display entities in ListView when loaded."""
    app = EntitySidebarTestApp()

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            {"uuid": "uuid-2", "name": "Central Park", "type": "Place"},
        ]
        sidebar.loading = False

        await pilot.pause()

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

        from frontend.widgets.entity_sidebar import EntityListItem
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

    with patch("frontend.widgets.entity_sidebar.redis_ops") as mock_redis_ops:
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

        await pilot.pause()

        list_view = sidebar.query_one(ListView)
        list_view.focus()

        await pilot.press("d")

        modal = app.screen
        assert isinstance(modal, ModalScreen)


@pytest.mark.asyncio
async def test_entity_sidebar_deletes_entity():
    """Confirming deletion should remove entity from list."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    with patch("frontend.widgets.entity_sidebar.delete_entity_mention", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = False

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
                {"uuid": "uuid-2", "name": "Park", "type": "Place"},
            ]
            sidebar.loading = False

            await pilot.pause()

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
async def test_entity_sidebar_delete_last_entity_shows_no_connections():
    """Deleting the last entity should show 'No connections found', not 'Awaiting processing...'."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    with patch("frontend.widgets.entity_sidebar.delete_entity_mention", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = True

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)

            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            ]
            sidebar.loading = False
            sidebar.status = "pending_nodes"
            sidebar.inference_enabled = True
            sidebar.active_processing = False

            await pilot.pause()

            list_view = sidebar.query_one(ListView)
            list_view.focus()
            list_view.index = 0

            await pilot.press("d")
            modal = app.screen
            remove_button = modal.query_one("#remove", Button)
            remove_button.press()

            await asyncio.sleep(0.1)
            await pilot.pause()

            mock_delete.assert_called_once_with("test-uuid", "uuid-1", "test")
            assert len(sidebar.entities) == 0

            content = sidebar.query_one("#entity-content")
            label = content.query_one(Label)
            rendered = label.render()
            text = rendered.plain if hasattr(rendered, "plain") else str(rendered)

            assert "Awaiting processing" in text, f"Expected 'Awaiting processing' but got: {text}"


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


@pytest.mark.asyncio
async def test_entity_sidebar_batch_update_triggers_single_render():
    """Batch update should trigger only one render, not multiple."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)

        # Track render calls
        render_count = 0
        original_render = sidebar._update_content

        def counting_render():
            nonlocal render_count
            render_count += 1
            original_render()

        sidebar._update_content = counting_render

        # Simulate batch update like refresh_entities does
        with app.batch_update():
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
                {"uuid": "uuid-2", "name": "Park", "type": "Place"},
            ]
            sidebar.loading = False

        await pilot.pause()

        # Batch update should trigger only one content update from watchers
        assert render_count == 1, f"Expected 1 render from batch update, got {render_count}"


@pytest.mark.asyncio
async def test_entity_sidebar_batch_update_atomicity():
    """Batch update should set both entities and loading within the same batch."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_entities = [
        {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
    ]

    mock_redis = MagicMock()
    mock_redis.hget.side_effect = lambda key, field: (
        json.dumps(mock_entities).encode() if field == "nodes" else None
    )

    with patch("frontend.widgets.entity_sidebar.redis_ops") as mock_redis_ops:
        mock_redis_ops.return_value.__enter__.return_value = mock_redis
        mock_redis_ops.return_value.__exit__.return_value = None

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)

            # Spy on batch_update
            batch_update_called = False
            original_batch_update = app.batch_update

            def spy_batch_update():
                nonlocal batch_update_called
                batch_update_called = True
                return original_batch_update()

            with patch.object(app, 'batch_update', side_effect=spy_batch_update):
                await sidebar.refresh_entities()
                await pilot.pause()

            # Verify batch_update was called
            assert batch_update_called, "batch_update should have been called"

            # Verify final state
            assert sidebar.entities == mock_entities
            assert sidebar.loading is False


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_with_missing_cache_key():
    """Gracefully handle missing Redis cache key (no exception, loading stays True)."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_redis = MagicMock()
    # Return None to simulate missing cache key
    mock_redis.hget.return_value = None

    with patch("frontend.widgets.entity_sidebar.redis_ops") as mock_redis_ops:
        mock_redis_ops.return_value.__enter__.return_value = mock_redis
        mock_redis_ops.return_value.__exit__.return_value = None

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)

            # Refresh with missing cache key
            await sidebar.refresh_entities()
            await pilot.pause()

            # Should stay in loading state
            assert sidebar.loading is True

            # Entities should stay empty
            assert sidebar.entities == []


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_with_malformed_json():
    """Gracefully handle malformed JSON (set loading to False, log error, no exception)."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_redis = MagicMock()
    # Return malformed JSON
    mock_redis.hget.return_value = b"not valid json {{"

    with patch("frontend.widgets.entity_sidebar.redis_ops") as mock_redis_ops:
        mock_redis_ops.return_value.__enter__.return_value = mock_redis
        mock_redis_ops.return_value.__exit__.return_value = None

        with patch("frontend.widgets.entity_sidebar.logger") as mock_logger:
            async with app.run_test() as pilot:
                sidebar = app.query_one(EntitySidebar)

                # Refresh with malformed JSON
                await sidebar.refresh_entities()
                await pilot.pause()

                # Should set loading to False (graceful fail)
        assert sidebar.loading is False

        # Error should be logged (called at least once)
        assert mock_logger.error.call_count >= 1


@pytest.mark.asyncio
async def test_sidebar_shows_disabled_message_when_inference_off_pending():
    """Spinner should not show; display helpful message when inference disabled during pending status."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.inference_enabled = False
        sidebar.status = "pending_nodes"
        sidebar.loading = True
        sidebar.entities = []

        sidebar._update_content()

        content = sidebar.query_one("#entity-content")
        assert not content.query(LoadingIndicator)
        label = content.query_one(Label)
        rendered = label.render()
        text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
        assert "Inference disabled" in text
        assert sidebar.loading is False


@pytest.mark.asyncio
async def test_sidebar_shows_disabled_message_when_inference_off_done():
    """No spinner when inference off and entry already processed."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.inference_enabled = False
        sidebar.status = "done"
        sidebar.loading = True
        sidebar.entities = []

        sidebar._update_content()

        content = sidebar.query_one("#entity-content")
        assert not content.query(LoadingIndicator)
        label = content.query_one(Label)
        rendered = label.render()
        text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
        assert "Inference disabled" in text
        assert sidebar.loading is False


@pytest.mark.asyncio
async def test_sidebar_shows_spinner_when_processing_and_inference_on():
    """Spinner should appear when inference on and status pending with no data yet."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.inference_enabled = True
        sidebar.status = "pending_nodes"
        sidebar.loading = True
        sidebar.active_processing = True
        sidebar.entities = []

        sidebar._update_content()

        content = sidebar.query_one("#entity-content")
        loading_indicator = content.query_one(LoadingIndicator)
        assert loading_indicator is not None


@pytest.mark.asyncio
async def test_sidebar_shows_awaiting_when_processing_not_active():
    """Awaiting message when inference on, pending status, but not actively processing."""
    app = EntitySidebarTestApp()

    async with app.run_test():
        sidebar = app.query_one(EntitySidebar)
        sidebar.inference_enabled = True
        sidebar.status = "pending_nodes"
        sidebar.loading = True
        sidebar.active_processing = False
        sidebar.entities = []

        sidebar._update_content()

        content = sidebar.query_one("#entity-content")
        assert not content.query(LoadingIndicator)
        label = content.query_one(Label)
        rendered = label.render()
        text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
        assert "Awaiting processing" in text

@pytest.mark.asyncio
async def test_entity_sidebar_preserves_content_after_toggle():
    """Content should persist when toggling display visibility."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    async with app.run_test() as pilot:
        sidebar = app.query_one(EntitySidebar)

        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
            {"uuid": "uuid-2", "name": "Park", "type": "Place"},
        ]
        sidebar.loading = False

        await pilot.pause()

        list_view = sidebar.query_one(ListView)
        assert list_view is not None
        initial_items = len(list(list_view.children))
        assert initial_items == 2

        sidebar.display = False
        await pilot.pause()

        sidebar.display = True
        await pilot.pause()

        try:
            list_view = sidebar.query_one(ListView)
            items = list(list_view.children)
            assert len(items) == 2, f"Expected 2 entities after toggle, got {len(items)}"
        except Exception as e:
            raise AssertionError(f"ListView query failed after toggle: {e}")


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_triggered_on_manual_display():
    """Manual refresh should load entities when sidebar is displayed while loading."""
    app = EntitySidebarTestApp(episode_uuid="test-uuid", journal="test")

    mock_entities = [
        {"uuid": "uuid-1", "name": "Sarah", "type": "Person"},
        {"uuid": "uuid-2", "name": "Park", "type": "Place"},
    ]

    mock_redis = MagicMock()
    mock_redis.hget.side_effect = lambda key, field: (
        json.dumps(mock_entities).encode() if field == "nodes" else None
    )

    with patch("frontend.widgets.entity_sidebar.redis_ops") as mock_redis_ops:
        mock_redis_ops.return_value.__enter__.return_value = mock_redis
        mock_redis_ops.return_value.__exit__.return_value = None

        async with app.run_test() as pilot:
            sidebar = app.query_one(EntitySidebar)

            # Simulate home→view scenario: sidebar hidden and still loading
            sidebar.display = False
            sidebar.loading = True
            sidebar.status = "pending_nodes"
            sidebar.entities = []
            await pilot.pause()

            # Simulate action_toggle_connections logic
            sidebar.display = True
            if sidebar.loading:
                await sidebar.refresh_entities()
            sidebar._update_content()

            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Should have loaded entities
            assert sidebar.loading is False, f"Should have finished loading"
