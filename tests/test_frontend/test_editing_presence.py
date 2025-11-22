"""Tests for EditScreen editing presence detection.

Tests verify that Redis keys are set when users type in TextArea to indicate
active editing sessions, and that these keys are properly cleaned up on exit.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch, Mock

import pytest

from charlie import CharlieApp, EditScreen
from textual.widgets import TextArea


@asynccontextmanager
async def app_test_context(app):
    """Context manager that wraps app.run_test() and suppresses shutdown CancelledError.

    Textual 6.6.0 raises CancelledError during test cleanup when removing screens.
    This is a framework issue that doesn't affect test correctness - the test logic
    completes successfully before the error occurs during cleanup.
    """
    try:
        async with app.run_test() as pilot:
            yield pilot
            await pilot.exit()
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_database():
    """Mock all database operations for testing."""
    with patch('charlie.ensure_database_ready', new_callable=AsyncMock) as mock_ensure, \
         patch('charlie.get_all_episodes', new_callable=AsyncMock) as mock_get_all, \
         patch('charlie.get_episode', new_callable=AsyncMock) as mock_get, \
         patch('charlie.add_journal_entry', new_callable=AsyncMock) as mock_add, \
         patch('charlie.update_episode', new_callable=AsyncMock) as mock_update, \
         patch('charlie.delete_episode', new_callable=AsyncMock) as mock_delete, \
         patch('charlie.shutdown_database') as mock_shutdown, \
         patch('charlie.get_inference_enabled') as mock_get_inference_enabled, \
         patch('charlie.set_inference_enabled') as mock_set_inference_enabled:

        mock_ensure.return_value = None
        mock_get_all.return_value = []
        mock_get_inference_enabled.return_value = False

        yield {
            'ensure': mock_ensure,
            'get_all': mock_get_all,
            'get': mock_get,
            'add': mock_add,
            'update': mock_update,
            'delete': mock_delete,
            'shutdown': mock_shutdown,
            'get_inference_enabled': mock_get_inference_enabled,
            'set_inference_enabled': mock_set_inference_enabled,
        }


class TestEditingPresence:
    """Tests for editing presence detection via Redis keys."""

    @pytest.mark.asyncio
    async def test_text_change_sets_redis_key_with_ttl(self, mock_database):
        """Should set Redis key with TTL when user types in TextArea."""
        from datetime import datetime

        mock_episode = {
            "uuid": "test-episode-uuid",
            "content": "# Original\nOriginal content",
            "name": "Original",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open EditScreen with existing episode
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            mock_redis = Mock()
            with patch('charlie.redis_ops') as mock_redis_ops:
                mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
                mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

                # Type some text to trigger TextArea.Changed event
                await pilot.press("t", "e", "s", "t")
                await pilot.pause()

                # Verify Redis key was set with correct TTL
                assert mock_redis.setex.call_count >= 1, \
                    f"Expected setex to be called at least once, but was called {mock_redis.setex.call_count} times"

                # Verify it was called with editing:active key
                mock_redis.setex.assert_called_with(
                    "editing:active",
                    120,
                    "active"
                )

    @pytest.mark.asyncio
    async def test_key_deleted_on_save_and_return(self, mock_database):
        """Should delete Redis key when user saves and returns."""
        from datetime import datetime

        mock_episode = {
            "uuid": "test-episode-uuid",
            "content": "# Original\nOriginal content",
            "name": "Original",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode
        mock_database['update'].return_value = True

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open EditScreen with existing episode
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            mock_redis = Mock()
            with patch('charlie.redis_ops') as mock_redis_ops:
                mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
                mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

                with patch('backend.services.tasks.extract_nodes_task'):
                    # Save and return
                    await pilot.press("escape")
                    await pilot.pause()

                    # Verify Redis key was deleted
                    assert mock_redis.delete.call_count >= 1, \
                        f"Expected delete to be called at least once, but was called {mock_redis.delete.call_count} times"
                    mock_redis.delete.assert_called_with("editing:active")

    @pytest.mark.asyncio
    async def test_key_deleted_on_unmount(self, mock_database):
        """Should delete Redis key when EditScreen is unmounted."""
        from datetime import datetime

        mock_episode = {
            "uuid": "test-episode-uuid",
            "content": "# Original\nOriginal content",
            "name": "Original",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode
        mock_database['update'].return_value = False

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open EditScreen with existing episode
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            mock_redis = Mock()
            with patch('charlie.redis_ops') as mock_redis_ops:
                mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
                mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

                with patch('backend.services.tasks.extract_nodes_task'):
                    # Navigate back (triggers save_and_return which then pops)
                    await pilot.press("q")
                    await pilot.pause()

                    # Verify Redis key was deleted (called once in action_save_and_return, once in on_unmount)
                    assert mock_redis.delete.call_count >= 1, \
                        f"Expected delete to be called at least once, but was called {mock_redis.delete.call_count} times"
                    mock_redis.delete.assert_called_with("editing:active")

    @pytest.mark.asyncio
    async def test_redis_error_does_not_interrupt_editing(self, mock_database):
        """Should handle Redis errors gracefully without interrupting editing."""
        from datetime import datetime

        mock_episode = {
            "uuid": "test-episode-uuid",
            "content": "# Original\nOriginal content",
            "name": "Original",
            "valid_at": datetime(2025, 11, 19, 10, 0, 0)
        }
        mock_database['get_all'].return_value = [mock_episode]
        mock_database['get'].return_value = mock_episode

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open EditScreen with existing episode
            await pilot.press("space")
            await pilot.pause()
            await pilot.pause()

            await pilot.press("e")
            await pilot.pause()

            mock_redis = Mock()
            mock_redis.setex.side_effect = Exception("Redis connection failed")

            with patch('charlie.redis_ops') as mock_redis_ops:
                mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
                mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

                # Type some text - should not crash despite Redis error
                await pilot.press("t", "e", "s", "t")
                await pilot.pause()

                # Editing should still work
                editor = app.query_one("#editor", TextArea)
                assert "test" in editor.text

    @pytest.mark.asyncio
    async def test_new_entry_sets_editing_active_key(self, mock_database):
        """Should set editing:active key for new entries even without UUID."""
        mock_database['add'].return_value = "new-uuid"

        app = CharlieApp()
        async with app_test_context(app) as pilot:
            await pilot.pause()

            # Open EditScreen for new entry
            await pilot.press("n")
            await pilot.pause()

            mock_redis = Mock()
            with patch('charlie.redis_ops') as mock_redis_ops:
                mock_redis_ops.return_value.__enter__ = Mock(return_value=mock_redis)
                mock_redis_ops.return_value.__exit__ = Mock(return_value=None)

                # Type some text
                await pilot.press("t", "e", "s", "t")
                await pilot.pause()

                # Verify Redis key was set with editing:active
                assert mock_redis.setex.call_count >= 1, \
                    f"Expected setex to be called at least once for new entries, but was called {mock_redis.setex.call_count} times"
                mock_redis.setex.assert_called_with(
                    "editing:active",
                    120,
                    "active"
                )
