"""Frontend test fixtures - mocking for UI tests."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def reset_lifecycle_for_frontend_tests():
    """Reset lifecycle state before each frontend test.

    Frontend tests mock the database but some code paths still check
    is_shutdown_requested(). Ensure the flag is False at test start.
    """
    import backend.database.lifecycle as lifecycle
    lifecycle.reset_lifecycle_state()
    yield
    lifecycle.reset_lifecycle_state()


@pytest.fixture(autouse=True)
def patch_backend_for_frontend_tests(reset_lifecycle_for_frontend_tests):
    """Provide default async mocks for backend operations in frontend tests.

    This ensures frontend tests don't accidentally trigger real database ops.
    Frontend tests should use the explicit mock_database fixture for control.

    Also mocks huey consumer to prevent background Redis access from the task queue.
    """
    with patch("backend.services.queue.start_huey_consumer"), \
         patch("backend.services.queue.stop_huey_consumer"), \
         patch("charlie.start_huey_consumer"), \
         patch("charlie.stop_huey_consumer"), \
         patch("backend.add_journal_entry", new_callable=AsyncMock) as mock_add, \
         patch("frontend.screens.edit_screen.add_journal_entry", new_callable=AsyncMock) as screen_add, \
         patch("backend.database.update_episode", new_callable=AsyncMock) as backend_update, \
         patch("frontend.screens.edit_screen.update_episode", new_callable=AsyncMock) as screen_update, \
         patch("frontend.screens.edit_screen.get_episode", new_callable=AsyncMock) as screen_get, \
         patch("frontend.screens.view_screen.get_episode", new_callable=AsyncMock) as view_get:
        mock_add.return_value = "new-uuid"
        screen_add.return_value = "new-uuid"
        backend_update.return_value = True
        screen_update.return_value = True
        screen_get.return_value = {"uuid": "new-uuid", "content": "# Test Entry\nSome content"}
        view_get.return_value = {"uuid": "new-uuid", "content": "# Test Entry\nSome content"}
        yield
