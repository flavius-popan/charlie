"""Tests for add_journal_entry status tracking."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_add_journal_entry_sets_pending_status():
    """Should persist episode and add to pending queue."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock) as mock_persist, \
        patch("backend.add_pending_episode") as mock_add_pending:

        episode_uuid = await add_journal_entry("Test content", journal="test")

        # Verify episode was persisted
        mock_persist.assert_awaited()

        # Verify added to pending queue (task will be enqueued by caller)
        mock_add_pending.assert_called_once()
        call_args = mock_add_pending.call_args
        assert call_args[0][0] == episode_uuid
        assert call_args[0][1] == "test"


@pytest.mark.asyncio
async def test_add_journal_entry_returns_uuid():
    """Should return the episode UUID after creation."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock), \
        patch("backend.add_pending_episode"):

        episode_uuid = await add_journal_entry("Content", journal="default")

        # Verify UUID is returned
        assert episode_uuid is not None
        assert isinstance(episode_uuid, str)
