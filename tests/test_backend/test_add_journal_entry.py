"""Tests for add_journal_entry status tracking."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_add_journal_entry_sets_pending_status():
    """Should persist episode and set pending_nodes status."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock) as mock_persist, \
        patch("backend.set_episode_status") as mock_set_status:

        episode_uuid = await add_journal_entry("Test content", journal="test")

        # Verify episode was persisted
        mock_persist.assert_awaited()

        # Verify status was set to pending_nodes (task will be enqueued by caller)
        mock_set_status.assert_called_once_with(episode_uuid, "pending_nodes", journal="test")


@pytest.mark.asyncio
async def test_add_journal_entry_returns_uuid():
    """Should return the episode UUID after creation."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock), \
        patch("backend.set_episode_status"):

        episode_uuid = await add_journal_entry("Content", journal="default")

        # Verify UUID is returned
        assert episode_uuid is not None
        assert isinstance(episode_uuid, str)
