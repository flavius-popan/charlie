"""Tests for add_journal_entry queue integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_add_journal_entry_enqueues_when_enabled():
    """Should enqueue extract_nodes_task when inference is enabled."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock) as mock_persist, \
        patch("backend.set_episode_status") as mock_set_status, \
        patch("backend.get_inference_enabled", return_value=True) as mock_get_enabled, \
        patch("backend.services.tasks.extract_nodes_task") as mock_extract_task:

        episode_uuid = await add_journal_entry("Test content", journal="test")

        mock_persist.assert_awaited()
        mock_set_status.assert_called_once()
        mock_get_enabled.assert_called_once()
        mock_extract_task.assert_called_once_with(episode_uuid, "test")


@pytest.mark.asyncio
async def test_add_journal_entry_skips_enqueue_when_disabled():
    """Should not enqueue task when inference is disabled, but still set status."""
    from backend import add_journal_entry

    with patch("backend.persist_episode", new_callable=AsyncMock), \
        patch("backend.set_episode_status") as mock_set_status, \
        patch("backend.get_inference_enabled", return_value=False) as mock_get_enabled, \
        patch("backend.services.tasks.extract_nodes_task") as mock_extract_task:

        await add_journal_entry("Content", journal="default")

        mock_set_status.assert_called_once()
        mock_get_enabled.assert_called_once()
        mock_extract_task.assert_not_called()
