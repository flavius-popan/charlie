"""Tests for active episode tracking - determines which post is currently processing."""

from __future__ import annotations

from uuid import uuid4

import pytest

from backend.settings import DEFAULT_JOURNAL
from backend.database.redis_ops import (
    set_episode_status,
    remove_episode_from_queue,
    set_active_episode,
    clear_active_episode,
    is_episode_actively_processing,
)


class TestActiveEpisodeTracking:
    """Test active episode tracking for spinner display."""

    def test_set_and_clear_active_episode(self, falkordb_test_context):
        """Set active episode should be retrievable, clear should remove it."""
        episode_uuid = str(uuid4())

        set_active_episode(episode_uuid, DEFAULT_JOURNAL)
        assert is_episode_actively_processing(episode_uuid) is True

        clear_active_episode()
        assert is_episode_actively_processing(episode_uuid) is False

    def test_is_episode_actively_processing_false_for_different_episode(
        self, falkordb_test_context
    ):
        """Should return False for a different episode."""
        active_episode = str(uuid4())
        other_episode = str(uuid4())

        set_active_episode(active_episode, DEFAULT_JOURNAL)

        assert is_episode_actively_processing(active_episode) is True
        assert is_episode_actively_processing(other_episode) is False

        clear_active_episode()

    def test_is_episode_actively_processing_false_when_no_active(
        self, falkordb_test_context
    ):
        """Should return False when no episode is active."""
        episode_uuid = str(uuid4())
        clear_active_episode()  # Ensure nothing is active

        assert is_episode_actively_processing(episode_uuid) is False


class TestTwoPostsInQueueSpinnerBehavior:
    """Test that only the actively processing post shows spinner."""

    def test_only_active_episode_shows_as_processing(self, falkordb_test_context):
        """When two posts are pending, only the active one should be marked as processing.

        This simulates the scenario where:
        - User saves two journal entries quickly
        - Both are queued with status=pending_nodes
        - Huey worker picks up the first one to process
        - First episode should show spinner, second should show "awaiting"
        """
        episode1 = str(uuid4())
        episode2 = str(uuid4())

        # Both episodes are pending_nodes (queued for processing)
        set_episode_status(episode1, "pending_nodes", DEFAULT_JOURNAL)
        set_episode_status(episode2, "pending_nodes", DEFAULT_JOURNAL)

        # Huey task starts processing episode1
        set_active_episode(episode1, DEFAULT_JOURNAL)

        # Episode1 should show as actively processing (show spinner)
        assert is_episode_actively_processing(episode1) is True
        # Episode2 should NOT show as actively processing (show "awaiting")
        assert is_episode_actively_processing(episode2) is False

        # Cleanup
        clear_active_episode()
        remove_episode_from_queue(episode1, DEFAULT_JOURNAL)
        remove_episode_from_queue(episode2, DEFAULT_JOURNAL)

    def test_second_episode_becomes_active_after_first_completes(
        self, falkordb_test_context
    ):
        """When first episode completes, second becomes active."""
        episode1 = str(uuid4())
        episode2 = str(uuid4())

        set_episode_status(episode1, "pending_nodes", DEFAULT_JOURNAL)
        set_episode_status(episode2, "pending_nodes", DEFAULT_JOURNAL)

        # Episode1 is processed first
        set_active_episode(episode1, DEFAULT_JOURNAL)
        assert is_episode_actively_processing(episode1) is True
        assert is_episode_actively_processing(episode2) is False

        # Episode1 completes, episode2 starts
        clear_active_episode()
        set_active_episode(episode2, DEFAULT_JOURNAL)

        assert is_episode_actively_processing(episode1) is False
        assert is_episode_actively_processing(episode2) is True

        # Cleanup
        clear_active_episode()
        remove_episode_from_queue(episode1, DEFAULT_JOURNAL)
        remove_episode_from_queue(episode2, DEFAULT_JOURNAL)
