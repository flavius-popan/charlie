"""Tests for Charlie TUI utility functions."""

from datetime import datetime, timedelta, timezone

import pytest

from frontend.utils import get_display_title, group_entries_by_period


class TestGetDisplayTitle:
    """Tests for get_display_title function."""

    def test_get_display_title_uses_preview(self):
        """Should prefer preview when provided."""
        episode = {"preview": "Preview text", "name": "fallback"}
        assert get_display_title(episode) == "Preview text"

    def test_get_display_title_falls_back_to_name(self):
        """Should use name when preview missing."""
        episode = {"name": "Episode Name"}
        assert get_display_title(episode) == "Episode Name"

    def test_get_display_title_truncates_with_ellipsis(self):
        """Should truncate long preview and add ellipsis."""
        long_preview = "a" * 100
        episode = {"preview": long_preview, "name": "fallback"}
        result = get_display_title(episode, max_chars=50)
        assert result.endswith("...")
        assert len(result) <= 53  # 50 chars plus ellipsis adjustments

    def test_get_display_title_no_truncation_when_short(self):
        """Should not truncate when text fits."""
        episode = {"preview": "Short", "name": "fallback"}
        assert get_display_title(episode, max_chars=50) == "Short"

    def test_get_display_title_handles_missing_fields(self):
        """Should fall back to default when fields absent."""
        episode = {}
        assert get_display_title(episode) == "Untitled"


class TestGroupEntriesByPeriod:
    """Tests for group_entries_by_period function."""

    def test_groups_this_week_entries(self):
        """Entries from current week should be grouped as 'This Week'."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)  # Thursday
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 27, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "2", "valid_at": datetime(2025, 11, 24, 10, 0, 0, tzinfo=timezone.utc)},  # Monday
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 1
        assert result[0][0] == "This Week"
        assert len(result[0][1]) == 2

    def test_groups_last_week_entries(self):
        """Entries from previous week should be grouped as 'Last Week'."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)  # Thursday
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 20, 10, 0, 0, tzinfo=timezone.utc)},  # Last Thursday
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 1
        assert result[0][0] == "Last Week"

    def test_groups_older_entries_by_month(self):
        """Entries older than last week should be grouped by month."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "2", "valid_at": datetime(2025, 9, 5, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 2
        assert result[0][0] == "October 2025"
        assert result[1][0] == "September 2025"

    def test_handles_mixed_periods(self):
        """Should correctly separate entries across multiple periods."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0, tzinfo=timezone.utc)},  # This week
            {"uuid": "2", "valid_at": datetime(2025, 11, 18, 10, 0, 0, tzinfo=timezone.utc)},  # Last week
            {"uuid": "3", "valid_at": datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc)},  # October
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 3
        assert result[0][0] == "This Week"
        assert result[1][0] == "Last Week"
        assert result[2][0] == "October 2025"

    def test_handles_naive_datetimes(self):
        """Should work with timezone-naive datetimes (treated as UTC)."""
        now = datetime(2025, 11, 27, 12, 0, 0)  # naive
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0)},  # naive
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 1
        assert result[0][0] == "This Week"

    def test_handles_mixed_timezone_awareness(self):
        """Should handle mix of naive and aware datetimes."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0)},  # naive
        ]
        result = group_entries_by_period(episodes, now=now)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        """Empty episode list should return empty result."""
        result = group_entries_by_period([])
        assert result == []

    def test_preserves_episode_order_within_groups(self):
        """Episodes within a group should maintain their original order."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "first", "valid_at": datetime(2025, 11, 27, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "second", "valid_at": datetime(2025, 11, 26, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = group_entries_by_period(episodes, now=now)
        assert result[0][1][0]["uuid"] == "first"
        assert result[0][1][1]["uuid"] == "second"
