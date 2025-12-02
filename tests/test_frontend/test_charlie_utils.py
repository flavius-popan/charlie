"""Tests for Charlie TUI utility functions."""

from datetime import datetime, timedelta, timezone

import pytest

from frontend.utils import calculate_periods, emphasize_rich, get_display_title, group_entries_by_period


class TestEmphasizeRich:
    """Tests for emphasize_rich function."""

    def test_basic_emphasis(self):
        """Should wrap entity name with bold markup."""
        result = emphasize_rich("Hello Bob today", "Bob")
        assert result == "Hello [bold]Bob[/bold] today"

    def test_escapes_brackets_in_content(self):
        """Content with brackets should be escaped to prevent markup errors."""
        result = emphasize_rich("He said [quote] about Bob", "Bob")
        assert "[bold]Bob[/bold]" in result
        # Rich escapes opening bracket as \[ to prevent tag parsing
        assert "\\[quote]" in result

    def test_escapes_rich_tags_in_content(self):
        """Literal Rich tags in content should be escaped."""
        result = emphasize_rich("Text with [/bold] and Bob", "Bob")
        assert "[bold]Bob[/bold]" in result
        # The literal [/bold] in content should be escaped
        assert "\\[/bold]" in result

    def test_short_text_skipped(self):
        """Text shorter than 2 chars should not be emphasized."""
        result = emphasize_rich("A B C", "B")
        assert "[bold]" not in result

    def test_escapes_even_when_no_match(self):
        """Brackets should be escaped even when entity not found."""
        result = emphasize_rich("Some [tag] text", "Bob")
        assert "\\[tag]" in result


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


class TestCalculatePeriods:
    """Tests for calculate_periods function."""

    def test_returns_empty_for_no_episodes(self):
        """Empty episode list should return empty periods."""
        result = calculate_periods([])
        assert result == []

    def test_this_week_period_boundaries(self):
        """This Week period should have correct start/end boundaries."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)  # Thursday
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = calculate_periods(episodes, now=now)

        assert len(result) == 1
        assert result[0]["label"] == "This Week"
        # Week starts Monday Nov 24
        assert result[0]["start"] == datetime(2025, 11, 24, 0, 0, 0, tzinfo=timezone.utc)
        # Week ends Monday Dec 1
        assert result[0]["end"] == datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert result[0]["first_episode_index"] == 0

    def test_last_week_period_boundaries(self):
        """Last Week period should have correct start/end boundaries."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 20, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = calculate_periods(episodes, now=now)

        assert len(result) == 1
        assert result[0]["label"] == "Last Week"
        # Last week starts Monday Nov 17
        assert result[0]["start"] == datetime(2025, 11, 17, 0, 0, 0, tzinfo=timezone.utc)
        # Last week ends Monday Nov 24
        assert result[0]["end"] == datetime(2025, 11, 24, 0, 0, 0, tzinfo=timezone.utc)

    def test_monthly_period_boundaries(self):
        """Monthly periods should span entire month."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = calculate_periods(episodes, now=now)

        assert len(result) == 1
        assert result[0]["label"] == "October 2025"
        assert result[0]["start"] == datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert result[0]["end"] == datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_december_to_january_boundary(self):
        """December period should correctly roll over to January."""
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 12, 20, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = calculate_periods(episodes, now=now)

        assert result[0]["label"] == "December 2025"
        assert result[0]["start"] == datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert result[0]["end"] == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_first_episode_index_tracks_position(self):
        """first_episode_index should track episode positions correctly."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0, tzinfo=timezone.utc)},  # This Week
            {"uuid": "2", "valid_at": datetime(2025, 11, 25, 10, 0, 0, tzinfo=timezone.utc)},  # This Week
            {"uuid": "3", "valid_at": datetime(2025, 11, 18, 10, 0, 0, tzinfo=timezone.utc)},  # Last Week
            {"uuid": "4", "valid_at": datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc)},  # October
        ]
        result = calculate_periods(episodes, now=now)

        assert len(result) == 3
        assert result[0]["first_episode_index"] == 0  # This Week starts at index 0
        assert result[1]["first_episode_index"] == 2  # Last Week starts at index 2
        assert result[2]["first_episode_index"] == 3  # October starts at index 3

    def test_multiple_periods_all_have_boundaries(self):
        """All periods should have start, end, label, and first_episode_index."""
        now = datetime(2025, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        episodes = [
            {"uuid": "1", "valid_at": datetime(2025, 11, 26, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "2", "valid_at": datetime(2025, 11, 18, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "3", "valid_at": datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc)},
            {"uuid": "4", "valid_at": datetime(2025, 9, 5, 10, 0, 0, tzinfo=timezone.utc)},
        ]
        result = calculate_periods(episodes, now=now)

        assert len(result) == 4
        for period in result:
            assert "label" in period
            assert "start" in period
            assert "end" in period
            assert "first_episode_index" in period
            assert period["start"] < period["end"]
