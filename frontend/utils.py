"""Utility functions for Charlie TUI frontend."""

from datetime import datetime, timedelta, timezone


def get_display_title(episode: dict, max_chars: int = 50) -> str:
    """Get a short, display-ready title without inspecting full content."""

    source = (episode.get("preview") or episode.get("name") or "Untitled").strip()
    if len(source) <= max_chars:
        return source

    trimmed = source[:max_chars].rstrip()
    if not trimmed.endswith("..."):
        trimmed = trimmed.rstrip(".!? ") + "..."
    return trimmed


def _normalize_datetime(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _get_week_start(dt: datetime) -> datetime:
    """Get Monday 00:00 of the week containing dt."""
    days_since_monday = dt.weekday()
    monday = dt - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_period_label(valid_at: datetime, now: datetime) -> str:
    """Determine the period label for a given date."""
    valid_at = _normalize_datetime(valid_at)
    now = _normalize_datetime(now)

    this_week_start = _get_week_start(now)
    last_week_start = this_week_start - timedelta(days=7)

    if valid_at >= this_week_start:
        return "This Week"
    elif valid_at >= last_week_start:
        return "Last Week"
    else:
        return valid_at.strftime("%B %Y")


def group_entries_by_period(
    episodes: list[dict], now: datetime | None = None
) -> list[tuple[str, list[dict]]]:
    """Group episodes by time period for display.

    Args:
        episodes: List of episode dicts with 'valid_at' datetime field
        now: Reference datetime (defaults to current UTC time)

    Returns:
        List of (period_label, episodes) tuples in chronological order (newest first)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if not episodes:
        return []

    groups: dict[str, list[dict]] = {}
    period_order: list[str] = []

    for episode in episodes:
        valid_at = episode.get("valid_at")
        if valid_at is None:
            continue

        label = _get_period_label(valid_at, now)
        if label not in groups:
            groups[label] = []
            period_order.append(label)
        groups[label].append(episode)

    return [(label, groups[label]) for label in period_order]


def calculate_periods(
    episodes: list[dict], now: datetime | None = None
) -> list[dict]:
    """Calculate period boundaries from episodes.

    Args:
        episodes: List of episode dicts with 'valid_at' datetime field
        now: Reference datetime (defaults to current UTC time)

    Returns:
        List of period dicts with:
        - label: Display name (e.g., "This Week", "November 2024")
        - start: Period start datetime (inclusive)
        - end: Period end datetime (exclusive)
        - first_episode_index: Index of first episode in this period
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if not episodes:
        return []

    now = _normalize_datetime(now)
    this_week_start = _get_week_start(now)
    last_week_start = this_week_start - timedelta(days=7)
    next_week_start = this_week_start + timedelta(days=7)

    grouped = group_entries_by_period(episodes, now)
    periods: list[dict] = []
    episode_index = 0

    for label, period_episodes in grouped:
        if label == "This Week":
            start = this_week_start
            end = next_week_start
        elif label == "Last Week":
            start = last_week_start
            end = this_week_start
        else:
            # Monthly period - derive from first episode's date
            first_valid_at = _normalize_datetime(period_episodes[0]["valid_at"])
            start = first_valid_at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # End is first day of next month
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)

        periods.append({
            "label": label,
            "start": start,
            "end": end,
            "first_episode_index": episode_index,
        })
        episode_index += len(period_episodes)

    return periods
