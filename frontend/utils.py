"""Utility functions for Charlie TUI frontend."""

import re
from datetime import datetime, timedelta, timezone


def rough_token_estimate(text: str) -> int:
    """Estimate token count using average of char/4 and words*1.33."""
    return int(((len(text) / 4) + (len(text.split()) * 1.33)) / 2)


def _extract_preview_from_content(content: str, max_chars: int) -> str:
    """Extract preview from content - markdown title or first line."""
    lines = content.strip().split("\n")

    # Look for markdown title
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                if len(title) <= max_chars:
                    return title
                return title[:max_chars].rstrip(".!? ") + "..."

    # No title found - use first non-empty line
    for line in lines:
        stripped = line.strip()
        if stripped:
            if len(stripped) <= max_chars:
                return stripped
            return stripped[:max_chars].rstrip(".!? ") + "..."

    return "Untitled"


def get_display_title(episode: dict, max_chars: int = 50) -> str:
    """Get a short, display-ready title.

    Tries in order: preview field, name field, extracted from content, "Untitled".
    """
    source = episode.get("preview") or episode.get("name")

    if not source and episode.get("content"):
        return _extract_preview_from_content(episode["content"], max_chars)

    if not source:
        return "Untitled"

    source = source.strip()
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


def inject_entity_links(
    content: str,
    entities: list[dict],
    min_mentions: int = 1,
    exclude_uuids: set[str] | None = None,
) -> str:
    """Inject entity:// links into markdown content.

    Wraps entity name mentions with markdown links pointing to entity:// protocol.
    Handles possessives, case-insensitive matching, and avoids double-linking.

    Args:
        content: Raw markdown text
        entities: List of dicts with 'name', 'uuid', and optionally 'mention_count' keys
        min_mentions: Minimum mention count to create a link (default 1).
                      Entities with fewer mentions are skipped. Requires
                      'mention_count' field in entity dicts.
        exclude_uuids: Set of entity UUIDs to skip (e.g., current entity being viewed)

    Returns:
        Markdown with injected entity links
    """
    if not entities:
        return content

    result = content
    exclude_uuids = exclude_uuids or set()

    # Filter by minimum mentions if specified and mention_count is available
    if min_mentions > 1:
        entities = [
            e for e in entities
            if e.get("mention_count", min_mentions) >= min_mentions
        ]

    # Filter out excluded entities
    entities = [e for e in entities if e["uuid"] not in exclude_uuids]

    if not entities:
        return content

    # Sort by name length descending to avoid substring issues
    # ("Bobby" should be matched before "Bob")
    sorted_entities = sorted(entities, key=lambda e: len(e["name"]), reverse=True)

    for entity in sorted_entities:
        name = entity["name"]
        uuid = entity["uuid"]

        # Skip very short names (likely to cause false positives)
        if len(name) < 2:
            continue

        # Build pattern that:
        # - Uses word boundaries
        # - Handles possessives ('s, ')
        # - Is case-insensitive
        # - Avoids text already in markdown links [text](url)
        escaped_name = re.escape(name)

        # Pattern: word boundary + name + optional possessive + word boundary
        # Negative lookbehind to avoid already-linked text
        # (?<!\[) - not preceded by [
        # (?!\]) - not followed by ]
        # (?!\() - not followed by ( (would be part of link URL)
        pattern = (
            r"(?<!\[)"  # Not after [
            r"(?<![/])"  # Not after / (in URLs)
            r"\b(" + escaped_name + r")"  # Capture the name
            r"(?:'s|')?"  # Optional possessive (not captured)
            r"\b"
            r"(?!\])"  # Not before ]
            r"(?!\()"  # Not before (
        )

        def make_replacement(match: re.Match, _uuid: str = uuid) -> str:
            matched_text = match.group(0)
            # Preserve the original case and any possessive
            return f"[{matched_text}](entity://{_uuid})"

        result = re.sub(pattern, make_replacement, result, flags=re.IGNORECASE)

    return result


def emphasize_rich(content: str, text: str) -> str:
    """Wrap occurrences of text with Rich bold markup ([bold]text[/bold]).

    For use with Textual Static widgets that support Rich markup.

    Args:
        content: Raw text
        text: The text to emphasize (case-insensitive matching)

    Returns:
        Text with Rich bold markup applied
    """
    if not text or len(text) < 2:
        return content

    escaped_text = re.escape(text)

    pattern = (
        r"\b(" + escaped_text + r")"
        r"(?:'s|')?"  # Optional possessive
        r"\b"
    )

    def make_replacement(match: re.Match) -> str:
        matched_text = match.group(0)
        return f"[bold]{matched_text}[/bold]"

    return re.sub(pattern, make_replacement, content, flags=re.IGNORECASE)


def emphasize_text(content: str, text: str) -> str:
    """Wrap occurrences of text with markdown bold (**text**).

    Args:
        content: Raw markdown text
        text: The text to emphasize (case-insensitive matching)

    Returns:
        Markdown with text wrapped in bold
    """
    if not text or len(text) < 2:
        return content

    escaped_text = re.escape(text)

    # Pattern similar to inject_entity_links but simpler
    # Avoid already-bold text or text in links
    pattern = (
        r"(?<!\*\*)"  # Not after **
        r"(?<!\[)"  # Not after [
        r"(?<![/])"  # Not after / (in URLs)
        r"\b(" + escaped_text + r")"
        r"(?:'s|')?"  # Optional possessive
        r"\b"
        r"(?!\*\*)"  # Not before **
        r"(?!\])"  # Not before ]
    )

    def make_replacement(match: re.Match) -> str:
        matched_text = match.group(0)
        return f"**{matched_text}**"

    return re.sub(pattern, make_replacement, content, flags=re.IGNORECASE)
