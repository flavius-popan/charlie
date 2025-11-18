"""Backend API for Charlie - journal entry management without extraction."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from graphiti_core.nodes import EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import utc_now

from backend.settings import DEFAULT_JOURNAL
from backend.database import persist_episode

# Content validation limits
MAX_CONTENT_LENGTH = 100_000  # 100k characters


def validate_content(content: str) -> None:
    """Validate journal entry content.

    Args:
        content: Journal entry text to validate

    Raises:
        ValueError: If content is invalid (empty, whitespace-only, or too long)
    """
    if not content:
        raise ValueError("Journal entry content cannot be empty")
    if not content.strip():
        raise ValueError("Journal entry content cannot be whitespace-only")
    if len(content) > MAX_CONTENT_LENGTH:
        raise ValueError(
            f"Content too long: {len(content)} chars (max {MAX_CONTENT_LENGTH:,})"
        )


async def add_journal_entry(
    content: str,
    reference_time: datetime | None = None,
    journal: str | None = None,
    title: str | None = None,
    source_description: str = "Charlie",
    uuid: str | None = None,
) -> str:
    """
    Add a journal entry to the specified journal.

    This function creates and persists a journal entry to FalkorDB. It does NOT
    perform any entity extraction or enrichment. Call enrich_episode() separately
    to trigger graph operations.

    Args:
        content: Journal entry text (required)
        reference_time: When the entry was written (defaults to current time)
        journal: Journal name (defaults to DEFAULT_JOURNAL from settings).
                 Each journal is a separate isolated graph.
        title: Entry title (auto-generated from timestamp if not provided)
        source_description: Origin of entry - "Charlie" for native entries,
                          or "Day One", "Notion", "Obsidian", etc. for imports
        uuid: Pre-existing UUID for imports/backups (auto-generated if None).
             Must be valid UUID format if provided.

    Returns:
        UUID of the persisted journal entry

    Raises:
        ValueError: If content is empty/whitespace-only/too long (>100k chars),
                   journal name is invalid, or uuid format is invalid
        RuntimeError: If persistence fails

    Examples:
        # Basic usage (native entry)
        >>> uuid = await add_journal_entry("Today I went to the park...")

        # Import from Day One with preserved UUID
        >>> uuid = await add_journal_entry(
        ...     content="Imported entry...",
        ...     uuid="550e8400-e29b-41d4-a716-446655440000",
        ...     source_description="Day One"
        ... )

        # Multiple journals
        >>> work_uuid = await add_journal_entry(
        ...     content="Team meeting...",
        ...     journal="work"
        ... )
        >>> personal_uuid = await add_journal_entry(
        ...     content="Dinner with friends...",
        ...     journal="personal"
        ... )
    """
    # Validate content
    validate_content(content)

    # Apply defaults
    journal = journal if journal is not None else DEFAULT_JOURNAL
    reference_time = reference_time or utc_now()

    # Validate journal name (catches empty strings, special chars, etc.)
    from backend.database import validate_journal_name
    validate_journal_name(journal)

    # Validate and generate UUID
    if uuid is not None:
        try:
            UUID(uuid)  # Validate format
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid UUID format: {uuid}") from exc
        episode_uuid = uuid
    else:
        episode_uuid = str(uuid4())

    if title is None:
        title = reference_time.strftime("%A %b %d, %Y").replace(" 0", " ")

    # Create EpisodicNode
    # Note: group_id is set to journal name for graphiti-core compatibility
    episode = EpisodicNode(
        uuid=episode_uuid,
        name=title,
        group_id=journal,
        content=content,
        source=EpisodeType.text,
        source_description=source_description,
        valid_at=reference_time,
        created_at=utc_now(),
        labels=[],
        entity_edges=[],
    )

    # Persist to database (also ensures SELF entity exists)
    await persist_episode(episode, journal=journal)
    return episode_uuid


__all__ = [
    "add_journal_entry",
    "MAX_CONTENT_LENGTH",
    "validate_content",
]
