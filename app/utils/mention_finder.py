"""
Entity mention detection for Charlie.

Finds occurrences of entity names in text content with position tracking
to enable inline highlighting and tooltips.
"""

import re
from typing import List, Set
from dataclasses import dataclass

from app.models.graph import EntityNode


@dataclass
class EntityMention:
    """
    Represents a single mention of an entity in text.

    Attributes:
        entity_uuid: UUID of the mentioned entity
        entity_name: Display name of the entity
        entity_summary: Optional summary for tooltips
        start: Character position where mention starts
        end: Character position where mention ends
        original_text: The actual text that matched (preserves case)
    """

    entity_uuid: str
    entity_name: str
    entity_summary: str
    start: int
    end: int
    original_text: str


def find_entity_mentions(
    content: str, entities: List[EntityNode]
) -> List[EntityMention]:
    """
    Find all occurrences of entity names in content.

    Strategy:
    1. Sort entities by name length (longest first) to avoid partial matches
       Example: Match "John Smith" before "John" to avoid "John" match inside "John Smith"
    2. Use case-insensitive matching but preserve original case
    3. Track character positions for later HTML injection
    4. Avoid overlapping matches

    Args:
        content: The text content to search
        entities: List of entities to find mentions of

    Returns:
        List of EntityMention objects sorted by position
    """
    if not content or not entities:
        return []

    mentions = []
    occupied_ranges: Set[range] = set()

    # Sort by name length (longest first) to prioritize longer matches
    sorted_entities = sorted(entities, key=lambda e: len(e.name), reverse=True)

    for entity in sorted_entities:
        # Escape special regex characters in entity name
        pattern = re.escape(entity.name)

        # Case-insensitive search
        for match in re.finditer(pattern, content, re.IGNORECASE):
            start, end = match.start(), match.end()

            # Check if this position overlaps with existing mention
            if not _overlaps_existing(start, end, occupied_ranges):
                mentions.append(
                    EntityMention(
                        entity_uuid=entity.uuid,
                        entity_name=entity.name,
                        entity_summary=entity.display_summary,
                        start=start,
                        end=end,
                        original_text=content[start:end],
                    )
                )

                # Mark this range as occupied
                occupied_ranges.add(range(start, end))

    # Sort by position for sequential processing
    return sorted(mentions, key=lambda m: m.start)


def _overlaps_existing(start: int, end: int, occupied_ranges: Set[range]) -> bool:
    """
    Check if a position range overlaps with any existing mentions.

    Args:
        start: Start position of new mention
        end: End position of new mention
        occupied_ranges: Set of ranges already occupied by mentions

    Returns:
        True if overlap exists, False otherwise
    """
    new_range = range(start, end)

    for occupied in occupied_ranges:
        # Check for any overlap between ranges
        if start < occupied.stop and end > occupied.start:
            return True

    return False


def generate_excerpt(
    content: str, mention: EntityMention, context_chars: int = 100
) -> str:
    """
    Generate a text excerpt showing context around an entity mention.

    Extracts text before and after the mention, with ellipsis for truncation.
    Useful for views showing where an entity was mentioned.

    Args:
        content: Full text content
        mention: The entity mention to excerpt around
        context_chars: Number of characters to include before/after mention

    Returns:
        Excerpt string with entity mention in context
    """
    # Calculate excerpt boundaries
    excerpt_start = max(0, mention.start - context_chars)
    excerpt_end = min(len(content), mention.end + context_chars)

    # Extract raw excerpt
    excerpt = content[excerpt_start:excerpt_end]

    # Add leading ellipsis if we're not at the start
    if excerpt_start > 0:
        # Find the first space to avoid cutting mid-word
        first_space = excerpt.find(" ")
        if first_space > 0 and first_space < 20:  # Only trim if space is nearby
            excerpt = excerpt[first_space + 1 :]
        excerpt = "..." + excerpt

    # Add trailing ellipsis if we're not at the end
    if excerpt_end < len(content):
        # Find the last space to avoid cutting mid-word
        last_space = excerpt.rfind(" ")
        if last_space > len(excerpt) - 20:  # Only trim if space is nearby
            excerpt = excerpt[:last_space]
        excerpt = excerpt + "..."

    return excerpt.strip()


def highlight_mentions_in_text(content: str, mentions: List[EntityMention]) -> str:
    """
    Inject HTML highlighting into plain text content.

    WARNING: This operates on plain text, not HTML. Use before markdown processing
    or use inject_mentions_in_html() for post-processed HTML.

    Args:
        content: Plain text content
        mentions: List of mentions to highlight

    Returns:
        Text with HTML span tags injected around mentions
    """
    if not mentions:
        return content

    # Process mentions in reverse order to maintain position indices
    result = content
    for mention in reversed(mentions):
        # Build the HTML span with data attributes for tooltips
        span_open = (
            f'<span class="entity-mention" '
            f'data-entity-uuid="{mention.entity_uuid}" '
            f'data-entity-name="{mention.entity_name}" '
            f'data-entity-summary="{_escape_html(mention.entity_summary)}">'
        )
        span_close = "</span>"

        # Replace the mention text with wrapped version
        result = (
            result[: mention.start]
            + span_open
            + result[mention.start : mention.end]
            + span_close
            + result[mention.end :]
        )

    return result


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe attribute values."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
