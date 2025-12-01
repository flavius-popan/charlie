"""Post-extraction entity name cleanup.

Light-touch fixes for common extraction issues. Complex transformations
are left to DSPy optimization.
"""

import re
import logging

logger = logging.getLogger(__name__)


# Strip lowercase "the " - preserves proper names like "The Beatles"
LOWERCASE_THE_PATTERN = r"^the\s+"

# Discard overly generic places
DISCARD_PATTERNS = [
    r"^(living|bed|bath|dining|laundry)\s*room$",
    r"^room$",
    r"^(balcony|patio|porch|deck|basement|garage|attic)$",
]


def _should_discard(name: str) -> bool:
    """Check if entity should be discarded entirely."""
    for pattern in DISCARD_PATTERNS:
        if re.match(pattern, name, flags=re.IGNORECASE):
            return True
    return False


def cleanup_entity_name(name: str) -> str | None:
    """Clean up a single entity name.

    Returns cleaned name, or None if entity should be discarded.
    """
    original = name

    # Check discard patterns
    if _should_discard(name):
        logger.info("Discarding entity: %r", original)
        return None

    # Strip lowercase "the " (case-sensitive)
    name = re.sub(LOWERCASE_THE_PATTERN, "", name).strip()

    # Check discard again after cleanup
    if _should_discard(name):
        logger.info("Discarding entity after cleanup: %r -> %r", original, name)
        return None

    # Discard if empty or too short
    if len(name) < 2:
        logger.info("Discarding short entity: %r", original)
        return None

    if name != original:
        logger.info("Cleaned entity: %r -> %r", original, name)

    return name


def cleanup_extracted_entities(entities: list) -> list:
    """Clean up a list of extracted entities.

    Args:
        entities: List of ExtractedEntity objects (with .name and .entity_type_id)

    Returns:
        Cleaned list with modified names, discarded invalid entries,
        and duplicates merged (first occurrence wins)
    """
    from backend.graph.extract_nodes import ExtractedEntity

    seen: dict[str, int] = {}  # lowercase name -> index in cleaned list
    cleaned = []

    for entity in entities:
        cleaned_name = cleanup_entity_name(entity.name)
        if cleaned_name is None:
            continue

        # Dedupe within batch (case-insensitive, first wins)
        key = cleaned_name.lower()
        if key in seen:
            logger.info("Skipping duplicate: %r (have %r)",
                        cleaned_name, cleaned[seen[key]].name)
            continue

        seen[key] = len(cleaned)
        cleaned.append(ExtractedEntity(
            name=cleaned_name,
            entity_type_id=entity.entity_type_id,
        ))

    return cleaned
