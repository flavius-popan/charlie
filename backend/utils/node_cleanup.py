"""Post-extraction entity name cleanup.

Light-touch fixes for common extraction issues.
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
