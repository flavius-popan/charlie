"""Post-extraction entity name cleanup.

Fixes common LLM extraction issues that prompt engineering can't solve.
Rules are organized by category for easy maintenance.
"""

import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RULE CATEGORIES
# =============================================================================

# Structural pattern: verb-form + optional article + preposition
# Catches: "walking in X", "going to the X", "eating at X", "learning about X"
VERB_PREP_PATTERN = r"^\w+(ing|ed)\s+(a\s+|the\s+)?(in|at|to|with|around|about|for|on|from)\s+"

# Common verbs that take direct objects (no preposition)
# Keep this list small - only verbs that frequently appear in journal extractions
DIRECT_OBJECT_VERBS = [
    r"^watching\s+",
    r"^watched\s+",
    r"^reading\s+",
    r"^playing\s+",
    r"^preparing\s+",
    r"^bought\s+",
    r"^updated\s+",
    r"^planned\s+",
]

# Base form verbs + preposition (structural pattern only catches -ing/-ed)
BASE_VERB_PREP_PATTERNS = [
    r"^walk\s+(in|to|around|through)\s+(the\s+)?",
    r"^run\s+(in|to|around|through)\s+(the\s+)?",
    r"^go\s+(to|into)\s+(the\s+)?",
]

# Time-of-day prefixes to strip
TIME_PREFIXES = [
    r"^morning\s+",
    r"^evening\s+",
    r"^afternoon\s+",
    r"^late\s+",
    r"^early\s+",
]

# Descriptive prefixes to strip
DESCRIPTIVE_PREFIXES = [
    r"^quick\s+",
    r"^long\s+",
    r"^short\s+",
    r"^chill\s+",
    r"^lazy\s+",
    r"^sunny\s+",
    r"^family\s+",
]

# Phrases to strip (case-insensitive, applied with re.IGNORECASE)
PHRASE_PATTERNS = [
    (r"^a\s+", ""),  # "a podcast" -> "podcast"
    (r"^\d+\s+(more\s+)?episodes?\s+of\s+", ""),  # "2 more episodes of X" -> "X"
]

# Case-sensitive patterns - preserve proper names like "The Beatles", "New York"
LOWERCASE_THE_PATTERN = r"^the\s+"
# Only strip "the new " when followed by a capital (title like "Superman", not "york times")
LOWERCASE_THE_NEW_PATTERN = r"^the\s+new\s+(?=[A-Z])"

# Patterns that should DISCARD the entire entity (return None)
DISCARD_PATTERNS = [
    r"^\d+(st|nd|rd|th)\s+floor$",  # "4th floor", "1st floor"
    r"^(living|bed|bath|dining|laundry)\s*room$",  # generic rooms (not "Green Phone Room")
    r"^room$",  # standalone "room"
    r"^(the|his|her|my)\s+apt$",  # informal apartment refs
    r"^(balcony|patio|porch|deck|basement|garage|attic)$",  # single-word rooms/areas
    r"^(bench|couch|sofa|table|desk|chair)$",  # furniture
    r"^(morning|evening|afternoon|night)$",  # standalone time-of-day words
]


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def _strip_patterns(name: str, patterns: list[str]) -> str:
    """Apply regex patterns to strip prefixes from name."""
    for pattern in patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name.strip()


def _apply_phrase_patterns(name: str, patterns: list[tuple[str, str]]) -> str:
    """Apply find/replace patterns."""
    for pattern, replacement in patterns:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    return name.strip()


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

    # Check discard patterns first (on original)
    if _should_discard(name):
        logger.debug("Discarding entity (pattern match): %r", original)
        return None

    # Strip verb phrases: structural pattern (verb+prep), base verbs, direct object verbs
    name = re.sub(VERB_PREP_PATTERN, "", name, flags=re.IGNORECASE).strip()
    name = _strip_patterns(name, BASE_VERB_PREP_PATTERNS)
    name = _strip_patterns(name, DIRECT_OBJECT_VERBS)

    # Strip time prefixes
    name = _strip_patterns(name, TIME_PREFIXES)

    # Strip descriptive prefixes
    name = _strip_patterns(name, DESCRIPTIVE_PREFIXES)

    # Apply phrase patterns
    name = _apply_phrase_patterns(name, PHRASE_PATTERNS)

    # Strip lowercase "the new " then "the " (case-sensitive to preserve proper names)
    name = re.sub(LOWERCASE_THE_NEW_PATTERN, "", name).strip()
    name = re.sub(LOWERCASE_THE_PATTERN, "", name).strip()

    # Final cleanup
    name = name.strip()

    # Check discard patterns again (after cleanup)
    if _should_discard(name):
        logger.debug("Discarding entity (pattern match after cleanup): %r -> %r", original, name)
        return None

    # Discard if empty or too short after cleanup
    if len(name) < 2:
        logger.debug("Discarding empty/short entity: %r -> %r", original, name)
        return None

    if name != original:
        logger.debug("Cleaned entity: %r -> %r", original, name)

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
            logger.debug("Skipping duplicate entity: %r (already have %r)",
                        cleaned_name, cleaned[seen[key]].name)
            continue

        seen[key] = len(cleaned)
        cleaned.append(ExtractedEntity(
            name=cleaned_name,
            entity_type_id=entity.entity_type_id,
        ))

    return cleaned
