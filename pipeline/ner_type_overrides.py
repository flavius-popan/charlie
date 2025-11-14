"""Keyword-based heuristics for reclassifying DistilBERT NER outputs.

The goal is to keep this module extremely easy to edit manually. Update the
`MISC_KEYWORD_OVERRIDES` mapping to quickly adjust how MISC entities get mapped
to the journaling-friendly entity types defined in `entity_edge_models.py`.
"""

from __future__ import annotations

from typing import Iterable


# Base label mapping from DistilBERT BIO tags → pipeline entity types.
# Only the coarse label (PER/LOC/ORG/MISC) matters; the B-/I- prefix is removed.
BASE_LABEL_TO_ENTITY_TYPE: dict[str, str] = {
    "PER": "Person",
    "LOC": "Place",
    "ORG": "Organization",
    # MISC defaults to Activity unless a keyword override matches.
    "MISC": "Activity",
}


# Simple keyword overrides for MISC entities.
# The keys are journaling entity type names; the values are lowercase keywords.
MISC_KEYWORD_OVERRIDES: dict[str, list[str]] = {
    "Activity": [
        "trip",
        "vacation",
        "run",
        "workout",
        "jog",
        "hike",
        "meeting",
        "call",
        "appointment",
        "party",
        "concert",
        "dinner",
        "lunch",
        "coffee",
        "hangout",
        "practice",
        "game",
        "plan",
        "goal",
        "project",
        "habit",
        "idea",
        "resolution",
        "topic",
        "theme",
        "mindset",
        "routine",
    ],
}


def map_ner_label_to_entity_type(
    label: str,
    entity_name: str,
    available_types: Iterable[str] | None = None,
) -> str:
    """Map a DistilBERT BIO label + entity text to a pipeline entity type name."""
    short_label = (label or "").split("-")[-1].upper()
    base_type = BASE_LABEL_TO_ENTITY_TYPE.get(short_label, "Activity")

    if short_label == "MISC":
        override = _match_misc_keyword(entity_name)
        if override:
            base_type = override

    return _ensure_supported_type(base_type, available_types)


def _match_misc_keyword(entity_name: str) -> str | None:
    """Return the override entity type if any keyword matches the MISC entity."""
    candidate = entity_name.strip().lower()
    if not candidate:
        return None

    for entity_type, keywords in MISC_KEYWORD_OVERRIDES.items():
        for keyword in keywords:
            if keyword in candidate:
                return entity_type
    return None


def _ensure_supported_type(
    entity_type: str,
    available_types: Iterable[str] | None,
) -> str:
    """Only return an entity type if the pipeline knows about it."""
    if not available_types:
        return entity_type

    available_lower = {name.lower() for name in available_types}
    if entity_type.lower() in available_lower:
        return entity_type
    return "Entity"


__all__ = [
    "map_ner_label_to_entity_type",
    "MISC_KEYWORD_OVERRIDES",
    "BASE_LABEL_TO_ENTITY_TYPE",
]
