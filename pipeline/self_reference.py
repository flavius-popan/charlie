"""Utilities for anchoring journal entries to the author's SELF entity."""

from __future__ import annotations

import re
from uuid import UUID

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now


SELF_ENTITY_UUID = UUID("11111111-1111-1111-1111-111111111111")
SELF_ENTITY_NAME = "Self"
SELF_ENTITY_LABELS = ["Entity", "Person"]
SELF_PROMPT_NOTE = (
    "Represents the journal author. Map any first-person pronouns (I/me/my/mine/myself) "
    "to this entity so the author's relationships stay anchored."
)

_FIRST_PERSON_PATTERN = re.compile(
    r"\b(i|me|my|mine|myself)\b", re.IGNORECASE | re.MULTILINE
)
_SELF_ALIASES = {"i", "me", "my", "mine", "myself"}


def contains_first_person_reference(text: str | None) -> bool:
    """Return True when the journal entry uses first-person pronouns."""
    if not text:
        return False
    return bool(_FIRST_PERSON_PATTERN.search(text))


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def is_self_entity_name(name: str | None) -> bool:
    """Return True when the provided entity name clearly refers to the author."""
    return _normalize_token(name) in _SELF_ALIASES


def build_provisional_self_node(group_id: str) -> EntityNode:
    """Create a placeholder EntityNode for the author."""
    return EntityNode(
        name=SELF_ENTITY_NAME,
        group_id=group_id,
        labels=list(SELF_ENTITY_LABELS),
        summary="",
        created_at=utc_now(),
    )
