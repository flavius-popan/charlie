"""Entity and edge schemas for journaling-focused Graphiti tests.

This ontology is intentionally small so Stage 3 only extracts attributes that
carry real downstream value:

- Core entity types: Person, Place, Organization, Activity.
- Person entities only capture relationship_type strings.
- Activities keep a single `purpose` string for descriptive clustering.
- Edge types use labels only (no attribute schema) to reduce JSON parsing.

Exposed structures:
    * entity_types  – map of entity label → Pydantic model.
    * edge_meta     – map of edge type name → metadata (description + signatures).
    * edge_type_map – allowed edge types for every (source_type, target_type) pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity type Pydantic models
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A person mentioned in a journal entry."""

    relationship_type: Optional[str] = Field(
        default=None,
        description=(
            "How this person relates to the author "
            "(e.g., friend, partner, coworker, family, therapist)."
        ),
    )


class Place(BaseModel):
    """A location or venue."""

    category: Optional[str] = Field(
        default=None,
        description="Descriptor such as home, office, park, cafe, clinic, etc.",
    )


class Organization(BaseModel):
    """A company, team, or community group."""

    category: Optional[str] = Field(
        default=None,
        description="Type of organization (company, nonprofit, club, team, etc.).",
    )


class Activity(BaseModel):
    """An event, outing, or recurring routine."""

    purpose: Optional[str] = Field(
        default=None,
        description="Short description of why/what the activity is (therapy, bike ride, etc.).",
    )


entity_types: Dict[str, type[BaseModel]] = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Activity": Activity,
}


# ---------------------------------------------------------------------------
# Edge metadata (used to build edge_type_map)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeMeta:
    """Metadata describing an edge type and its allowed signatures."""

    description: str
    source_types: Tuple[str, ...]
    target_types: Tuple[str, ...]
    symmetric: bool = False


edge_meta: Dict[str, EdgeMeta] = {
    # Person ↔ Person: general relationship / familiarity
    "Knows": EdgeMeta(
        description="Two people know each other in some ongoing way.",
        source_types=("Person",),
        target_types=("Person",),
        symmetric=True,
    ),
    # Person ↔ Person: spending time together (friendship, dating, etc.)
    "SpendsTimeWith": EdgeMeta(
        description="Two people regularly spend meaningful time together.",
        source_types=("Person",),
        target_types=("Person",),
        symmetric=True,
    ),
    # Person → Person: emotional / practical support
    "Supports": EdgeMeta(
        description="One person offers support (emotional or practical) to another.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    # Person ↔ Person: conflict or tension
    "ConflictsWith": EdgeMeta(
        description="There is notable conflict, tension, or friction between two people.",
        source_types=("Person",),
        target_types=("Person",),
        symmetric=True,
    ),
    # Person → Activity: participation in shared or solo activities
    "ParticipatesIn": EdgeMeta(
        description="A person participates in or does an activity.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    # Activity → Place: where things happen
    "OccursAt": EdgeMeta(
        description="An activity typically occurs at or is associated with a place.",
        source_types=("Activity",),
        target_types=("Place",),
    ),
    # Optional: Person → Place (home base, favorite spots, etc.)
    "Visits": EdgeMeta(
        description="A person visits or frequently spends time at a place.",
        source_types=("Person",),
        target_types=("Place",),
    ),
    # Fallback for any entity pair; uses Graphiti's built-in RELATES_TO edge.
    "RELATES_TO": EdgeMeta(
        description="Generic relationship between any two entities.",
        source_types=("Entity",),
        target_types=("Entity",),
        symmetric=True,
    ),
}


# ---------------------------------------------------------------------------
# No edge attribute schemas. Edges are label-only.
edge_types: Dict[str, type[BaseModel]] = {}


# ---------------------------------------------------------------------------
# Edge type map
# ---------------------------------------------------------------------------


def _build_edge_type_map(
    metadata: Dict[str, EdgeMeta],
) -> Dict[Tuple[str, str], List[str]]:
    """Build (source_label, target_label) → [edge_type_name, ...] mapping."""
    mapping: Dict[Tuple[str, str], List[str]] = {}
    for edge_name, meta in metadata.items():
        for source in meta.source_types:
            for target in meta.target_types:
                key = (source, target)
                mapping.setdefault(key, []).append(edge_name)

                # For symmetric relationships, also allow the reversed (target, source) pair.
                if meta.symmetric and source != target:
                    reverse_key = (target, source)
                    mapping.setdefault(reverse_key, []).append(edge_name)
    return mapping


edge_type_map: Dict[Tuple[str, str], List[str]] = _build_edge_type_map(edge_meta)


__all__ = [
    # Entities
    "Activity",
    "Organization",
    "Person",
    "Place",
    "entity_types",
    # Edge metadata / schemas
    "EdgeMeta",
    "edge_meta",
    "edge_types",
    "edge_type_map",
]
