"""Entity and edge schemas for journaling-focused Graphiti tests.

This ontology is intentionally small and feelings-aware without being heavy:

- Core entity types: Person, Place, Organization, Activity.
- Person entities and person→person edges carry light emotional context
  as attributes (valence, closeness, intensity).
- A small set of edges for relationships, time spent together, and activities.
- A generic RELATES_TO fallback for all other relationships.

Exposed structures:
    * entity_types  – map of entity label → Pydantic model.
    * edge_meta     – map of edge type name → metadata (description + signatures).
    * edge_types    – map of edge type name → Pydantic model for per-edge attributes.
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
    closeness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall sense of closeness with this person (0–1), if known.",
    )
    overall_valence: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Overall emotional tone associated with this person (-1..1).",
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

    activity_type: Optional[str] = Field(
        default=None,
        description="Short label for the activity (meeting, walk, yoga, therapy session, etc.).",
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
# Edge attribute schemas (Pydantic models per *custom* edge type)
# ---------------------------------------------------------------------------


class Knows(BaseModel):
    """General relationship or familiarity between two people."""

    closeness_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How close these people are overall (0–1), if known.",
    )
    primary_context: Optional[str] = Field(
        default=None,
        description="Context for how they know each other (work, school, family, online, etc.).",
    )


class SpendsTimeWith(BaseModel):
    """Two people regularly spend meaningful time together."""

    typical_valence: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Typical emotional tone when they spend time together (-1..1).",
    )
    primary_activity: Optional[str] = Field(
        default=None,
        description="Main thing they tend to do together (e.g., climbing, talking, gaming).",
    )
    frequency_label: Optional[str] = Field(
        default=None,
        description="Rough frequency label like 'daily', 'weekly', 'monthly', etc.",
    )


class Supports(BaseModel):
    """One person offers support to another."""

    support_type: Optional[str] = Field(
        default=None,
        description="Kind of support (emotional, logistical, financial, mentoring, etc.).",
    )
    primary_emotion: Optional[str] = Field(
        default=None,
        description="Dominant emotion you feel about this support (grateful, reassured, etc.).",
    )
    valence: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Emotional tone of feeling supported by this person (-1..1).",
    )


class ConflictsWith(BaseModel):
    """Notable conflict or tension between two people."""

    intensity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How intense the conflict feels overall (0–1).",
    )
    primary_emotion: Optional[str] = Field(
        default=None,
        description="Emotion most associated with this conflict (anger, anxiety, resentment, etc.).",
    )


class ParticipatesIn(BaseModel):
    """A person participates in or does an activity."""

    role: Optional[str] = Field(
        default=None,
        description="Role in the activity, if any (host, guest, student, teammate, etc.).",
    )
    valence: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Emotional tone of doing this activity (-1..1).",
    )


class OccursAt(BaseModel):
    """An activity occurs at or is associated with a place."""

    note: Optional[str] = Field(
        default=None,
        description="Optional note about the location (e.g., usual venue, special place).",
    )


class Visits(BaseModel):
    """A person visits or frequently spends time at a place."""

    frequency_label: Optional[str] = Field(
        default=None,
        description="Rough frequency label like 'daily', 'weekly', 'occasionally', etc.",
    )
    typical_valence: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Typical emotional tone of being at this place (-1..1).",
    )


# Only *custom* edge types go here. RELATES_TO is provided by Graphiti itself.
edge_types: Dict[str, type[BaseModel]] = {
    "Knows": Knows,
    "SpendsTimeWith": SpendsTimeWith,
    "Supports": Supports,
    "ConflictsWith": ConflictsWith,
    "ParticipatesIn": ParticipatesIn,
    "OccursAt": OccursAt,
    "Visits": Visits,
}


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
    "Knows",
    "SpendsTimeWith",
    "Supports",
    "ConflictsWith",
    "ParticipatesIn",
    "OccursAt",
    "Visits",
    "edge_types",
    "edge_type_map",
]
