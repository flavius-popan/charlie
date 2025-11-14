"""Entity and edge schemas tuned for the current DSPy optimizers.

The optimizers ship with demonstrations that focus on four concrete entity
types – people, places, organizations, and activities – plus a fixed set of
relationship names such as ``MEETS_AT`` or ``WORKS_AT``. These definitions keep
the runtime pipeline and the optimizers in sync without rerunning expensive
teleprompter jobs.

The module exposes three structures:
    * ``entity_types`` – map of entity label → Pydantic model used by Stage 3.
    * ``edge_meta`` – map of relation name → metadata (description + signatures).
    * ``edge_types`` – map of relation name → Pydantic model describing per-edge attributes.
    * ``edge_type_map`` – convenience lookup of allowed relations for every
      (source_type, target_type) pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity type Pydantic models
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A person mentioned in a journal entry."""

    relationship_type: Optional[str] = Field(
        default=None,
        description="How this person relates to the author (e.g., friend, partner, coach).",
    )


class Place(BaseModel):
    """A location or venue."""

    category: Optional[str] = Field(
        default=None,
        description="Optional descriptor such as park, cafe, clinic, etc.",
    )


class Organization(BaseModel):
    """A company, team, or community group."""

    category: Optional[str] = Field(
        default=None,
        description="Type of organization (company, nonprofit, club, etc.).",
    )


class Activity(BaseModel):
    """An event, outing, or recurring routine."""

    activity_type: Optional[str] = Field(
        default=None,
        description="Short label for the activity (walk, yoga, therapy session, …).",
    )


entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Activity": Activity,
}


@dataclass(frozen=True)
class EdgeMeta:
    description: str
    source_types: Tuple[str, ...]
    target_types: Tuple[str, ...]
    symmetric: bool = False


edge_meta: dict[str, EdgeMeta] = {
    "MEETS_AT": EdgeMeta(
        description="A person meets or spends time at a specific place.",
        source_types=("Person",),
        target_types=("Place",),
    ),
    "VISITS": EdgeMeta(
        description="A person visits a location.",
        source_types=("Person",),
        target_types=("Place",),
    ),
    "PARTICIPATES_IN": EdgeMeta(
        description="A person or pet participates in an activity.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "VOLUNTEERS_AT": EdgeMeta(
        description="A person volunteers with an organization or place.",
        source_types=("Person",),
        target_types=("Organization", "Place"),
    ),
    "WORKS_AT": EdgeMeta(
        description="A person is employed by an organization or works at a place.",
        source_types=("Person",),
        target_types=("Organization", "Place"),
    ),
    "HOSTS": EdgeMeta(
        description="A person or organization hosts an activity or gathering.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "GUIDES": EdgeMeta(
        description="A person guides or mentors another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    "RUNS": EdgeMeta(
        description="A person or organization runs an activity or program.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "LEADS": EdgeMeta(
        description="A person leads or facilitates an activity.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "FOCUSES_ON": EdgeMeta(
        description="An activity centers on another activity or topic.",
        source_types=("Activity",),
        target_types=("Activity",),
    ),
    "FACILITATES": EdgeMeta(
        description="A person facilitates an activity or group.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "SUPPORTED_BY": EdgeMeta(
        description="A person receives support from another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    "COACHES_FOR": EdgeMeta(
        description="A person coaches for an organization or team.",
        source_types=("Person",),
        target_types=("Organization",),
    ),
    "TRAINS_WITH": EdgeMeta(
        description="Two people train or practice together.",
        source_types=("Person",),
        target_types=("Person",),
        symmetric=True,
    ),
    "ATTENDS": EdgeMeta(
        description="A person attends an activity or gathering.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "OWNS": EdgeMeta(
        description="A person owns or operates a place or organization.",
        source_types=("Person",),
        target_types=("Place", "Organization"),
    ),
    "INTRODUCES": EdgeMeta(
        description="A person introduces another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    "RELATES_TO": EdgeMeta(
        description="Generic relationship between entities.",
        source_types=("Person", "Place", "Organization", "Activity"),
        target_types=("Person", "Place", "Organization", "Activity"),
        symmetric=True,
    ),
}


# ---------------------------------------------------------------------------
# Edge attribute schemas (Pydantic models per edge type)
# ---------------------------------------------------------------------------


class MeetsAt(BaseModel):
    """A person meets or spends time at a specific place."""

    context: Optional[str] = Field(
        default=None,
        description="Optional short description of the meeting context.",
    )


class Visits(BaseModel):
    """A person visits a location."""

    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for the visit.",
    )


class ParticipatesIn(BaseModel):
    """A person or organization participates in an activity."""

    role: Optional[str] = Field(
        default=None,
        description="Role or position in the activity, if specified.",
    )


class VolunteersAt(BaseModel):
    """A person volunteers with an organization or place."""


class WorksAt(BaseModel):
    """A person is employed by an organization or works at a place."""

    title: Optional[str] = Field(
        default=None,
        description="Job title or role, if stated.",
    )


class Hosts(BaseModel):
    """A person or organization hosts an activity or gathering."""


class Guides(BaseModel):
    """A person guides or mentors another person."""


class Runs(BaseModel):
    """A person or organization runs an activity or program."""


class Leads(BaseModel):
    """A person leads or facilitates an activity."""


class FocusesOn(BaseModel):
    """An activity centers on another activity or topic."""


class Facilitates(BaseModel):
    """A person facilitates an activity or group."""


class SupportedBy(BaseModel):
    """A person receives support from another person."""


class CoachesFor(BaseModel):
    """A person coaches for an organization or team."""


class TrainsWith(BaseModel):
    """Two people train or practice together."""


class Attends(BaseModel):
    """A person attends an activity or gathering."""


class Owns(BaseModel):
    """A person owns or operates a place or organization."""


class Introduces(BaseModel):
    """A person introduces another person."""


class RelatesTo(BaseModel):
    """Generic fallback relationship between entities."""

    note: Optional[str] = Field(
        default=None,
        description="Optional note describing how the entities relate.",
    )


edge_types: dict[str, type[BaseModel]] = {
    "MEETS_AT": MeetsAt,
    "VISITS": Visits,
    "PARTICIPATES_IN": ParticipatesIn,
    "VOLUNTEERS_AT": VolunteersAt,
    "WORKS_AT": WorksAt,
    "HOSTS": Hosts,
    "GUIDES": Guides,
    "RUNS": Runs,
    "LEADS": Leads,
    "FOCUSES_ON": FocusesOn,
    "FACILITATES": Facilitates,
    "SUPPORTED_BY": SupportedBy,
    "COACHES_FOR": CoachesFor,
    "TRAINS_WITH": TrainsWith,
    "ATTENDS": Attends,
    "OWNS": Owns,
    "INTRODUCES": Introduces,
    "RELATES_TO": RelatesTo,
}


def _build_edge_type_map(
    metadata: dict[str, EdgeMeta],
) -> dict[tuple[str, str], list[str]]:
    mapping: dict[tuple[str, str], list[str]] = {}
    for edge_name, meta in metadata.items():
        for source in meta.source_types:
            for target in meta.target_types:
                key = (source, target)
                mapping.setdefault(key, []).append(edge_name)

                if meta.symmetric and source != target:
                    reverse_key = (target, source)
                    mapping.setdefault(reverse_key, []).append(edge_name)
    return mapping


edge_type_map = _build_edge_type_map(edge_meta)


__all__ = [
    "Activity",
    "Organization",
    "Person",
    "Place",
    "EdgeMeta",
    "entity_types",
    "edge_meta",
    "edge_types",
    "edge_type_map",
    "MeetsAt",
    "Visits",
    "ParticipatesIn",
    "VolunteersAt",
    "WorksAt",
    "Hosts",
    "Guides",
    "Runs",
    "Leads",
    "FocusesOn",
    "Facilitates",
    "SupportedBy",
    "CoachesFor",
    "TrainsWith",
    "Attends",
    "Owns",
    "Introduces",
    "RelatesTo",
]
