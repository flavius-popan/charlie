"""Entity and edge schemas tuned for the current DSPy optimizers.

The optimizers ship with demonstrations that focus on four concrete entity
types – people, places, organizations, and activities – plus a fixed set of
relationship names such as ``MEETS_AT`` or ``WORKS_AT``. These definitions keep
the runtime pipeline and the optimizers in sync without rerunning expensive
teleprompter jobs.

The module exposes three structures:
    * ``entity_types`` – map of entity label → Pydantic model used by Stage 3.
    * ``edge_types`` – map of relation name → ``EdgeType`` metadata.
    * ``edge_type_map`` – convenience lookup of allowed relations for every
      (source_type, target_type) pair.
"""

from __future__ import annotations

from typing import Iterable, Optional

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


# ---------------------------------------------------------------------------
# Edge schemas
# ---------------------------------------------------------------------------


class EdgeType(BaseModel):
    """Metadata describing a relationship type."""

    name: str
    description: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    symmetric: bool = False


edge_types: dict[str, EdgeType] = {
    "MEETS_AT": EdgeType(
        name="MEETS_AT",
        description="A person meets or spends time at a specific place.",
        source_types=("Person",),
        target_types=("Place",),
    ),
    "VISITS": EdgeType(
        name="VISITS",
        description="A person visits a location.",
        source_types=("Person",),
        target_types=("Place",),
    ),
    "PARTICIPATES_IN": EdgeType(
        name="PARTICIPATES_IN",
        description="A person or pet participates in an activity.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "VOLUNTEERS_AT": EdgeType(
        name="VOLUNTEERS_AT",
        description="A person volunteers with an organization or place.",
        source_types=("Person",),
        target_types=("Organization", "Place"),
    ),
    "WORKS_AT": EdgeType(
        name="WORKS_AT",
        description="A person is employed by an organization or works at a place.",
        source_types=("Person",),
        target_types=("Organization", "Place"),
    ),
    "HOSTS": EdgeType(
        name="HOSTS",
        description="A person or organization hosts an activity or gathering.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "GUIDES": EdgeType(
        name="GUIDES",
        description="A person guides or mentors another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    "RUNS": EdgeType(
        name="RUNS",
        description="A person or organization runs an activity or program.",
        source_types=("Person", "Organization"),
        target_types=("Activity",),
    ),
    "LEADS": EdgeType(
        name="LEADS",
        description="A person leads or facilitates an activity.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "FOCUSES_ON": EdgeType(
        name="FOCUSES_ON",
        description="An activity centers on another activity or topic.",
        source_types=("Activity",),
        target_types=("Activity",),
    ),
    "FACILITATES": EdgeType(
        name="FACILITATES",
        description="A person facilitates an activity or group.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "SUPPORTED_BY": EdgeType(
        name="SUPPORTED_BY",
        description="A person receives support from another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
    "COACHES_FOR": EdgeType(
        name="COACHES_FOR",
        description="A person coaches for an organization or team.",
        source_types=("Person",),
        target_types=("Organization",),
    ),
    "TRAINS_WITH": EdgeType(
        name="TRAINS_WITH",
        description="Two people train or practice together.",
        source_types=("Person",),
        target_types=("Person",),
        symmetric=True,
    ),
    "ATTENDS": EdgeType(
        name="ATTENDS",
        description="A person attends an activity or gathering.",
        source_types=("Person",),
        target_types=("Activity",),
    ),
    "OWNS": EdgeType(
        name="OWNS",
        description="A person owns or operates a place or organization.",
        source_types=("Person",),
        target_types=("Place", "Organization"),
    ),
    "INTRODUCES": EdgeType(
        name="INTRODUCES",
        description="A person introduces another person.",
        source_types=("Person",),
        target_types=("Person",),
    ),
}


def _build_edge_type_map(
    edge_defs: Iterable[EdgeType],
) -> dict[tuple[str, str], list[str]]:
    mapping: dict[tuple[str, str], list[str]] = {}
    for edge in edge_defs:
        for source in edge.source_types:
            for target in edge.target_types:
                key = (source, target)
                mapping.setdefault(key, []).append(edge.name)

                if edge.symmetric and source != target:
                    reverse_key = (target, source)
                    mapping.setdefault(reverse_key, []).append(edge.name)
    return mapping


edge_type_map = _build_edge_type_map(edge_types.values())


__all__ = [
    "Activity",
    "Organization",
    "Person",
    "Place",
    "EdgeType",
    "entity_types",
    "edge_types",
    "edge_type_map",
]
