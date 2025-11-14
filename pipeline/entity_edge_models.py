"""Entity and edge type definitions for journaling knowledge graph.

Defines five entity types optimized for personal journaling:
- Person: Individuals with relationship tracking
- Place: Geographic locations and venues
- Organization: Companies, institutions, groups
- Concept: Abstract topics and life themes
- Activity: Events, actions, and experiences

Edge types are defined as empty placeholders for future relationship extraction.
"""

from typing import Optional

from pydantic import BaseModel, Field


# Entity type Pydantic models
class Person(BaseModel):
    """A person mentioned in journal entries."""

    relationship_type: Optional[str] = Field(
        None,
        description="Relationship to journal author: friend, family, colleague, acquaintance, romantic partner, professional (therapist/doctor), or other",
    )


class Place(BaseModel):
    """A geographic location or venue mentioned in journal entries."""

    pass


class Organization(BaseModel):
    """A company, institution, or group mentioned in journal entries."""

    pass


class Concept(BaseModel):
    """An abstract topic, theme, or idea discussed in journal entries."""

    pass


class Activity(BaseModel):
    """An event, action, or experience described in journal entries."""

    pass


# Convenience exports for pipeline usage
entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Concept": Concept,
    "Activity": Activity,
}

edge_types = {}
edge_type_map = {}


__all__ = [
    "Person",
    "Place",
    "Organization",
    "Concept",
    "Activity",
    "entity_types",
    "edge_types",
    "edge_type_map",
]
