"""Entity and edge type definitions for journaling knowledge graph.

Defines five entity types optimized for personal journaling:
- Person: Individuals with relationship tracking
- Place: Geographic locations and venues
- Organization: Companies, institutions, groups
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
        description="How this person relates to the journal author",
    )


class Place(BaseModel):
    """A geographic location or venue mentioned in journal entries."""

    pass


class Organization(BaseModel):
    """A company, institution, or group mentioned in journal entries."""

    pass


class Activity(BaseModel):
    """An event, action, or experience described in journal entries."""

    pass


# Convenience exports for pipeline usage
entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Activity": Activity,
}

edge_types = {}
edge_type_map = {}


__all__ = [
    "Person",
    "Place",
    "Organization",
    "Activity",
    "entity_types",
    "edge_types",
    "edge_type_map",
]
