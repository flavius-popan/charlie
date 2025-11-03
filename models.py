"""Pydantic models for DSPy signature outputs."""
from typing import Any

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A factual statement about an entity."""
    entity: str = Field(description="Entity name this fact is about")
    text: str = Field(description="The factual statement")


class Facts(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Fact]


class Relationship(BaseModel):
    """A relationship between two entities."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship type (e.g., works_at, knows)")
    context: str = Field(description="Supporting fact/context for this relationship")
    valid_at: str | None = Field(
        None,
        description="ISO 8601 datetime when the relationship became true (e.g., 2025-04-30T00:00:00Z)"
    )
    invalid_at: str | None = Field(
        None,
        description="ISO 8601 datetime when the relationship stopped being true (e.g., 2025-04-30T00:00:00Z)"
    )


class Relationships(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Relationship]


class EntitySummary(BaseModel):
    """Summary of a specific entity."""
    entity: str = Field(description="Entity name")
    summary: str = Field(description="Concise entity summary")


class EntitySummaries(BaseModel):
    """Collection wrapper for entity summaries."""
    items: list[EntitySummary] = Field(default_factory=list)


class EntityAttribute(BaseModel):
    """Attributes inferred for an entity."""
    entity: str = Field(description="Entity name")
    labels: list[str] = Field(
        default_factory=list,
        description="Graphiti-compatible labels for the entity",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Attribute dictionary to merge onto the entity",
    )


class EntityAttributes(BaseModel):
    """Collection wrapper for entity attributes."""
    items: list[EntityAttribute] = Field(default_factory=list)
