"""Pydantic models for DSPy signature outputs."""
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


class Relationships(BaseModel):
    """Collection wrapper for DSPy output."""
    items: list[Relationship]
