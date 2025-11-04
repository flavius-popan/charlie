"""Entity and edge type definitions for journaling knowledge graph.

This module defines custom entity and edge types for a personal journaling
knowledge graph focused on memory recall and emotional pattern analysis.

v1 Scope:
- Person entities: Track individuals mentioned in journal entries
- Emotion entities: Track emotional states using circumplex model of affect
- Edge types defined but not extracted (will use default RELATES_TO until Stage 2)

Future expansion can add Place, Activity, Object entities without pipeline changes.
"""

from typing import Optional

from pydantic import BaseModel, Field


# Emotion reference data - circumplex model of affect (2D: energy × valence)
# These provide vocabulary guidance for LLM extraction but are not enforced

HIGH_ENERGY_PLEASANT = [
    "excited",
    "joyful",
    "enthusiastic",
    "elated",
    "energetic",
    "thrilled",
    "amazed",
    "inspired",
    "delighted",
    "ecstatic",
]

HIGH_ENERGY_UNPLEASANT = [
    "angry",
    "anxious",
    "stressed",
    "frustrated",
    "nervous",
    "irritated",
    "panicked",
    "tense",
    "overwhelmed",
    "agitated",
]

LOW_ENERGY_PLEASANT = [
    "calm",
    "content",
    "peaceful",
    "relaxed",
    "serene",
    "satisfied",
    "grateful",
    "comfortable",
    "secure",
    "tranquil",
]

LOW_ENERGY_UNPLEASANT = [
    "sad",
    "tired",
    "bored",
    "lonely",
    "melancholy",
    "exhausted",
    "disappointed",
    "dejected",
    "apathetic",
    "withdrawn",
]


# Entity type Pydantic models
class Person(BaseModel):
    """A person mentioned in journal entries."""

    relationship_type: Optional[str] = Field(
        None,
        description="Relationship to journal author: friend, family, colleague, acquaintance, romantic partner, professional (therapist/doctor), or other",
    )


class Emotion(BaseModel):
    """An emotional state experienced by the journal author."""

    specific_emotion: Optional[str] = Field(
        None,
        description="The specific emotion name (e.g., joyful, anxious, content, frustrated)",
    )
    category: Optional[str] = Field(
        None,
        description="Emotion category: high_energy_pleasant, high_energy_unpleasant, low_energy_pleasant, low_energy_unpleasant",
    )


# Edge type Pydantic models (v1 - defined but not extracted yet)
class CoOccurrence(BaseModel):
    """Temporal co-occurrence - entities mentioned together in a journal entry."""

    context: Optional[str] = Field(
        None,
        description="Brief context of how entities appeared together",
    )


class EmotionalAssociation(BaseModel):
    """Association between a person and an emotion in the journal author's experience."""

    context: Optional[str] = Field(
        None,
        description="Context of the emotional association",
    )


class SocialConnection(BaseModel):
    """Relationship between two people in the journal author's life."""

    connection_type: Optional[str] = Field(
        None,
        description="Type of connection: friends, family, colleagues, acquaintances, etc.",
    )


# Convenience exports for pipeline usage
entity_types = {
    "Person": Person,
    "Emotion": Emotion,
}

edge_types = {
    "CoOccurrence": CoOccurrence,
    "EmotionalAssociation": EmotionalAssociation,
    "SocialConnection": SocialConnection,
}

edge_type_map = {
    ("Person", "Emotion"): ["EmotionalAssociation", "CoOccurrence"],
    ("Person", "Person"): ["SocialConnection", "CoOccurrence"],
    ("Emotion", "Emotion"): ["CoOccurrence"],
    ("Entity", "Entity"): ["CoOccurrence"],
}


__all__ = [
    "Person",
    "Emotion",
    "CoOccurrence",
    "EmotionalAssociation",
    "SocialConnection",
    "entity_types",
    "edge_types",
    "edge_type_map",
    "HIGH_ENERGY_PLEASANT",
    "HIGH_ENERGY_UNPLEASANT",
    "LOW_ENERGY_PLEASANT",
    "LOW_ENERGY_UNPLEASANT",
]
