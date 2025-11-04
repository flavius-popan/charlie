# Journal Entity and Edge Types Design

**Date:** 2025-11-04
**Status:** Approved
**Scope:** v1 - Person + Emotion entities with minimal edge types

## Overview

Design for custom entity and edge types for a personal journaling knowledge graph focused on memory recall and emotional pattern analysis. This follows graphiti-core conventions and integrates with the existing two-layer DSPy pipeline architecture.

## Goals

- Enable memory recall through narrative entities (people mentioned in entries)
- Track emotional patterns across journal entries
- Build foundation for future expansion (places, activities, objects)
- Maintain graphiti-core compliance for downstream pipeline stages
- Keep v1 scope minimal to validate extraction quality

## Entity Types (v1)

### Person

**Purpose:** Track individuals mentioned in journal entries and their relationship to the author.

**Pydantic Model:**
```python
class Person(BaseModel):
    """A person mentioned in journal entries."""
    relationship_type: Optional[str] = Field(
        None,
        description="Relationship to journal author: friend, family, colleague, acquaintance, romantic partner, professional (therapist/doctor), or other"
    )
```

**Graph labels:** `["Entity", "Person"]`

**Attributes stored in EntityNode.attributes:**
- `relationship_type`: Optional string for relationship classification

### Emotion

**Purpose:** Track emotional states experienced by the journal author, using a circumplex model of affect (high/low energy × pleasant/unpleasant).

**Pydantic Model:**
```python
class Emotion(BaseModel):
    """An emotional state experienced by the journal author."""
    specific_emotion: Optional[str] = Field(
        None,
        description="The specific emotion name (e.g., joyful, anxious, content, frustrated)"
    )
    category: Optional[str] = Field(
        None,
        description="Emotion category: high_energy_pleasant, high_energy_unpleasant, low_energy_pleasant, low_energy_unpleasant"
    )
```

**Graph labels:** `["Entity", "Emotion"]`

**Attributes stored in EntityNode.attributes:**
- `specific_emotion`: The specific emotion word (e.g., "anxious", "joyful")
- `category`: One of four quadrants in circumplex model

**Emotion Reference Data:**
- Module will include sample emotions for each quadrant (subset of eventual 144-emotion taxonomy)
- Not enforced as enums - LLM can extract any emotion words
- Available for future prompt engineering to guide vocabulary

## Edge Types (v1 - Defined but Not Extracted Yet)

These edge types are defined in v1 for Stage 2 planning but won't be extracted until Stage 2 is implemented. v1 will use default `RELATES_TO` edges.

### CoOccurrence

**Purpose:** Generic temporal co-occurrence - entities mentioned together in same journal entry.

```python
class CoOccurrence(BaseModel):
    """Temporal co-occurrence - entities mentioned together in a journal entry."""
    context: Optional[str] = Field(
        None,
        description="Brief context of how entities appeared together"
    )
```

### EmotionalAssociation

**Purpose:** Capture associations between people and emotions (directional: Person → Emotion).

```python
class EmotionalAssociation(BaseModel):
    """Association between a person and an emotion in the journal author's experience."""
    context: Optional[str] = Field(
        None,
        description="Context of the emotional association"
    )
```

### SocialConnection

**Purpose:** Map relationships between people in the author's social network (bidirectional).

```python
class SocialConnection(BaseModel):
    """Relationship between two people in the journal author's life."""
    connection_type: Optional[str] = Field(
        None,
        description="Type of connection: friends, family, colleagues, acquaintances, etc."
    )
```

## Edge Type Mapping (for Stage 2)

```python
edge_type_map = {
    # Person-Emotion associations (v1 primary use case)
    ("Person", "Emotion"): ["EmotionalAssociation", "CoOccurrence"],

    # Person-Person connections (v1 for social mapping)
    ("Person", "Person"): ["SocialConnection", "CoOccurrence"],

    # Emotion-Emotion (for future emotional transitions)
    ("Emotion", "Emotion"): ["CoOccurrence"],

    # Fallback for any entity pairs
    ("Entity", "Entity"): ["CoOccurrence"],
}
```

## File Structure

**New file:** `pipeline/entity_edge_models.py`

**Contents:**
1. Emotion reference constants (4 lists, one per quadrant, subset of 144 emotions)
2. Entity type Pydantic models (Person, Emotion)
3. Edge type Pydantic models (CoOccurrence, EmotionalAssociation, SocialConnection)
4. Convenience exports: `entity_types`, `edge_types`, `edge_type_map`

**Exports:**
```python
__all__ = [
    "Person",
    "Emotion",
    "CoOccurrence",
    "EmotionalAssociation",
    "SocialConnection",
    "entity_types",
    "edge_types",
    "edge_type_map",
]
```

## Integration with Existing Pipeline

**No modifications needed to:**
- `pipeline/extract_nodes.py`
- DSPy signatures
- EntityExtractor module
- Resolution logic
- Database queries

**Usage pattern:**
```python
from pipeline.entity_edge_models import entity_types
from pipeline.extract_nodes import ExtractNodes

extractor = ExtractNodes(group_id="user_123")
result = await extractor(
    content="Today I had coffee with Sarah. She always makes me feel anxious...",
    entity_types=entity_types
)
```

**What happens:**
1. `ExtractNodes._format_entity_types()` converts Pydantic models to JSON schema
2. EntityExtractor receives entity type descriptions from docstrings
3. LLM extracts entities and assigns entity_type_id (0=Entity, 1=Person, 2=Emotion)
4. Nodes get proper labels: Person → `["Entity", "Person"]`, Emotion → `["Entity", "Emotion"]`
5. Attributes stored in EntityNode.attributes dict

## Future Expansion Plan

**v2 entities (future):**
- Place (physical locations and abstract spaces)
- Activity (what the author does)
- Object (meaningful possessions)

**v2 edges (future):**
- Directed emotional causation (beyond co-occurrence)
- Temporal sequences (activity chains, emotional transitions)
- Spatial associations (emotions tied to places)

**Expansion approach:**
1. Add new Pydantic models to entity_edge_models.py
2. Update entity_types and edge_type_map dictionaries
3. Test extraction quality with new types
4. Iterate on prompts if needed

No pipeline code changes required - just extend the type definitions.

## Implementation Notes

- All attributes are Optional following graphiti-core convention
- LLM populates what it can extract from text
- No strict vocabulary enforcement (enables flexible extraction)
- Follows graphiti-core's entity extraction and resolution patterns
- Compatible with existing MinHash LSH deduplication
- Works with all downstream stages (edges, attributes, summaries, persistence)

## Success Criteria

v1 is successful when:
1. System extracts Person entities with reasonable accuracy from journal text
2. System extracts Emotion entities and categorizes them into quadrants
3. Extracted entities resolve correctly via existing deduplication logic
4. Person.relationship_type and Emotion attributes populate when inferable
5. Graph queries can surface patterns like "which emotions appear with which people"
