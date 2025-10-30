# Fact Extraction from Journal Entries: Complete Guide

## Overview

This document provides a comprehensive guide to how "facts" (relationships/edges) are extracted from journal entries and episode text in graphiti-core, with concrete examples and the complete extraction logic.

## What is a "Fact"?

In graphiti-core, a **fact** is a relationship between two entities that can be represented as a graph edge. Each fact has:

1. **Two distinct entities** (source and target)
2. **A relationship type** (SCREAMING_SNAKE_CASE predicate)
3. **Natural language description** (the "fact text")
4. **Temporal bounds** (when the relationship became true/ended)

### Fact vs Entity Mention

| Type | Definition | Example |
|------|------------|---------|
| **Entity** | A person, place, concept, or thing | "Alice", "Stanford University", "Machine Learning" |
| **Fact** | A relationship between two entities | "Alice WORKS_AT Stanford University since 2020" |

**Key Distinction**: Entities are nodes; facts are edges connecting nodes.

## Fact Extraction Process

### Step-by-Step Flow

```
Journal Entry (raw text)
    ↓
1. Entity Extraction → [Alice, Bob, Stanford, Microsoft]
    ↓
2. Fact Extraction → [
      (Alice, WORKS_AT, Stanford),
      (Bob, FOUNDED, Microsoft),
      (Alice, KNOWS, Bob)
   ]
    ↓
3. Temporal Resolution → Add valid_at/invalid_at timestamps
    ↓
4. Fact Deduplication → Merge with existing similar facts
    ↓
5. Contradiction Detection → Expire outdated facts
```

## The Fact Extraction Prompt

### System Prompt
```
You are an expert fact extractor that extracts fact triples from text.
1. Extracted fact triples should also be extracted with relevant date information.
2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent.
   All temporal information should be extracted relative to this time.
```

### Extraction Rules (from `prompts/extract_edges.py:115-125`)

```
# EXTRACTION RULES

1. Entity ID Validation: source_entity_id and target_entity_id must use
   only the id values from the ENTITIES list provided above.
   CRITICAL: Using IDs not in the list will cause the edge to be rejected

2. Each fact must involve two DISTINCT entities.

3. Use a SCREAMING_SNAKE_CASE string as the relation_type
   (e.g., FOUNDED, WORKS_AT).

4. Do not emit duplicate or semantically redundant facts.

5. The fact should closely paraphrase the original source sentence(s).
   Do not verbatim quote the original text.

6. Use REFERENCE_TIME to resolve vague or relative temporal expressions
   (e.g., "last week").

7. Do NOT hallucinate or infer temporal bounds from unrelated events.
```

### DateTime Rules (lines 127-133)

```
# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set valid_at to REFERENCE_TIME.
- If a change/termination is expressed, set invalid_at to the relevant timestamp.
- Leave both fields null if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
```

## Concrete Examples

### Example 1: Simple Present-Tense Fact

**Journal Entry:**
```
January 15, 2024, 10:30 AM

Alice works at Stanford University as a research scientist.
She's been there for about 3 years now.
```

**Reference Time:** `2024-01-15T10:30:00Z`

**Extracted Entities:**
```json
[
  {"id": 0, "name": "Alice", "entity_type": "Person"},
  {"id": 1, "name": "Stanford University", "entity_type": "Organization"}
]
```

**Extracted Facts:**
```json
{
  "edges": [
    {
      "source_entity_id": 0,
      "target_entity_id": 1,
      "relation_type": "WORKS_AT",
      "fact": "Alice works at Stanford University as a research scientist",
      "valid_at": "2021-01-15T00:00:00Z",  // ~3 years before reference time
      "invalid_at": null
    }
  ]
}
```

**Key Points:**
- Fact text paraphrases the source (not verbatim)
- `valid_at` resolved from "3 years ago" relative to reference time
- `invalid_at` is null because relationship is ongoing
- Relation type is SCREAMING_SNAKE_CASE

### Example 2: Past Event with Definite End

**Journal Entry:**
```
March 10, 2024, 2:00 PM

Bob told me he founded Microsoft back in 1975 with Paul Allen.
They ran it together until Paul left in 1983.
```

**Reference Time:** `2024-03-10T14:00:00Z`

**Extracted Entities:**
```json
[
  {"id": 0, "name": "Bob", "entity_type": "Person"},
  {"id": 1, "name": "Microsoft", "entity_type": "Organization"},
  {"id": 2, "name": "Paul Allen", "entity_type": "Person"}
]
```

**Extracted Facts:**
```json
{
  "edges": [
    {
      "source_entity_id": 0,
      "target_entity_id": 1,
      "relation_type": "FOUNDED",
      "fact": "Bob founded Microsoft in 1975",
      "valid_at": "1975-01-01T00:00:00Z",
      "invalid_at": null
    },
    {
      "source_entity_id": 2,
      "target_entity_id": 1,
      "relation_type": "CO_FOUNDED",
      "fact": "Paul Allen co-founded Microsoft with Bob in 1975",
      "valid_at": "1975-01-01T00:00:00Z",
      "invalid_at": null
    },
    {
      "source_entity_id": 2,
      "target_entity_id": 1,
      "relation_type": "WORKED_AT",
      "fact": "Paul Allen worked at Microsoft from 1975 to 1983",
      "valid_at": "1975-01-01T00:00:00Z",
      "invalid_at": "1983-01-01T00:00:00Z"
    }
  ]
}
```

**Key Points:**
- Multiple facts extracted from single sentence
- "in 1975" → January 1st, 1975 (year-only dates)
- "until 1983" becomes `invalid_at`
- Different relation types for different relationships

### Example 3: Relative Time Resolution

**Journal Entry:**
```
Today is December 5, 2024, 9:00 AM

I met Sarah for coffee yesterday. She mentioned she just started
working at Google two weeks ago. She used to work at Apple
until last month.
```

**Reference Time:** `2024-12-05T09:00:00Z`

**Extracted Entities:**
```json
[
  {"id": 0, "name": "I", "entity_type": "Person"},
  {"id": 1, "name": "Sarah", "entity_type": "Person"},
  {"id": 2, "name": "Google", "entity_type": "Organization"},
  {"id": 3, "name": "Apple", "entity_type": "Organization"}
]
```

**Extracted Facts:**
```json
{
  "edges": [
    {
      "source_entity_id": 0,
      "target_entity_id": 1,
      "relation_type": "MET_WITH",
      "fact": "I met Sarah for coffee on December 4, 2024",
      "valid_at": "2024-12-04T00:00:00Z",  // yesterday
      "invalid_at": "2024-12-04T23:59:59Z"
    },
    {
      "source_entity_id": 1,
      "target_entity_id": 2,
      "relation_type": "WORKS_AT",
      "fact": "Sarah works at Google as of late November 2024",
      "valid_at": "2024-11-21T00:00:00Z",  // ~2 weeks ago
      "invalid_at": null
    },
    {
      "source_entity_id": 1,
      "target_entity_id": 3,
      "relation_type": "WORKED_AT",
      "fact": "Sarah worked at Apple until November 2024",
      "valid_at": null,  // start date unknown
      "invalid_at": "2024-11-01T00:00:00Z"  // last month
    }
  ]
}
```

**Key Points:**
- "yesterday" → calculated relative to reference time
- "two weeks ago" → approximated to Nov 21
- "last month" → November 1st
- Past events can have definite end times (`invalid_at`)

### Example 4: Implicit Relationships

**Journal Entry:**
```
May 20, 2024, 6:00 PM

Had a great conversation with Dr. Chen today. She's advising
my PhD research on neural networks. We discussed my paper
that was just accepted to NeurIPS.
```

**Reference Time:** `2024-05-20T18:00:00Z`

**Extracted Entities:**
```json
[
  {"id": 0, "name": "I", "entity_type": "Person"},
  {"id": 1, "name": "Dr. Chen", "entity_type": "Person"},
  {"id": 2, "name": "Neural Networks", "entity_type": "Concept"},
  {"id": 3, "name": "NeurIPS", "entity_type": "Organization"}
]
```

**Extracted Facts:**
```json
{
  "edges": [
    {
      "source_entity_id": 0,
      "target_entity_id": 1,
      "relation_type": "SPOKE_WITH",
      "fact": "I had a conversation with Dr. Chen on May 20, 2024",
      "valid_at": "2024-05-20T00:00:00Z",
      "invalid_at": "2024-05-20T23:59:59Z"
    },
    {
      "source_entity_id": 1,
      "target_entity_id": 0,
      "relation_type": "ADVISES",
      "fact": "Dr. Chen advises my PhD research",
      "valid_at": "2024-05-20T18:00:00Z",  // ongoing as of reference time
      "invalid_at": null
    },
    {
      "source_entity_id": 0,
      "target_entity_id": 2,
      "relation_type": "RESEARCHES",
      "fact": "I am researching neural networks",
      "valid_at": "2024-05-20T18:00:00Z",
      "invalid_at": null
    },
    {
      "source_entity_id": 0,
      "target_entity_id": 3,
      "relation_type": "PUBLISHED_AT",
      "fact": "I have a paper accepted at NeurIPS",
      "valid_at": "2024-05-20T00:00:00Z",  // recently accepted
      "invalid_at": null
    }
  ]
}
```

**Key Points:**
- "advising my PhD" → ongoing relationship (no `invalid_at`)
- Implicit relationships extracted ("researches neural networks")
- Event-based facts have definite times (conversation on specific day)
- State-based facts are ongoing (advises, researches)

### Example 5: Contradicting Facts with Temporal Updates

**Episode 1 (March 1, 2024):**
```
Alice works at Microsoft as a software engineer.
```

**Extracted Fact:**
```json
{
  "source_entity_id": 0,
  "target_entity_id": 1,
  "relation_type": "WORKS_AT",
  "fact": "Alice works at Microsoft as a software engineer",
  "valid_at": "2024-03-01T00:00:00Z",
  "invalid_at": null,
  "uuid": "fact-123"
}
```

**Episode 2 (June 1, 2024):**
```
Alice just started her new job at Google yesterday.
```

**New Fact:**
```json
{
  "source_entity_id": 0,
  "target_entity_id": 2,
  "relation_type": "WORKS_AT",
  "fact": "Alice works at Google",
  "valid_at": "2024-05-31T00:00:00Z",  // yesterday
  "invalid_at": null
}
```

**Contradiction Resolution:**
```json
// Old fact automatically updated:
{
  "uuid": "fact-123",
  "fact": "Alice works at Microsoft as a software engineer",
  "valid_at": "2024-03-01T00:00:00Z",
  "invalid_at": "2024-05-31T00:00:00Z",  // ← UPDATED: ended when new job started
  "expired_at": "2024-06-01T00:00:00Z"
}

// New fact added:
{
  "uuid": "fact-456",
  "fact": "Alice works at Google",
  "valid_at": "2024-05-31T00:00:00Z",
  "invalid_at": null
}
```

**Key Points:**
- System detects contradiction (Alice can't work at two places)
- Old fact's `invalid_at` set to when new fact's `valid_at`
- Old fact's `expired_at` set to current time
- Timeline remains consistent

## Reflexion Loop: Ensuring Completeness

After initial extraction, graphiti uses a **reflexion loop** to catch missed facts:

### Reflexion Prompt (from `prompts/extract_edges.py:139-164`)

```
Given the above MESSAGES, list of EXTRACTED ENTITIES entities,
and list of EXTRACTED FACTS; determine if any facts haven't
been extracted.
```

**Example:**

**Initial Extraction:**
```json
{
  "edges": [
    {"fact": "Alice works at Stanford"}
  ]
}
```

**Reflexion Check:**
```
Missing facts: ["Alice collaborates with Bob on research project"]
```

**Second Pass:**
```json
{
  "edges": [
    {"fact": "Alice works at Stanford"},
    {"fact": "Alice collaborates with Bob on research project"}  // ← Added
  ]
}
```

This continues up to `MAX_REFLEXION_ITERATIONS` (typically 2-3).

## Custom Fact Types

### Defining Custom Fact Types

You can specify domain-specific fact types with structured attributes:

```python
class Employment(BaseModel):
    """Employment relationship between person and organization"""
    position: str
    department: str | None = None
    start_date: str | None = None

class Collaboration(BaseModel):
    """Research collaboration between people"""
    project_name: str
    role: str | None = None

edge_types = {
    "WORKS_AT": Employment,
    "COLLABORATES_WITH": Collaboration,
}

edge_type_map = {
    ("Person", "Organization"): ["WORKS_AT"],
    ("Person", "Person"): ["COLLABORATES_WITH"],
}
```

**Extraction with Custom Types:**

**Journal Entry:**
```
Alice works at Stanford in the CS department as a postdoc researcher.
```

**Extracted Fact with Attributes:**
```json
{
  "relation_type": "WORKS_AT",
  "fact": "Alice works at Stanford as a postdoc researcher in the CS department",
  "attributes": {
    "position": "postdoc researcher",
    "department": "CS",
    "start_date": null
  }
}
```

## Fact Quality Guidelines

### Good Facts

✅ **Specific and Grounded**
```
"Alice published a paper on transformer architectures at NeurIPS 2023"
```

✅ **Properly Paraphrased**
```
Original: "I think Alice is really into deep learning these days"
Fact: "Alice is interested in deep learning"
```

✅ **Temporally Bounded**
```
"Bob founded Microsoft in 1975"
(valid_at: 1975-01-01)
```

✅ **Distinct Entities**
```
Source: Alice (Person)
Target: Stanford (Organization)
Relation: WORKS_AT
```

### Bad Facts

❌ **Single Entity**
```
"Alice is smart"  // No second entity
```

❌ **Verbatim Quotes**
```
"I think Alice is really into deep learning these days"  // Should be paraphrased
```

❌ **Hallucinated Dates**
```
Journal: "Alice works at Stanford"
Fact: valid_at: "2020-01-01"  // ❌ No date mentioned in source
```

❌ **Ambiguous References**
```
"She works there"  // Pronouns should be resolved to entity names
```

## Integration with Episode Processing

### Complete Flow from Journal Entry to Graph

```
1. User Input
   ├─ Journal text: "Alice met Bob at Stanford yesterday..."
   └─ Reference time: 2024-06-01T10:00:00Z

2. Episode Creation
   └─ EpisodicNode(content=text, valid_at=reference_time)

3. Entity Extraction
   ├─ EntityNode(name="Alice", labels=["Person"])
   ├─ EntityNode(name="Bob", labels=["Person"])
   └─ EntityNode(name="Stanford", labels=["Organization"])

4. Fact Extraction  ← THIS STEP
   ├─ Extract edges between entities
   ├─ Resolve temporal bounds
   └─ Reflexion loop for completeness

5. Fact Resolution
   ├─ Deduplicate against existing facts
   ├─ Detect contradictions
   └─ Update temporal bounds

6. Database Save
   ├─ EntityEdge(fact="Alice met Bob at Stanford...", ...)
   ├─ EpisodicEdge(episode → Alice) [MENTIONS]
   ├─ EpisodicEdge(episode → Bob) [MENTIONS]
   └─ EpisodicEdge(episode → Stanford) [MENTIONS]
```

## Fact Storage Model

### EntityEdge Structure (the "fact" in the database)

```python
class EntityEdge(Edge):
    # Identification
    uuid: str
    group_id: str

    # Relationship
    name: str                    # "WORKS_AT", "FOUNDED", etc.
    source_node_uuid: str        # Entity A
    target_node_uuid: str        # Entity B

    # Content
    fact: str                    # Natural language description
    fact_embedding: list[float]  # For semantic search

    # Temporal
    created_at: datetime         # When fact was added to graph
    valid_at: datetime           # When relationship became true
    invalid_at: datetime         # When relationship ended
    expired_at: datetime         # When fact was superseded/invalidated

    # Provenance
    episodes: list[str]          # Which episodes mention this fact

    # Custom attributes
    attributes: dict[str, Any]   # Structured data from custom fact types
```

## Best Practices for Journal Entries

To get high-quality fact extraction:

### ✅ DO

1. **Be explicit about relationships**
   ```
   Good: "Alice collaborates with Bob on the neural networks project"
   Bad:  "Alice and Bob work together"
   ```

2. **Include temporal information**
   ```
   Good: "I met Sarah last Tuesday at the conference"
   Bad:  "I met Sarah"
   ```

3. **Use full names first**
   ```
   Good: "Alice Chen works at Stanford. She is a research scientist."
   Bad:  "She works at Stanford as a research scientist."
   ```

4. **Distinguish between facts and opinions**
   ```
   Fact: "Alice published 3 papers this year"
   Opinion: "Alice is a brilliant researcher"  // Won't create useful edge
   ```

### ❌ AVOID

1. **Ambiguous pronouns without prior reference**
2. **Implied relationships without explicit mentions**
3. **Dates like "soon", "later", "eventually" (non-resolvable)
4. **Complex multi-hop inferences**

## Summary

Facts in graphiti-core are:

1. **Relationships between entities** extracted from journal text
2. **Stored as graph edges** with temporal bounds
3. **Enriched with metadata** (fact text, embeddings, custom attributes)
4. **Automatically deduplicated and reconciled** across episodes
5. **Temporally tracked** with valid_at, invalid_at, expired_at

The extraction process uses:
- **LLM prompts** with structured output (Pydantic models)
- **Reflexion loops** for completeness
- **Temporal resolution** using reference timestamps
- **Contradiction detection** for fact updates
- **Hybrid search** for deduplication

This enables building a consistent, temporally-aware knowledge graph from unstructured journal entries.
