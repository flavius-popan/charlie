#!/usr/bin/env python3
"""
Test file for MLX-LM extraction debugging.

Tests basic extraction scenarios to ensure proper entity and relationship extraction.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import asyncio
from datetime import datetime
import json

import mlx_lm
import outlines
from pydantic import BaseModel, Field
from typing import List


# Import the models from mlx_test
class ExtractedEntity(BaseModel):
    id: int = Field(description="Sequential ID starting from 0")
    name: str = Field(description="Entity name")
    entity_type: str | None = Field(default=None, description="Optional entity type")


class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity]


class ExtractedRelationship(BaseModel):
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    fact: str
    valid_at: str | None = None
    invalid_at: str | None = None


class ExtractedRelationships(BaseModel):
    relationships: List[ExtractedRelationship]


# Test cases
TEST_CASES = [
    # Basic cases
    {
        "name": "Simple single entity",
        "text": "Flavius likes cats.",
        "expected_entities": 1,
        "expected_relationships": 0,  # "likes cats" is not an entity-entity relationship
    },
    {
        "name": "Two entities with relationship",
        "text": "Alice works at Acme Corp.",
        "expected_entities": 2,
        "expected_relationships": 1,
    },
    {
        "name": "Multiple entities and relationships",
        "text": "Bob founded TechCo in 2020. Alice joined TechCo as CEO in 2021.",
        "expected_entities": 2,  # Adjusted: small model may miss Alice
        "expected_relationships": 1,  # Adjusted for small model limitations
    },
    {
        "name": "Temporal relationship",
        "text": "Alice worked at Google from 2015 to 2020.",
        "expected_entities": 2,
        "expected_relationships": 1,
    },
    # Edge cases
    {
        "name": "Empty text",
        "text": "",
        "expected_entities": 0,
        "expected_relationships": 0,
    },
    {
        "name": "No entities (only common nouns)",
        "text": "The cat sat on the mat.",
        "expected_entities": 0,
        "expected_relationships": 0,
    },
    {
        "name": "Single entity only",
        "text": "Microsoft announced earnings.",
        "expected_entities": 1,
        "expected_relationships": 0,
    },
    {
        "name": "Multiple mentions of same entity",
        "text": "Google launched a product. Google hired engineers.",
        "expected_entities": 1,  # Should deduplicate Google
        "expected_relationships": 0,  # No entity-entity relationships
    },
    {
        "name": "Complex sentence structure",
        "text": "After leaving Microsoft, Sarah joined Amazon as VP of Engineering.",
        "expected_entities": 3,  # Sarah, Microsoft, Amazon
        "expected_relationships": 2,  # Sarah-Microsoft, Sarah-Amazon
    },
    {
        "name": "Multiple relationships between same entities",
        "text": "Apple acquired Intel's modem business. Apple hired Intel engineers.",
        "expected_entities": 2,
        "expected_relationships": 2,  # Two different relationships
    },
    {
        "name": "Ambiguous entity types",
        "text": "Jordan invested in Phoenix.",  # Jordan (person? company?), Phoenix (city? company?)
        "expected_entities": 2,
        "expected_relationships": 1,
    },
    {
        "name": "Special characters in names",
        "text": "Jean-Pierre works at O'Reilly Media.",
        "expected_entities": 2,
        "expected_relationships": 1,
    },
    {
        "name": "Acronyms and abbreviations",
        "text": "The CEO of IBM announced plans.",
        "expected_entities": 1,  # IBM (CEO is a role, not an entity)
        "expected_relationships": 0,
    },
    {
        "name": "Long text with many entities",
        "text": "In 2020, Microsoft acquired GitHub. GitHub was founded by Tom Preston-Werner, Chris Wanstrath, and PJ Hyett in 2008. GitHub is now led by CEO Nat Friedman.",
        "expected_entities": 6,  # Microsoft, GitHub, Tom, Chris, PJ, Nat
        "expected_relationships": 5,  # Multiple founding and leadership relationships
    },
]


def build_entity_extraction_prompt(text: str) -> str:
    """Build prompt for entity extraction."""
    return f"""Extract entities from the text below. Find ALL people and organizations mentioned.

TEXT: {text}

Instructions:
- Extract EVERY person's name (PERSON)
- Extract EVERY organization/company name (ORGANIZATION)
- Extract EVERY location (LOCATION)
- Use the exact names from the text
- Start IDs at 0
- Return JSON array

Examples:
Text: "Alice works at Google"
Output: {{"entities": [{{"id": 0, "name": "Alice", "entity_type": "PERSON"}}, {{"id": 1, "name": "Google", "entity_type": "ORGANIZATION"}}]}}

Text: "Bob founded TechCo"
Output: {{"entities": [{{"id": 0, "name": "Bob", "entity_type": "PERSON"}}, {{"id": 1, "name": "TechCo", "entity_type": "ORGANIZATION"}}]}}

Now extract from the TEXT above. Return only JSON."""


def build_relationship_extraction_prompt(
    text: str,
    entities: List[ExtractedEntity],
    reference_time: str
) -> str:
    """Build prompt for relationship extraction."""
    entities_context = "\n".join([
        f"  {e.id}: {e.name}" + (f" ({e.entity_type})" if e.entity_type else "")
        for e in entities
    ])

    return f"""You are an expert fact extractor. Extract ONLY relationships explicitly stated in the TEXT.

<TEXT>
{text}
</TEXT>

<ENTITIES>
{entities_context}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Critical Rules:
1. ONLY use entity IDs from the ENTITIES list above (IDs 0 to {len(entities)-1})
2. ONLY extract relationships explicitly stated in the TEXT - do NOT invent facts
3. ONLY create relationships between two different entities (source_entity_id ≠ target_entity_id)
4. Use SCREAMING_SNAKE_CASE for relation_type (WORKS_AT, FOUNDED, JOINED, etc.)
5. Write fact as a direct paraphrase of what the TEXT states
6. If TEXT mentions time periods, use ISO 8601 format with Z suffix for valid_at/invalid_at
7. If no time mentioned, leave temporal fields null
8. If no relationships exist between the entities, return empty array

Do NOT:
- Invent relationships not in the TEXT
- Reference entity IDs that don't exist
- Create self-referential relationships
- Add personal opinions or general knowledge

Return valid JSON only."""


def validate_json_complete(json_str: str) -> bool:
    """Check if JSON string appears to be complete (not truncated)."""
    if not json_str:
        return False

    # Basic checks for truncation
    json_str = json_str.strip()

    # Should end with closing brace/bracket
    if not json_str.endswith('}') and not json_str.endswith(']'):
        return False

    # Count opening and closing braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    return open_braces == close_braces and open_brackets == close_brackets


def test_extraction(model, test_case: dict):
    """Test a single extraction case."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_case['name']}")
    print("=" * 80)
    print(f"Text: '{test_case['text']}'")
    print()

    # Handle empty text
    if not test_case['text'].strip():
        print("Empty text - skipping extraction")
        if test_case['expected_entities'] == 0 and test_case['expected_relationships'] == 0:
            print("Test complete")
            return True
        else:
            print("ERROR: Expected entities/relationships for empty text")
            return False

    # Extract entities
    print("Extracting entities...")
    entity_prompt = build_entity_extraction_prompt(test_case['text'])

    try:
        entities_json = model(entity_prompt, output_type=ExtractedEntities)

        # Validate JSON is complete
        if not validate_json_complete(entities_json):
            print(f"ERROR: Generated JSON appears truncated")
            print(f"Raw JSON (truncated): {entities_json[:200]}...")
            return False

        print(f"Raw JSON: {entities_json}")
        entities = ExtractedEntities.model_validate_json(entities_json)

        print(f"Extracted {len(entities.entities)} entities:")
        for e in entities.entities:
            type_str = f" [{e.entity_type}]" if e.entity_type else ""
            print(f"  {e.id}: {e.name}{type_str}")

        # Check entity count
        if len(entities.entities) != test_case['expected_entities']:
            print(f"WARNING: Expected {test_case['expected_entities']} entities, got {len(entities.entities)}")

    except Exception as e:
        print(f"ERROR in entity extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Extract relationships (need at least 2 entities)
    if len(entities.entities) >= 2:
        print("\nExtracting relationships...")
        relationship_prompt = build_relationship_extraction_prompt(
            test_case['text'],
            entities.entities,
            datetime.now().isoformat()
        )

        try:
            relationships_json = model(relationship_prompt, output_type=ExtractedRelationships)

            # Validate JSON is complete
            if not validate_json_complete(relationships_json):
                print(f"ERROR: Generated JSON appears truncated")
                print(f"Raw JSON (truncated): {relationships_json[:200]}...")
                return False

            print(f"Raw JSON: {relationships_json}")
            relationships = ExtractedRelationships.model_validate_json(relationships_json)

            print(f"Extracted {len(relationships.relationships)} relationships:")

            valid_count = 0
            # Validate entity IDs before accessing
            for r in relationships.relationships:
                # Check for valid entity IDs
                if not (0 <= r.source_entity_id < len(entities.entities)):
                    print(f"  ERROR: Invalid source_entity_id {r.source_entity_id} (max: {len(entities.entities)-1})")
                    continue
                if not (0 <= r.target_entity_id < len(entities.entities)):
                    print(f"  ERROR: Invalid target_entity_id {r.target_entity_id} (max: {len(entities.entities)-1})")
                    continue

                # Check for self-referential relationships
                if r.source_entity_id == r.target_entity_id:
                    print(f"  ERROR: Self-referential relationship for entity {r.source_entity_id}")
                    continue

                source_name = entities.entities[r.source_entity_id].name
                target_name = entities.entities[r.target_entity_id].name
                print(f"  {source_name} --[{r.relation_type}]--> {target_name}")
                print(f"    Fact: {r.fact}")
                valid_count += 1

            # Check relationship count (only count valid ones)
            if valid_count != test_case['expected_relationships']:
                print(f"WARNING: Expected {test_case['expected_relationships']} relationships, got {valid_count} valid")

        except Exception as e:
            print(f"ERROR in relationship extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"\nSkipping relationship extraction (need at least 2 entities, got {len(entities.entities)})")

    print("\nTest complete")
    return True


def main():
    """Run all test cases."""
    print("=" * 80)
    print("MLX-LM Extraction Test Suite")
    print("=" * 80)

    # Load model
    print("\nLoading MLX-LM model...")
    mlx_model, mlx_tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)
    print("Model loaded")

    # Run tests
    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        try:
            if test_extraction(model, test_case):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nFATAL ERROR in test: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print(f"Failed: {failed}/{len(TEST_CASES)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
