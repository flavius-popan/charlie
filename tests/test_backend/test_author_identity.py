"""Guardrails for canonical author identity handling ("I")."""

from __future__ import annotations

import pytest

from backend.database import is_self_entity_name


def test_is_self_entity_name_exact_matches_only() -> None:
    # Accept only first-person pronouns
    assert is_self_entity_name("I")
    assert is_self_entity_name("me")
    assert is_self_entity_name("My")
    assert is_self_entity_name("mine")
    assert is_self_entity_name("myself")

    # Reject near-matches or substrings
    for value in ["Priya", "Liam", "Ivy", "self", "", None]:
        assert is_self_entity_name(value) is False


def test_has_self_detection_uses_exact_token() -> None:
    entity_names = ["Priya", "Liam", "Ivy"]
    has_self = "i" in [e.lower() for e in entity_names]
    assert has_self is False, "Lowercasing should not misclassify names containing the letter 'i'"

    entity_names.append("I")
    has_self = "i" in [e.lower() for e in entity_names]
    assert has_self is True, "Detection should trigger only on the standalone author name 'I'"


@pytest.mark.asyncio
async def test_author_entity_seeded_with_canonical_name(isolated_graph) -> None:
    """Seeding ensures exactly one author entity named 'I' per journal."""

    from backend.database.persistence import ensure_self_entity
    from backend.database import (
        SELF_ENTITY_UUID,
        to_cypher_literal,
        get_falkordb_graph,
    )

    journal = "author-identity-guard"
    await ensure_self_entity(journal)

    query = f"""
    MATCH (self:Entity:Person {{uuid: {to_cypher_literal(str(SELF_ENTITY_UUID))}}})
    RETURN self.name AS name, count(self) AS count
    """
    graph = get_falkordb_graph(journal)
    result = graph.query(query)

    assert result.result_set, "Author entity should exist after seeding"

    row = result.result_set[0]
    name_val = row[0]
    count_val = row[1]

    name = name_val.decode("utf-8") if hasattr(name_val, "decode") else name_val
    count = int(count_val)

    assert name == "I"
    assert count == 1
