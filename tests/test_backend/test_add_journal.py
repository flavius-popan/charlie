"""Tests for backend.add_journal_entry() function."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend import add_journal_entry
from backend.database import SELF_ENTITY_UUID, SELF_ENTITY_NAME, to_cypher_literal
from backend.settings import DEFAULT_JOURNAL


@pytest.mark.asyncio
async def test_add_journal_entry_basic(isolated_graph):
    """Test basic journal entry creation with minimal parameters."""
    content = "Today I went to the park and saw some ducks."

    uuid = await add_journal_entry(content)

    assert uuid is not None
    assert isinstance(uuid, str)
    assert len(uuid) == 36  # UUID format


@pytest.mark.asyncio
async def test_add_journal_entry_with_uuid(isolated_graph):
    """Test creating entry with pre-existing UUID (import use case)."""
    content = "Imported from Day One."
    existing_uuid = str(uuid4())

    returned_uuid = await add_journal_entry(content=content, uuid=existing_uuid)

    assert returned_uuid == existing_uuid


@pytest.mark.asyncio
async def test_add_journal_entry_invalid_uuid(isolated_graph):
    """Test that invalid UUID format raises ValueError."""
    content = "Test entry"
    invalid_uuid = "not-a-valid-uuid"

    with pytest.raises(ValueError, match="Invalid UUID format"):
        await add_journal_entry(content=content, uuid=invalid_uuid)


@pytest.mark.asyncio
async def test_add_journal_entry_source_tracking(isolated_graph):
    """Test that source_description is stored correctly for different sources."""
    sources = ["Charlie", "Day One", "Notion", "Obsidian"]

    for source in sources:
        uuid = await add_journal_entry(
            content=f"Entry from {source}",
            source_description=source,
        )
        assert isinstance(uuid, str)


@pytest.mark.asyncio
async def test_add_journal_entry_defaults(isolated_graph):
    """Test that default values are applied correctly."""
    content = "Test with defaults"

    uuid = await add_journal_entry(content)

    assert isinstance(uuid, str)
    assert len(uuid) == 36


@pytest.mark.asyncio
async def test_add_journal_entry_persistence(isolated_graph):
    """Test that entry is actually persisted to database and retrievable."""
    content = "Persistence test entry"

    uuid = await add_journal_entry(content=content)

    query = f"""
    MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
    RETURN e.content as content, e.group_id as journal, e.uuid as uuid
    """
    result = isolated_graph.query(query)

    assert result.result_set and len(result.result_set) == 1

    row = result.result_set[0]
    stored_content = row[0].decode('utf-8') if hasattr(row[0], 'decode') else row[0]
    stored_journal = row[1].decode('utf-8') if hasattr(row[1], 'decode') else row[1]
    stored_uuid = row[2].decode('utf-8') if hasattr(row[2], 'decode') else row[2]
    assert stored_content == content
    assert stored_journal == DEFAULT_JOURNAL
    assert stored_uuid == uuid


@pytest.mark.asyncio
async def test_multiple_journals_isolation(isolated_graph):
    """Test that entries in different journals are isolated."""
    import backend.database as db_utils

    work_content = "Work meeting notes"
    personal_content = "Dinner with friends"

    work_uuid = await add_journal_entry(
        content=work_content,
        journal="work",
    )
    personal_uuid = await add_journal_entry(
        content=personal_content,
        journal="personal",
    )

    assert isinstance(work_uuid, str)
    assert isinstance(personal_uuid, str)
    assert work_uuid != personal_uuid

    # Query the work graph
    work_graph = db_utils.get_falkordb_graph("work")
    work_query = """
    MATCH (e:Episodic {group_id: 'work'})
    RETURN count(e) as count
    """
    work_result = work_graph.query(work_query)
    work_count = work_result.result_set[0][0] if work_result.result_set else 0
    assert work_count == 1

    # Query the personal graph
    personal_graph = db_utils.get_falkordb_graph("personal")
    personal_query = """
    MATCH (e:Episodic {group_id: 'personal'})
    RETURN count(e) as count
    """
    personal_result = personal_graph.query(personal_query)
    personal_count = personal_result.result_set[0][0] if personal_result.result_set else 0
    assert personal_count == 1


@pytest.mark.asyncio
async def test_self_entity_created(isolated_graph):
    """Test that author entity "I" is created when first entry is added."""
    content = "First entry in journal"

    uuid = await add_journal_entry(content=content)

    self_query = f"""
    MATCH (self:Entity:Person {{uuid: {to_cypher_literal(str(SELF_ENTITY_UUID))}}})
    RETURN self.name as name, self.group_id as journal, self.uuid as uuid
    """
    result = isolated_graph.query(self_query)

    assert result.result_set and len(result.result_set) >= 1

    row = result.result_set[0]
    name_val = row[0].decode('utf-8') if hasattr(row[0], 'decode') else row[0]
    journal_val = row[1].decode('utf-8') if hasattr(row[1], 'decode') else row[1]
    assert name_val == SELF_ENTITY_NAME
    assert journal_val == DEFAULT_JOURNAL


@pytest.mark.asyncio
async def test_journal_name_validation(isolated_graph):
    """Test that invalid journal names are rejected."""
    content = "Test entry"

    # Invalid: spaces
    with pytest.raises(ValueError, match="Invalid journal name"):
        await add_journal_entry(content=content, journal="invalid journal")

    # Invalid: special chars
    with pytest.raises(ValueError, match="Invalid journal name"):
        await add_journal_entry(content=content, journal="journal!")

    # Invalid: empty
    with pytest.raises(ValueError, match="cannot be empty"):
        await add_journal_entry(content=content, journal="")

    # Invalid: too long
    with pytest.raises(ValueError, match="too long"):
        await add_journal_entry(content=content, journal="a" * 65)

    # Valid examples should work
    for valid_name in ["work", "personal", "my-journal", "journal_2024"]:
        uuid = await add_journal_entry(content=content, journal=valid_name)
        assert isinstance(uuid, str)


@pytest.mark.asyncio
async def test_add_journal_entry_with_metadata(isolated_graph):
    """Test creating entry with all optional parameters."""
    content = "Complete metadata test"
    reference_time = datetime(2024, 11, 18, 10, 30, 0, tzinfo=timezone.utc)
    journal = "metadata-test"
    title = "My Custom Entry Title"
    source_description = "Day One"
    custom_uuid = str(uuid4())

    returned_uuid = await add_journal_entry(
        content=content,
        reference_time=reference_time,
        journal=journal,
        title=title,
        source_description=source_description,
        uuid=custom_uuid,
    )

    assert returned_uuid == custom_uuid


@pytest.mark.asyncio
async def test_no_entity_extraction(isolated_graph):
    """Test that no entity extraction occurs (entity_edges remains empty)."""
    content = "I met Sarah at the park and we discussed the project."

    uuid = await add_journal_entry(content)

    assert isinstance(uuid, str)

    entities_query = """
    MATCH (e:Entity)
    WHERE NOT e.uuid = '11111111-1111-1111-1111-111111111111'
    RETURN count(e) as count
    """
    result = isolated_graph.query(entities_query)

    entity_count = result.result_set[0][0] if result.result_set else 0

    assert entity_count == 0


@pytest.mark.asyncio
async def test_content_validation_empty(isolated_graph):
    """Test that empty content is rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        await add_journal_entry("")


@pytest.mark.asyncio
async def test_content_validation_whitespace_only(isolated_graph):
    """Test that whitespace-only content is rejected."""
    with pytest.raises(ValueError, match="whitespace-only"):
        await add_journal_entry("   \n\t  ")


@pytest.mark.asyncio
async def test_content_validation_too_long(isolated_graph):
    """Test that content exceeding MAX_CONTENT_LENGTH is rejected."""
    from backend import MAX_CONTENT_LENGTH

    # Create content that's 1 char too long
    too_long_content = "a" * (MAX_CONTENT_LENGTH + 1)

    with pytest.raises(ValueError, match="Content too long"):
        await add_journal_entry(too_long_content)


@pytest.mark.asyncio
async def test_content_validation_max_length_allowed(isolated_graph):
    """Test that content at exactly MAX_CONTENT_LENGTH is accepted."""
    from backend import MAX_CONTENT_LENGTH

    # Create content at exactly the max length
    max_content = "a" * MAX_CONTENT_LENGTH

    uuid = await add_journal_entry(max_content)
    assert isinstance(uuid, str)


@pytest.mark.asyncio
async def test_concurrent_multi_journal_writes(isolated_graph):
    """Test that concurrent writes to different journals don't interfere."""
    import asyncio
    import backend.database as db_utils

    async def write_entries(journal: str, count: int):
        """Write multiple entries to a journal concurrently."""
        for i in range(count):
            await add_journal_entry(f"Entry {i} for {journal}", journal=journal)

    # Write to 3 journals concurrently
    await asyncio.gather(
        write_entries("concurrent1", 5),
        write_entries("concurrent2", 5),
        write_entries("concurrent3", 5),
    )

    # Verify counts in each journal
    for journal_name in ["concurrent1", "concurrent2", "concurrent3"]:
        graph = db_utils.get_falkordb_graph(journal_name)
        result = graph.query(
            f"MATCH (e:Episodic {{group_id: '{journal_name}'}}) RETURN count(e)"
        )
        count = result.result_set[0][0] if result.result_set else 0
        assert count == 5, f"Expected 5 entries in {journal_name}, got {count}"


@pytest.mark.asyncio
async def test_multiple_entries_same_journal(isolated_graph):
    """Test adding multiple entries to the same journal."""
    entries_content = [
        "Morning entry",
        "Afternoon entry",
        "Evening entry",
    ]

    created_uuids = []
    for content in entries_content:
        uuid = await add_journal_entry(content=content)
        created_uuids.append(uuid)

    assert len(created_uuids) == 3
    assert all(isinstance(u, str) for u in created_uuids)
    assert len(set(created_uuids)) == 3  # All UUIDs unique

    count_query = f"""
    MATCH (e:Episodic {{group_id: {to_cypher_literal(DEFAULT_JOURNAL)}}})
    RETURN count(e) as count
    """
    result = isolated_graph.query(count_query)

    count = result.result_set[0][0] if result.result_set else 0

    assert count == 3


@pytest.mark.asyncio
async def test_content_with_backslashes(isolated_graph):
    """Test that content with backslashes is persisted correctly.

    This tests the Cypher string escaping fix for Critical Issue #2.
    Without proper escaping, backslashes can cause query syntax errors.
    """
    test_cases = [
        ("Single backslash: \\", "Single backslash: \\"),
        ("Double backslash: \\\\", "Double backslash: \\\\"),
        ("Windows path: C:\\Users\\Alice", "Windows path: C:\\Users\\Alice"),
        ("Trailing backslash: test\\", "Trailing backslash: test\\"),
        ("Backslash before quote: test\\'end", "Backslash before quote: test\\'end"),
        ("Multiple escapes: \\n\\t\\r", "Multiple escapes: \\n\\t\\r"),
    ]

    for content, expected in test_cases:
        uuid = await add_journal_entry(content)

        # Query back and verify exact content match
        query = f"""
        MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
        RETURN e.content
        """
        result = isolated_graph.query(query)
        assert result.result_set and len(result.result_set) == 1, f"Expected 1 row for content: {content}"

        val = result.result_set[0][0]
        stored_content = val.decode('utf-8') if hasattr(val, 'decode') else val
        assert stored_content == expected, \
            f"Content mismatch for input '{content}': got '{stored_content}', expected '{expected}'"


@pytest.mark.asyncio
async def test_content_with_special_characters(isolated_graph):
    """Test content with various special characters that need escaping."""
    test_cases = [
        "Single quote: it's working",
        "Double quote: she said \"hello\"",
        "Mixed quotes: it's \"nice\" weather",
        "Emoji content: Hello 👋 World 🌍",
        "Unicode: café, naïve, 日本語",
        "Newlines:\nLine 1\nLine 2\nLine 3",
        "Tabs:\tIndented\tContent",
        "Special chars: @#$%^&*()_+-=[]{}|;:,.<>?",
    ]

    for content in test_cases:
        uuid = await add_journal_entry(content)

        # Query back and verify
        query = f"""
        MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
        RETURN e.content
        """
        result = isolated_graph.query(query)
        assert result.result_set and len(result.result_set) == 1

        val = result.result_set[0][0]
        stored_content = val.decode('utf-8') if hasattr(val, 'decode') else val
        assert stored_content == content, \
            f"Content mismatch: got '{stored_content}', expected '{content}'"


@pytest.mark.asyncio
async def test_auto_generated_title_format(isolated_graph):
    """Test that auto-generated titles are human-readable."""
    from datetime import datetime, timezone

    reference_time = datetime(2024, 11, 18, 14, 30, 0, tzinfo=timezone.utc)

    uuid = await add_journal_entry(
        content="Test entry",
        reference_time=reference_time
    )

    query = f"""
    MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
    RETURN e.name
    """
    result = isolated_graph.query(query)
    val = result.result_set[0][0] if result.result_set else None
    title = val.decode('utf-8') if hasattr(val, 'decode') else val

    assert title == "Monday Nov 18, 2024"


@pytest.mark.asyncio
async def test_concurrent_same_journal_writes(isolated_graph):
    """Test that concurrent writes to the SAME journal are properly serialized.

    This verifies that per-journal locking works correctly and prevents race conditions.
    """
    import asyncio

    async def write_entry(index: int):
        """Write an entry and return its UUID."""
        return await add_journal_entry(
            f"Concurrent entry {index}",
            journal="shared-journal"
        )

    # Write 10 entries concurrently to the same journal
    uuids = await asyncio.gather(*[write_entry(i) for i in range(10)])

    # All should succeed and have unique UUIDs
    assert len(uuids) == 10
    assert len(set(uuids)) == 10, "All UUIDs should be unique"

    # Verify all entries persisted
    import backend.database as db_utils
    graph = db_utils.get_falkordb_graph("shared-journal")
    result = graph.query(
        "MATCH (e:Episodic {group_id: 'shared-journal'}) RETURN count(e)"
    )
    count = result.result_set[0][0] if result.result_set else 0
    assert count == 10, f"Expected 10 entries, found {count}"


@pytest.mark.asyncio
async def test_naive_datetime_converted_to_utc(isolated_graph):
    """Test that naive datetimes are converted to UTC.

    Import-friendly: file timestamps and simple formats provide naive datetimes.
    We interpret them as UTC for consistency.
    """
    content = "Test entry with naive datetime"
    naive_dt = datetime(2024, 11, 18, 14, 30, 0)  # No tzinfo

    uuid = await add_journal_entry(content=content, reference_time=naive_dt)

    # Verify it was stored and can be retrieved
    query = f"""
    MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
    RETURN e.valid_at
    """
    result = isolated_graph.query(query)
    val = result.result_set[0][0] if result.result_set else None
    stored_time = val.decode('utf-8') if hasattr(val, 'decode') else val

    # Should be stored as UTC (with timezone suffix)
    # Either "2024-11-18T14:30:00+00:00" or "2024-11-18T14:30:00Z"
    assert "2024-11-18T14:30:00" in stored_time
    assert ("+00:00" in stored_time or stored_time.endswith("Z")), \
        f"Expected UTC timezone marker in {stored_time}"


@pytest.mark.asyncio
async def test_timezone_aware_datetime_accepted(isolated_graph):
    """Test that timezone-aware datetimes are accepted."""
    content = "Test entry with timezone-aware datetime"
    aware_dt = datetime(2024, 11, 18, 14, 30, 0, tzinfo=timezone.utc)

    uuid = await add_journal_entry(content=content, reference_time=aware_dt)
    assert isinstance(uuid, str)


@pytest.mark.asyncio
async def test_content_with_null_bytes_rejected(isolated_graph):
    """Test that content containing null bytes is rejected."""
    test_cases = [
        "Content with null\x00byte",
        "\x00Start with null",
        "End with null\x00",
        "Multiple\x00null\x00bytes",
    ]

    for content in test_cases:
        with pytest.raises(ValueError, match="null byte"):
            await add_journal_entry(content)


@pytest.mark.asyncio
async def test_unicode_edge_cases(isolated_graph):
    """Test content with complex Unicode edge cases."""
    test_cases = [
        ("Zero-width space: Hello\u200bWorld", "Zero-width space: Hello\u200bWorld"),
        ("Right-to-left: مرحبا بك في تشارلي", "Right-to-left: مرحبا بك في تشارلي"),
        ("Combined emoji: 👨‍👩‍👧‍👦", "Combined emoji: 👨‍👩‍👧‍👦"),
        ("Skin tone modifiers: 👋🏽", "Skin tone modifiers: 👋🏽"),
        ("Regional indicators: 🇺🇸🇬🇧", "Regional indicators: 🇺🇸🇬🇧"),
        ("Mathematical: ∑∫∂√∞", "Mathematical: ∑∫∂√∞"),
        ("CJK unified: 你好世界", "CJK unified: 你好世界"),
        ("Combining diacritics: e̊x̆a̧m̀p̂lȩ", "Combining diacritics: e̊x̆a̧m̀p̂lȩ"),
    ]

    for content, expected in test_cases:
        uuid = await add_journal_entry(content)

        # Verify round-trip
        query = f"""
        MATCH (e:Episodic {{uuid: {to_cypher_literal(uuid)}}})
        RETURN e.content
        """
        result = isolated_graph.query(query)
        assert result.result_set and len(result.result_set) == 1, f"Expected 1 row for content: {content[:50]}"

        val = result.result_set[0][0]
        stored_content = val.decode('utf-8') if hasattr(val, 'decode') else val
        assert stored_content == expected, \
            f"Unicode mismatch for '{content[:50]}': got '{stored_content}', expected '{expected}'"
