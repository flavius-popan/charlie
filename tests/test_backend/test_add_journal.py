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

    # Parse _raw_response: [headers, [data_rows], stats]
    assert hasattr(result, '_raw_response')
    data_rows = result._raw_response[1] if len(result._raw_response) > 1 else []
    assert len(data_rows) == 1

    row = data_rows[0]
    # Each field is [type_id, value]
    assert row[0][1].decode('utf-8') == content  # content
    assert row[1][1].decode('utf-8') == DEFAULT_JOURNAL  # journal
    assert row[2][1].decode('utf-8') == uuid  # uuid


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
    # Parse _raw_response: [headers, [data_rows], stats]
    work_data_rows = work_result._raw_response[1] if len(work_result._raw_response) > 1 else []
    work_count = work_data_rows[0][0][1] if work_data_rows else 0
    assert work_count == 1

    # Query the personal graph
    personal_graph = db_utils.get_falkordb_graph("personal")
    personal_query = """
    MATCH (e:Episodic {group_id: 'personal'})
    RETURN count(e) as count
    """
    personal_result = personal_graph.query(personal_query)
    # Parse _raw_response: [headers, [data_rows], stats]
    personal_data_rows = personal_result._raw_response[1] if len(personal_result._raw_response) > 1 else []
    personal_count = personal_data_rows[0][0][1] if personal_data_rows else 0
    assert personal_count == 1


@pytest.mark.asyncio
async def test_self_entity_created(isolated_graph):
    """Test that SELF entity is created when first entry is added."""
    content = "First entry in journal"

    uuid = await add_journal_entry(content=content)

    self_query = f"""
    MATCH (self:Entity:Person {{uuid: {to_cypher_literal(str(SELF_ENTITY_UUID))}}})
    RETURN self.name as name, self.group_id as journal, self.uuid as uuid
    """
    result = isolated_graph.query(self_query)

    # Parse _raw_response: [headers, [data_rows], stats]
    assert hasattr(result, '_raw_response')
    data_rows = result._raw_response[1] if len(result._raw_response) > 1 else []
    assert len(data_rows) >= 1

    row = data_rows[0]
    assert row[0][1].decode('utf-8') == SELF_ENTITY_NAME  # name
    assert row[1][1].decode('utf-8') == DEFAULT_JOURNAL  # journal


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

    # Parse _raw_response: [headers, [data_rows], stats]
    data_rows = result._raw_response[1] if len(result._raw_response) > 1 else []
    entity_count = data_rows[0][0][1] if data_rows else 0

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
        data_rows = result._raw_response[1] if len(result._raw_response) > 1 else []
        count = data_rows[0][0][1] if data_rows else 0
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

    # Parse _raw_response: [headers, [data_rows], stats]
    data_rows = result._raw_response[1] if len(result._raw_response) > 1 else []
    count = data_rows[0][0][1] if data_rows else 0

    assert count == 3
