"""Tests for database query functions."""

import json
from datetime import datetime, timezone

import pytest

from backend import add_journal_entry
from backend.database.queries import (
    delete_entity_mention,
    get_entry_entities,
    get_n_plus_one_neighbors,
    get_period_entities,
)
from backend.database.redis_ops import redis_ops, get_suppressed_entities
from backend.graph.extract_nodes import extract_nodes
from backend.settings import DEFAULT_JOURNAL


@pytest.fixture(autouse=True)
def clear_test_journal_suppression():
    """Clear suppression list for test_journal before each test."""
    with redis_ops() as r:
        suppression_key = "journal:test_journal:suppressed_entities"
        r.delete(suppression_key)

    yield

    with redis_ops() as r:
        suppression_key = "journal:test_journal:suppressed_entities"
        r.delete(suppression_key)


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_orphaned(isolated_graph, require_llm):
    """Should delete entity when it's only mentioned in this episode."""
    episode_uuid = await add_journal_entry(
        content="I met Sarah at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        assert nodes_json is not None
        entities = json.loads(nodes_json.decode())
        assert len(entities) > 0
        entity_uuid = entities[0]["uuid"]

    entity_name = entities[0]["name"]

    was_deleted = await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    assert was_deleted is True

    suppressed = get_suppressed_entities("test_journal")
    assert entity_name.lower() in suppressed, "Deleted entity should be globally suppressed"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_shared(isolated_graph, require_llm):
    """Should only remove MENTIONS edge when entity referenced elsewhere."""
    ep1_uuid = await add_journal_entry(
        content="I met Sarah today.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep1_uuid, journal="test_journal")

    ep2_uuid = await add_journal_entry(
        content="Sarah came over again.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep2_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{ep1_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        entities = json.loads(nodes_json.decode())
        sarah = next((e for e in entities if "Sarah" in e["name"]), None)
        assert sarah is not None

    was_deleted = await delete_entity_mention(ep1_uuid, sarah["uuid"], "test_journal")

    assert was_deleted is False

    suppressed = get_suppressed_entities("test_journal")
    assert "sarah" in suppressed, "Deleted entity should be globally suppressed even if not orphaned"


@pytest.mark.inference
@pytest.mark.asyncio
async def test_delete_entity_mention_updates_redis_cache(isolated_graph, require_llm):
    """Test that deletion removes entity from Redis cache."""
    episode_uuid = await add_journal_entry(
        content="I met Sarah and John at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        entities = json.loads(nodes_json.decode())
        initial_count = len(entities)
        assert initial_count >= 2

        entity_to_delete = entities[0]
        entity_uuid = entity_to_delete["uuid"]
        entity_name = entity_to_delete["name"]

    await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    with redis_ops() as r:
        cache_key = f"journal:test_journal:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        updated_entities = json.loads(nodes_json.decode())

        assert len(updated_entities) == initial_count - 1
        assert entity_uuid not in [e["uuid"] for e in updated_entities]
        assert entity_name not in [e["name"] for e in updated_entities]

    suppressed = get_suppressed_entities("test_journal")
    assert entity_name.lower() in suppressed, "Deleted entity should be globally suppressed"


@pytest.mark.asyncio
async def test_delete_entity_mention_sets_done_when_only_self_remains(isolated_graph, episode_uuid):
    """Deleting last visible entity should set status to done, even if 'I' remains."""
    from backend.database.queries import delete_entity_mention
    from backend.database.redis_ops import get_episode_status
    from backend.settings import DEFAULT_JOURNAL

    journal = DEFAULT_JOURNAL

    # Setup: Create episode node
    isolated_graph.query(f"""
        CREATE (:Episodic {{
            uuid: '{episode_uuid}',
            group_id: '{journal}',
            content: 'Test entry',
            name: 'Test',
            source: 'text',
            source_description: 'test',
            entity_edges: [],
            created_at: '2025-01-01T00:00:00Z',
            valid_at: '2025-01-01T00:00:00Z'
        }})
    """)

    # Create "I" entity and visible entity "Alice"
    isolated_graph.query(f"""
        CREATE (:Entity:Person {{uuid: 'self-uuid', group_id: '{journal}', name: 'I'}}),
               (:Entity:Person {{uuid: 'alice-uuid', group_id: '{journal}', name: 'Alice'}})
    """)

    # Create MENTIONS edges
    isolated_graph.query(f"""
        MATCH (ep:Episodic {{uuid: '{episode_uuid}'}}), (i:Entity {{uuid: 'self-uuid'}})
        CREATE (ep)-[:MENTIONS {{uuid: 'm1'}}]->(i)
    """)
    isolated_graph.query(f"""
        MATCH (ep:Episodic {{uuid: '{episode_uuid}'}}), (alice:Entity {{uuid: 'alice-uuid'}})
        CREATE (ep)-[:MENTIONS {{uuid: 'm2'}}]->(alice)
    """)

    # Setup Redis cache with both entities and pending_edges status
    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        r.hset(cache_key, "nodes", json.dumps([
            {"uuid": "self-uuid", "name": "I", "type": "Person"},
            {"uuid": "alice-uuid", "name": "Alice", "type": "Person"},
        ]))
        r.hset(cache_key, "mentions_edges", json.dumps(["m1", "m2"]))
        r.hset(cache_key, "status", "pending_edges")
        r.hset(cache_key, "journal", journal)

    # Act: Delete the visible entity (Alice)
    await delete_entity_mention(episode_uuid, "alice-uuid", journal)

    # Assert: Status should be "done" since only "I" remains
    status = get_episode_status(episode_uuid, journal)
    assert status == "done", f"Expected 'done' but got '{status}'"

    # Verify "I" is still in the nodes cache
    with redis_ops() as r:
        cache_key = f"journal:{journal}:{episode_uuid}"
        nodes_json = r.hget(cache_key, "nodes")
        nodes = json.loads(nodes_json.decode())
        assert len(nodes) == 1
        assert nodes[0]["name"] == "I"


class TestGetEntryEntities:
    """Tests for get_entry_entities query function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_cache(self, isolated_graph):
        """Should return empty list when no cache exists."""
        result = await get_entry_entities("nonexistent-uuid", DEFAULT_JOURNAL)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_entities_from_cache(self, isolated_graph, episode_uuid):
        """Should return entities from Redis cache."""
        # Setup cache with entities
        with redis_ops() as r:
            cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
            r.hset(cache_key, "nodes", json.dumps([
                {"uuid": "e1", "name": "Alice", "type": "Person"},
                {"uuid": "e2", "name": "Bob", "type": "Person"},
            ]))

        result = await get_entry_entities(episode_uuid, DEFAULT_JOURNAL)

        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_excludes_self_entity(self, isolated_graph, episode_uuid):
        """Should filter out the 'I' (self) entity."""
        with redis_ops() as r:
            cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
            r.hset(cache_key, "nodes", json.dumps([
                {"uuid": "self-uuid", "name": "I", "type": "Person"},
                {"uuid": "e1", "name": "Alice", "type": "Person"},
            ]))

        result = await get_entry_entities(episode_uuid, DEFAULT_JOURNAL)

        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_returns_empty_when_only_self(self, isolated_graph, episode_uuid):
        """Should return empty list when only 'I' entity exists."""
        with redis_ops() as r:
            cache_key = f"journal:{DEFAULT_JOURNAL}:{episode_uuid}"
            r.hset(cache_key, "nodes", json.dumps([
                {"uuid": "self-uuid", "name": "I", "type": "Person"},
            ]))

        result = await get_entry_entities(episode_uuid, DEFAULT_JOURNAL)

        assert result == []


class TestGetPeriodEntities:
    """Tests for get_period_entities query function."""

    @pytest.mark.asyncio
    async def test_returns_zeros_for_empty_period(self, isolated_graph):
        """Should return zero counts when no episodes in date range."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 2, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, DEFAULT_JOURNAL)

        assert result["entry_count"] == 0
        assert result["connection_count"] == 0
        assert result["top_entities"] == []

    @pytest.mark.asyncio
    async def test_counts_entries_in_period(self, isolated_graph):
        """Should count episodes within the date range."""
        journal = DEFAULT_JOURNAL

        # Create episodes in the graph
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-10-15T10:00:00Z',
                content: 'Test 1',
                name: 'Test 1'
            }})
        """)
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep2',
                group_id: '{journal}',
                valid_at: '2025-10-20T10:00:00Z',
                content: 'Test 2',
                name: 'Test 2'
            }})
        """)

        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, journal)

        assert result["entry_count"] == 2

    @pytest.mark.asyncio
    async def test_aggregates_entities_across_episodes(self, isolated_graph):
        """Should aggregate entity mentions across all episodes in period."""
        journal = DEFAULT_JOURNAL

        # Create episodes
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-10-15T10:00:00Z',
                content: 'Test 1',
                name: 'Test 1'
            }})
        """)
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep2',
                group_id: '{journal}',
                valid_at: '2025-10-20T10:00:00Z',
                content: 'Test 2',
                name: 'Test 2'
            }})
        """)

        # Setup Redis cache with entities
        with redis_ops() as r:
            r.hset(f"journal:{journal}:ep1", "nodes", json.dumps([
                {"uuid": "alice", "name": "Alice", "type": "Person"},
                {"uuid": "bob", "name": "Bob", "type": "Person"},
            ]))
            r.hset(f"journal:{journal}:ep2", "nodes", json.dumps([
                {"uuid": "alice", "name": "Alice", "type": "Person"},
                {"uuid": "charlie", "name": "Charlie", "type": "Person"},
            ]))

        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, journal)

        assert result["entry_count"] == 2
        assert result["connection_count"] == 4  # Alice x2, Bob, Charlie

    @pytest.mark.asyncio
    async def test_ranks_entities_by_frequency(self, isolated_graph):
        """Should rank top entities by mention frequency."""
        journal = DEFAULT_JOURNAL

        # Create episodes
        for i in range(3):
            isolated_graph.query(f"""
                CREATE (:Episodic {{
                    uuid: 'ep{i}',
                    group_id: '{journal}',
                    valid_at: '2025-10-{15+i}T10:00:00Z',
                    content: 'Test',
                    name: 'Test'
                }})
            """)

        # Alice appears 3 times, Bob 2 times, Charlie 1 time
        with redis_ops() as r:
            r.hset(f"journal:{journal}:ep0", "nodes", json.dumps([
                {"uuid": "alice", "name": "Alice", "type": "Person"},
                {"uuid": "bob", "name": "Bob", "type": "Person"},
            ]))
            r.hset(f"journal:{journal}:ep1", "nodes", json.dumps([
                {"uuid": "alice", "name": "Alice", "type": "Person"},
                {"uuid": "bob", "name": "Bob", "type": "Person"},
                {"uuid": "charlie", "name": "Charlie", "type": "Person"},
            ]))
            r.hset(f"journal:{journal}:ep2", "nodes", json.dumps([
                {"uuid": "alice", "name": "Alice", "type": "Person"},
            ]))

        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, journal)

        top_names = [e["name"] for e in result["top_entities"]]
        assert top_names[0] == "Alice"  # Most frequent
        assert top_names[1] == "Bob"    # Second most frequent
        assert top_names[2] == "Charlie"  # Least frequent

    @pytest.mark.asyncio
    async def test_excludes_self_entity_from_counts(self, isolated_graph):
        """Should not count 'I' (self) entity in aggregations."""
        journal = DEFAULT_JOURNAL

        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-10-15T10:00:00Z',
                content: 'Test',
                name: 'Test'
            }})
        """)

        with redis_ops() as r:
            r.hset(f"journal:{journal}:ep1", "nodes", json.dumps([
                {"uuid": "self", "name": "I", "type": "Person"},
                {"uuid": "alice", "name": "Alice", "type": "Person"},
            ]))

        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, journal)

        assert result["connection_count"] == 1  # Only Alice
        assert len(result["top_entities"]) == 1
        assert result["top_entities"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_returns_max_twenty_five_top_entities(self, isolated_graph):
        """Should return at most 25 top entities."""
        journal = DEFAULT_JOURNAL

        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-10-15T10:00:00Z',
                content: 'Test',
                name: 'Test'
            }})
        """)

        # Create 30 different entities
        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "type": "Thing"} for i in range(30)]
        with redis_ops() as r:
            r.hset(f"journal:{journal}:ep1", "nodes", json.dumps(entities))

        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, tzinfo=timezone.utc)

        result = await get_period_entities(start, end, journal)

        assert len(result["top_entities"]) == 25
        assert result["connection_count"] == 30


class TestGetNPlusOneNeighbors:
    """Tests for get_n_plus_one_neighbors query function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_source_uuids(self, isolated_graph):
        """Should return empty list when given no source UUIDs."""
        result = await get_n_plus_one_neighbors([], DEFAULT_JOURNAL)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_neighbors_exist(self, isolated_graph):
        """Should return empty list when source entities have no co-occurrences."""
        journal = DEFAULT_JOURNAL

        # Create isolated entity with no co-occurrences
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'lonely', group_id: '{journal}', name: 'Lonely'}})
        """)

        result = await get_n_plus_one_neighbors(["lonely"], journal)
        assert result == []

    @pytest.mark.asyncio
    async def test_finds_cooccurring_entities(self, isolated_graph):
        """Should find entities that co-occur with source entities."""
        journal = DEFAULT_JOURNAL

        # Create entities
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'alice', group_id: '{journal}', name: 'Alice'}}),
                   (:Entity {{uuid: 'bob', group_id: '{journal}', name: 'Bob'}}),
                   (:Entity {{uuid: 'charlie', group_id: '{journal}', name: 'Charlie'}})
        """)

        # Create episode mentioning Alice and Bob
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-01-15T10:00:00Z',
                content: 'Test',
                name: 'Test'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}), (alice:Entity {{uuid: 'alice'}}), (bob:Entity {{uuid: 'bob'}})
            CREATE (ep)-[:MENTIONS]->(alice), (ep)-[:MENTIONS]->(bob)
        """)

        # Query: Alice as source, should find Bob as neighbor
        result = await get_n_plus_one_neighbors(["alice"], journal)

        assert len(result) == 1
        assert result[0]["uuid"] == "bob"
        assert result[0]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_ranks_by_intersection_count(self, isolated_graph):
        """Should rank neighbors by how many source entities they connect to."""
        journal = DEFAULT_JOURNAL

        # Create entities: Alice, Bob, Charlie, Diana
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'alice', group_id: '{journal}', name: 'Alice'}}),
                   (:Entity {{uuid: 'bob', group_id: '{journal}', name: 'Bob'}}),
                   (:Entity {{uuid: 'charlie', group_id: '{journal}', name: 'Charlie'}}),
                   (:Entity {{uuid: 'diana', group_id: '{journal}', name: 'Diana'}})
        """)

        # Episode 1: Alice + Diana
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-01-15T10:00:00Z',
                content: 'Test 1',
                name: 'Test 1'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}), (alice:Entity {{uuid: 'alice'}}), (diana:Entity {{uuid: 'diana'}})
            CREATE (ep)-[:MENTIONS]->(alice), (ep)-[:MENTIONS]->(diana)
        """)

        # Episode 2: Bob + Diana
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep2',
                group_id: '{journal}',
                valid_at: '2025-01-16T10:00:00Z',
                content: 'Test 2',
                name: 'Test 2'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep2'}}), (bob:Entity {{uuid: 'bob'}}), (diana:Entity {{uuid: 'diana'}})
            CREATE (ep)-[:MENTIONS]->(bob), (ep)-[:MENTIONS]->(diana)
        """)

        # Episode 3: Alice + Charlie
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep3',
                group_id: '{journal}',
                valid_at: '2025-01-17T10:00:00Z',
                content: 'Test 3',
                name: 'Test 3'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep3'}}), (alice:Entity {{uuid: 'alice'}}), (charlie:Entity {{uuid: 'charlie'}})
            CREATE (ep)-[:MENTIONS]->(alice), (ep)-[:MENTIONS]->(charlie)
        """)

        # Query with Alice and Bob as sources
        # Diana connects to BOTH (intersection=2), Charlie only to Alice (intersection=1)
        result = await get_n_plus_one_neighbors(["alice", "bob"], journal)

        assert len(result) == 2
        assert result[0]["uuid"] == "diana"  # Higher intersection count
        assert result[0]["intersection_count"] == 2
        assert result[1]["uuid"] == "charlie"  # Lower intersection count
        assert result[1]["intersection_count"] == 1

    @pytest.mark.asyncio
    async def test_respects_exclude_uuids(self, isolated_graph):
        """Should exclude specified UUIDs from results."""
        journal = DEFAULT_JOURNAL

        # Create entities
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'alice', group_id: '{journal}', name: 'Alice'}}),
                   (:Entity {{uuid: 'bob', group_id: '{journal}', name: 'Bob'}}),
                   (:Entity {{uuid: 'charlie', group_id: '{journal}', name: 'Charlie'}})
        """)

        # Episode mentioning all three
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-01-15T10:00:00Z',
                content: 'Test',
                name: 'Test'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}),
                  (alice:Entity {{uuid: 'alice'}}),
                  (bob:Entity {{uuid: 'bob'}}),
                  (charlie:Entity {{uuid: 'charlie'}})
            CREATE (ep)-[:MENTIONS]->(alice),
                   (ep)-[:MENTIONS]->(bob),
                   (ep)-[:MENTIONS]->(charlie)
        """)

        # Query with exclusion
        result = await get_n_plus_one_neighbors(
            ["alice"],
            journal,
            exclude_uuids={"bob"}
        )

        assert len(result) == 1
        assert result[0]["uuid"] == "charlie"

    @pytest.mark.asyncio
    async def test_excludes_self_entity(self, isolated_graph):
        """Should always exclude the 'I' (self) entity."""
        journal = DEFAULT_JOURNAL

        # Create entities including "I"
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'alice', group_id: '{journal}', name: 'Alice'}}),
                   (:Entity {{uuid: 'self', group_id: '{journal}', name: 'I'}}),
                   (:Entity {{uuid: 'bob', group_id: '{journal}', name: 'Bob'}})
        """)

        # Episode mentioning Alice, I, and Bob
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2025-01-15T10:00:00Z',
                content: 'Test',
                name: 'Test'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}),
                  (alice:Entity {{uuid: 'alice'}}),
                  (self:Entity {{uuid: 'self'}}),
                  (bob:Entity {{uuid: 'bob'}})
            CREATE (ep)-[:MENTIONS]->(alice),
                   (ep)-[:MENTIONS]->(self),
                   (ep)-[:MENTIONS]->(bob)
        """)

        result = await get_n_plus_one_neighbors(["alice"], journal)

        # Should only return Bob, not "I"
        assert len(result) == 1
        assert result[0]["uuid"] == "bob"

    @pytest.mark.asyncio
    async def test_orders_by_inverse_recency_for_ties(self, isolated_graph):
        """Should order by oldest co-occurrence when intersection counts are equal."""
        journal = DEFAULT_JOURNAL

        # Create entities
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'alice', group_id: '{journal}', name: 'Alice'}}),
                   (:Entity {{uuid: 'bob', group_id: '{journal}', name: 'Bob'}}),
                   (:Entity {{uuid: 'charlie', group_id: '{journal}', name: 'Charlie'}})
        """)

        # Episode 1 (older): Alice + Charlie
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep1',
                group_id: '{journal}',
                valid_at: '2024-01-15T10:00:00Z',
                content: 'Test 1',
                name: 'Test 1'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}), (alice:Entity {{uuid: 'alice'}}), (charlie:Entity {{uuid: 'charlie'}})
            CREATE (ep)-[:MENTIONS]->(alice), (ep)-[:MENTIONS]->(charlie)
        """)

        # Episode 2 (newer): Alice + Bob
        isolated_graph.query(f"""
            CREATE (:Episodic {{
                uuid: 'ep2',
                group_id: '{journal}',
                valid_at: '2025-01-15T10:00:00Z',
                content: 'Test 2',
                name: 'Test 2'
            }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep2'}}), (alice:Entity {{uuid: 'alice'}}), (bob:Entity {{uuid: 'bob'}})
            CREATE (ep)-[:MENTIONS]->(alice), (ep)-[:MENTIONS]->(bob)
        """)

        result = await get_n_plus_one_neighbors(["alice"], journal)

        # Both have intersection_count=1, Charlie's co-occurrence is older
        assert len(result) == 2
        assert result[0]["uuid"] == "charlie"  # Older co-occurrence
        assert result[1]["uuid"] == "bob"  # Newer co-occurrence

    @pytest.mark.asyncio
    async def test_respects_limit(self, isolated_graph):
        """Should respect the limit parameter."""
        journal = DEFAULT_JOURNAL

        # Create many entities
        for i in range(10):
            isolated_graph.query(f"""
                CREATE (:Entity {{uuid: 'entity{i}', group_id: '{journal}', name: 'Entity{i}'}})
            """)

        # Create episode mentioning source and all entities
        isolated_graph.query(f"""
            CREATE (:Entity {{uuid: 'source', group_id: '{journal}', name: 'Source'}}),
                   (:Episodic {{
                       uuid: 'ep1',
                       group_id: '{journal}',
                       valid_at: '2025-01-15T10:00:00Z',
                       content: 'Test',
                       name: 'Test'
                   }})
        """)
        isolated_graph.query(f"""
            MATCH (ep:Episodic {{uuid: 'ep1'}}), (source:Entity {{uuid: 'source'}})
            CREATE (ep)-[:MENTIONS]->(source)
        """)
        for i in range(10):
            isolated_graph.query(f"""
                MATCH (ep:Episodic {{uuid: 'ep1'}}), (e:Entity {{uuid: 'entity{i}'}})
                CREATE (ep)-[:MENTIONS]->(e)
            """)

        result = await get_n_plus_one_neighbors(["source"], journal, limit=3)

        assert len(result) == 3
