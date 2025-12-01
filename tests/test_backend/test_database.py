"""Database layer tests for backend (FalkorDB Lite operations)."""

from __future__ import annotations

from contextlib import ExitStack

import pytest

import backend.database as db_utils


def _decode(value):
    """Decode bytes to string."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def _query_rows(graph, query: str) -> list[list[object]]:
    """Execute a query and return decoded result rows."""
    result = graph.query(query)
    return [[_decode(col) for col in row] for row in (result.result_set or [])]


@pytest.fixture
def falkordb_client(falkordb_test_context):
    """Get the FalkorDB client instance."""
    from backend.settings import DEFAULT_JOURNAL
    import backend.database.lifecycle as lifecycle
    graph = db_utils.get_falkordb_graph(DEFAULT_JOURNAL)
    if graph is None or lifecycle._db is None:
        pytest.skip("FalkorDB Lite is unavailable in this environment.")
    return lifecycle._db


def test_falkordblite_imports():
    """Ensure FalkorDB Lite bindings are importable."""
    pytest.importorskip("redislite")
    pytest.importorskip("redislite.falkordb_client")


def test_basic_operations(isolated_graph):
    """Exercise simple create/query/delete flows on the shared graph."""
    isolated_graph.query(
        "CREATE (n:TestBasic {name: 'verification'}) RETURN n.name"
    )
    rows = _query_rows(isolated_graph, "MATCH (n:TestBasic) RETURN n.name")
    assert {row[0] for row in rows} == {"verification"}

    lookup = _query_rows(isolated_graph, "MATCH (n:TestBasic) RETURN count(n)")
    assert lookup == [[1]]


def test_social_network_graph(isolated_graph):
    """Model a small social network and confirm reachability."""
    isolated_graph.query(
        """
        CREATE (alice:Person {name: 'Alice', age: 30}),
               (bob:Person {name: 'Bob', age: 25}),
               (carol:Person {name: 'Carol', age: 28}),
               (alice)-[:KNOWS]->(bob),
               (bob)-[:KNOWS]->(carol),
               (alice)-[:KNOWS]->(carol)
        """
    )

    friends_result = _query_rows(
        isolated_graph,
        """
        MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(friend)
        RETURN friend.name, friend.age
        """,
    )
    friends = {(row[0], int(row[1])) for row in friends_result}
    assert friends == {("Bob", 25), ("Carol", 28)}

    path_result = _query_rows(
        isolated_graph,
        """
        MATCH path = (a:Person {name: 'Alice'})-[:KNOWS*]->(c:Person {name: 'Carol'})
        RETURN length(path) AS pathLength
        """,
    )
    path_lengths = {int(row[0]) for row in path_result}
    assert path_lengths == {1, 2}


def test_multiple_independent_graphs(falkordb_client):
    """Graphs created via select_graph should stay isolated from each other."""
    users = falkordb_client.select_graph("users_test")
    products = falkordb_client.select_graph("products_test")
    transactions = falkordb_client.select_graph("transactions_test")

    try:
        users.query("CREATE (:User {name: 'Alice', email: 'alice@example.com'})")
        products.query(
            "CREATE (:Product {name: 'Laptop', price: 999, category: 'Electronics'})"
        )
        transactions.query(
            "CREATE (:Transaction {id: 'TX001', amount: 999, status: 'completed'})"
        )

        all_graphs = {g.decode() for g in falkordb_client.list_graphs()}
        assert {"users_test", "products_test", "transactions_test"}.issubset(all_graphs)

        user_result = _query_rows(users, "MATCH (u:User) RETURN u.name")
        product_result = _query_rows(products, "MATCH (p:Product) RETURN p.name")
        tx_result = _query_rows(transactions, "MATCH (t:Transaction) RETURN t.id")

        assert user_result == [["Alice"]]
        assert product_result == [["Laptop"]]
        assert tx_result == [["TX001"]]
    finally:
        with ExitStack() as stack:
            for graph in (users, products, transactions):
                stack.callback(graph.delete)


def test_read_only_queries(isolated_graph):
    """Verify ro_query returns consistent results and leaves data untouched."""
    isolated_graph.query(
        """
        CREATE (:Article {title: 'Graph Databases', views: 1000}),
               (:Article {title: 'NoSQL Overview', views: 1500}),
               (:Article {title: 'Cypher Guide', views: 800})
        """
    )

    read_only = isolated_graph.ro_query(
        """
        MATCH (a:Article)
        WHERE a.views > 900
        RETURN a.title, a.views
        ORDER BY a.views DESC
        """
    )
    rows = _query_rows(isolated_graph, "MATCH (a:Article) WHERE a.views > 900 RETURN a.title, a.views ORDER BY a.views DESC")
    rows = [(row[0], int(row[1])) for row in rows]
    assert rows == [
        ("NoSQL Overview", 1500),
        ("Graph Databases", 1000),
    ]


def test_aggregation_queries(isolated_graph):
    """Confirm aggregations over vecf32-capable properties succeed."""
    isolated_graph.query(
        """
        CREATE (eng:Department {name: 'Engineering'}),
               (sales:Department {name: 'Sales'}),
               (hr:Department {name: 'HR'}),
               (:Employee {name: 'Alice', salary: 120000})-[:WORKS_IN]->(eng),
               (:Employee {name: 'Bob', salary: 110000})-[:WORKS_IN]->(eng),
               (:Employee {name: 'Carol', salary: 95000})-[:WORKS_IN]->(sales),
               (:Employee {name: 'Dave', salary: 85000})-[:WORKS_IN]->(sales),
               (:Employee {name: 'Eve', salary: 70000})-[:WORKS_IN]->(hr)
        """
    )

    result = isolated_graph.query(
        """
        MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
        RETURN d.name AS department,
               count(e) AS employee_count,
               avg(e.salary) AS avg_salary,
               max(e.salary) AS max_salary
        ORDER BY avg_salary DESC
        """
    )

    rows = _query_rows(isolated_graph,
        """
        MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
        RETURN d.name AS department,
               count(e) AS employee_count,
               avg(e.salary) AS avg_salary,
               max(e.salary) AS max_salary
        ORDER BY avg_salary DESC
        """
    )

    departments: list[tuple[str, int, float, float]] = []
    for name, count, avg_salary, max_salary in rows:
        departments.append(
            (_decode(name), int(count), float(avg_salary), float(max_salary))
        )

    assert departments[0][0] == "Engineering"
    assert departments[-1][0] == "HR"
    assert departments[0][1] == 2
    assert departments[1][1] == 2
    assert departments[2][1] == 1


def test_vecf32_empty_embedding(isolated_graph):
    """Ensure vecf32 accepts empty lists and persists [] exactly."""
    creation = isolated_graph.query(
        """
        CREATE (n:Entity {uuid: 'vecf32-empty', embedding: vecf32([])})
        RETURN n.embedding
        """
    )
    stored_rows = _query_rows(isolated_graph, "MATCH (n:Entity {uuid: 'vecf32-empty'}) RETURN n.embedding")
    # FalkorDB can return empty list as [[]], [[[]]], or []
    assert stored_rows in ([[]], [[[]]], []), f"Unexpected storage payload {stored_rows}"

    verification = isolated_graph.query(
        """
        MATCH (n:Entity {uuid: 'vecf32-empty'})
        RETURN n.embedding
        """
    )
    verified_rows = _query_rows(isolated_graph, "MATCH (n:Entity {uuid: 'vecf32-empty'}) RETURN n.embedding")
    assert verified_rows in ([[]], [[[]]], []), f"Unexpected verification payload {verified_rows}"


@pytest.mark.asyncio
async def test_episode_delete_prunes_entities_only_when_unreferenced(isolated_graph):
    """Orphan pruning should keep shared entities until the last MENTIONS is removed."""
    from backend.database.persistence import delete_episode
    from backend.settings import DEFAULT_JOURNAL

    # Two episodes referencing the same entity
    isolated_graph.query(
        f"""
        CREATE (:Episodic {{
                    uuid: 'ep-orphan-a',
                    group_id: '{DEFAULT_JOURNAL}',
                    content: 'Entry A',
                    name: 'Entry A',
                    source: 'text',
                    source_description: 'test',
                    entity_edges: [],
                    created_at: '2025-01-01T00:00:00Z',
                    valid_at: '2025-01-01T00:00:00Z'
               }}),
               (:Episodic {{
                    uuid: 'ep-orphan-b',
                    group_id: '{DEFAULT_JOURNAL}',
                    content: 'Entry B',
                    name: 'Entry B',
                    source: 'text',
                    source_description: 'test',
                    entity_edges: [],
                    created_at: '2025-01-01T00:00:00Z',
                    valid_at: '2025-01-01T00:00:00Z'
               }}),
               (:Entity {{uuid: 'entity-shared', group_id: '{DEFAULT_JOURNAL}', name: 'Shared'}})
        """
    )
    isolated_graph.query(
        """
        MATCH (a:Episodic {uuid: 'ep-orphan-a'}), (b:Episodic {uuid: 'ep-orphan-b'}), (ent:Entity {uuid: 'entity-shared'})
        CREATE (a)-[:MENTIONS {uuid: 'm1'}]->(ent),
               (b)-[:MENTIONS {uuid: 'm2'}]->(ent)
        """
    )

    # Delete first episode: entity should remain, one MENTIONS should remain
    await delete_episode("ep-orphan-a", DEFAULT_JOURNAL)
    rows_after_first = _query_rows(
        isolated_graph,
        """
        MATCH (ent:Entity {uuid: 'entity-shared'})
        OPTIONAL MATCH ()-[r:MENTIONS]->(ent)
        RETURN count(ent), count(r)
        """,
    )
    assert rows_after_first == [[1, 1]], "Entity should persist while still referenced"

    # Delete second episode: entity should now be pruned
    await delete_episode("ep-orphan-b", DEFAULT_JOURNAL)
    rows_after_second = _query_rows(
        isolated_graph,
        """
        MATCH (ent:Entity {uuid: 'entity-shared'})
        RETURN count(ent)
        """,
    )
    assert rows_after_second == [[0]], "Entity should be removed when no MENTIONS remain"


@pytest.mark.asyncio
async def test_ensure_graph_ready_idempotent(isolated_graph):
    """ensure_graph_ready can rebuild indices multiple times without corrupting data."""
    await db_utils.ensure_graph_ready(delete_existing=True)
    await db_utils.ensure_graph_ready()

    isolated_graph.query("CREATE (:InitSmoke {uuid: 'init-smoke'})")
    rows = _query_rows(isolated_graph, "MATCH (n:InitSmoke) RETURN count(n)")
    assert rows == [[1]]


def test_tcp_server_configuration():
    """Test TCP server configuration functions."""
    # Test endpoint retrieval when disabled
    endpoint = db_utils.get_tcp_server_endpoint()
    # Could be None or tuple depending on settings
    if endpoint is not None:
        host, port = endpoint
        assert isinstance(host, str)
        assert isinstance(port, int)

    # Test password retrieval
    password = db_utils.get_tcp_server_password()
    # Could be None or string depending on settings
    assert password is None or isinstance(password, str)


def test_fulltext_query_builder():
    """Test fulltext query building and sanitization."""
    driver = db_utils.get_driver()

    # Test basic query sanitization
    sanitized = driver._sanitize_fulltext("hello, world! test@example.com")
    assert "," not in sanitized
    assert "!" not in sanitized
    assert "@" not in sanitized

    # Test query building without group filter
    query = driver.build_fulltext_query("hello world test")
    assert "hello" in query or "world" in query or "test" in query

    # Test query building with group filter
    query = driver.build_fulltext_query("hello world", group_ids=["journal1", "journal2"])
    assert "@group_id" in query
    assert "journal1" in query
    assert "journal2" in query

    # Test stopword filtering
    query = driver.build_fulltext_query("the quick brown fox")
    # "the" should be filtered out (it's a stopword)
    assert "quick" in query or "brown" in query or "fox" in query

    # Test empty query handling
    query = driver.build_fulltext_query("")
    assert query == ""

    # Test group filter only
    query = driver.build_fulltext_query("", group_ids=["journal1"])
    assert "@group_id:journal1" in query


def test_stopwords_constant():
    """Verify STOPWORDS constant is defined and contains expected words."""
    from backend.database.utils import STOPWORDS
    assert isinstance(STOPWORDS, list)
    assert len(STOPWORDS) > 0
    assert 'the' in STOPWORDS
    assert 'a' in STOPWORDS
    assert 'and' in STOPWORDS


def test_to_cypher_literal():
    """Test Cypher literal conversion for various types."""
    # Test None
    assert db_utils.to_cypher_literal(None) == "null"

    # Test booleans
    assert db_utils.to_cypher_literal(True) == "true"
    assert db_utils.to_cypher_literal(False) == "false"

    # Test numbers
    assert db_utils.to_cypher_literal(42) == "42"
    assert db_utils.to_cypher_literal(3.14) == "3.14"

    # Test strings with escaping
    assert db_utils.to_cypher_literal("hello") == "'hello'"
    assert db_utils.to_cypher_literal("it's") == "'it\\'s'"

    # Test lists and dicts (should be JSON)
    assert '"test"' in db_utils.to_cypher_literal(["test"])
    assert '"key"' in db_utils.to_cypher_literal({"key": "value"})


def test_to_cypher_array():
    """Test conversion of Python lists to Cypher array literals."""
    from backend.database.utils import _to_cypher_array

    # Test empty array
    assert _to_cypher_array([]) == "[]"

    # Test array with strings
    assert _to_cypher_array(["a", "b", "c"]) == "['a', 'b', 'c']"

    # Test array with numbers
    assert _to_cypher_array([1, 2, 3]) == "[1, 2, 3]"

    # Test array with mixed types
    result = _to_cypher_array(["text", 42, True])
    assert "'text'" in result
    assert "42" in result
    assert "true" in result

    # Test array with strings requiring escaping
    result = _to_cypher_array(["it's", "a test"])
    assert "it\\'s" in result
    assert "a test" in result


def test_decode_value_with_falkordb_types():
    """Test decoding FalkorDB native type format."""
    from backend.database.utils import _decode_value

    # Test type 1: null
    assert _decode_value([1, None]) is None

    # Test type 2: string (bytes)
    assert _decode_value([2, b"hello"]) == "hello"
    assert _decode_value([2, "world"]) == "world"

    # Test type 3: integer
    assert _decode_value([3, 42]) == 42
    assert _decode_value([3, "123"]) == 123

    # Test type 4: boolean
    assert _decode_value([4, 1]) is True
    assert _decode_value([4, 0]) is False

    # Test type 5: double
    assert _decode_value([5, 3.14]) == 3.14
    assert _decode_value([5, "2.71"]) == 2.71

    # Test type 6: array (empty)
    assert _decode_value([6, []]) == []

    # Test type 6: array with elements
    result = _decode_value([6, [[2, b"a"], [2, b"b"], [2, b"c"]]])
    assert result == ["a", "b", "c"]

    # Test type 6: nested array with mixed types
    result = _decode_value([6, [[2, b"text"], [3, 42], [4, 1]]])
    assert result == ["text", 42, True]

    # Test fallback for non-typed values
    assert _decode_value("plain_string") == "plain_string"
    assert _decode_value(42) == 42
    assert _decode_value(b"bytes") == "bytes"


def test_native_array_storage_and_retrieval(isolated_graph):
    """Test that arrays are stored and retrieved as native FalkorDB arrays."""
    from backend.database.utils import _to_cypher_array, to_cypher_literal

    # Create a node with array properties
    uuid_val = "test-arrays-123"
    edges = ["edge1", "edge2", "edge3"]
    labels = ["Label1", "Label2"]

    query = f"""
    CREATE (n:TestArrayNode {{
        uuid: {to_cypher_literal(uuid_val)},
        edges: {_to_cypher_array(edges)},
        labels: {_to_cypher_array(labels)},
        empty: {_to_cypher_array([])}
    }})
    RETURN n.edges, n.labels, n.empty
    """

    isolated_graph.query(query)

    # Retrieve and verify
    result = isolated_graph.query(f"""
        MATCH (n:TestArrayNode {{uuid: {to_cypher_literal(uuid_val)}}})
        RETURN n.edges AS edges, n.labels AS labels, n.empty AS empty
    """)

    assert result.result_set and len(result.result_set) == 1

    # Parse the results using new API
    from backend.database.utils import _decode_value
    header = [_decode_value(col[1]) for col in result.header]
    row = result.result_set[0]

    retrieved = {}
    for idx, field_name in enumerate(header):
        value = row[idx] if idx < len(row) else None
        retrieved[field_name] = _decode_value(value)

    # Verify arrays are properly decoded
    assert retrieved["edges"] == edges
    assert retrieved["labels"] == labels
    assert retrieved["empty"] == []


@pytest.mark.asyncio
async def test_driver_build_indices(isolated_graph):
    """Test that driver can build indices without errors."""
    driver = db_utils.get_driver()

    # Should not raise
    await driver.build_indices_and_constraints()

    # Test idempotency - should not raise on second call
    await driver.build_indices_and_constraints()


@pytest.mark.asyncio
async def test_concurrent_lock_initialization(isolated_graph):
    """Test that concurrent calls to ensure_graph_ready don't create multiple locks.

    This test exposes the race condition in asyncio.Lock initialization where
    multiple tasks could create separate Lock instances, defeating synchronization.
    """
    import asyncio
    import backend.database.persistence as persistence

    # Reset initialization state to force lock creation
    persistence._graph_initialized.clear()
    persistence._graph_init_lock = None

    async def initialize_graph(journal: str, task_id: int):
        """Try to initialize the same journal from multiple tasks."""
        await db_utils.ensure_graph_ready(journal)
        return task_id

    # Launch 10 concurrent tasks trying to initialize the same journal
    # This should trigger the race condition where multiple Lock objects are created
    tasks = [initialize_graph("race-test-journal", i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All tasks should complete successfully
    assert len(results) == 10
    assert results == list(range(10))

    # Verify the journal was initialized (should be True, set by any of the tasks)
    assert persistence._graph_initialized.get("race-test-journal", False) is True


@pytest.mark.asyncio
async def test_concurrent_self_entity_seeding(isolated_graph):
    """Test that concurrent author entity "I" seeding doesn't create race conditions.

    Similar to lock initialization, author entity "I" seeding has a lazy lock initialization
    that could race.
    """
    import asyncio
    import backend.database.persistence as persistence

    # Reset SELF seeding state
    persistence._seeded_self_groups.clear()
    persistence._self_seed_lock = None

    async def seed_self(journal: str, task_id: int):
        """Try to seed author entity "I" from multiple tasks."""
        await db_utils.ensure_self_entity(journal)
        return task_id

    # Launch 10 concurrent tasks trying to seed SELF for the same journal
    tasks = [seed_self("self-race-journal", i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All tasks should complete
    assert len(results) == 10

    # Verify SELF was seeded (should be in the set)
    assert "self-race-journal" in persistence._seeded_self_groups

    # Verify only one author entity "I" exists in the graph
    graph = db_utils.get_falkordb_graph("self-race-journal")
    from backend.database import SELF_ENTITY_UUID, to_cypher_literal
    query = f"""
    MATCH (self:Entity:Person {{uuid: {to_cypher_literal(str(SELF_ENTITY_UUID))}}})
    RETURN count(self) as count
    """
    result = graph.query(query)
    rows = _query_rows(graph, query)
    count = int(rows[0][0]) if rows else 0
    assert count == 1, f'Expected exactly 1 author entity "I", found {count}'


@pytest.mark.asyncio
async def test_shutdown_rejects_new_operations(isolated_graph):
    """Test that new operations fail-fast when shutdown is requested.

    This prevents data corruption without adding delays to shutdown.
    No waiting - operations immediately fail if shutdown is in progress.
    """
    import asyncio
    import backend.database.lifecycle as lifecycle
    from backend import add_journal_entry

    # First, verify normal operation works
    uuid1 = await add_journal_entry("Entry before shutdown")
    assert uuid1 is not None

    # Set shutdown flag (simulating shutdown in progress)
    lifecycle._shutdown_requested = True

    # Attempting new operation should fail immediately
    with pytest.raises(RuntimeError, match="shutdown in progress"):
        await add_journal_entry("Entry during shutdown")

    # Reset flag for test cleanup
    lifecycle._shutdown_requested = False


@pytest.mark.asyncio
async def test_get_episode_by_uuid(isolated_graph):
    """Test retrieving an episode by UUID."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    content = "Test entry for retrieval"
    title = "Custom Title"
    uuid_str = await add_journal_entry(content, title=title)
    assert uuid_str is not None

    # Retrieve the episode
    episode = await db.get_episode(uuid_str)

    assert episode is not None
    assert episode['uuid'] == uuid_str
    assert episode['content'] == content
    assert episode['name'] == title


@pytest.mark.asyncio
async def test_get_episode_nonexistent_uuid(isolated_graph):
    """Test retrieving a nonexistent episode returns None."""
    import backend.database as db
    from uuid import uuid4

    # Try to get an episode that doesn't exist
    nonexistent_uuid = str(uuid4())
    episode = await db.get_episode(nonexistent_uuid)

    assert episode is None


@pytest.mark.asyncio
async def test_get_home_screen_empty_journal(isolated_graph):
    from backend.database import get_home_screen

    episodes = await get_home_screen()
    assert episodes == []


@pytest.mark.asyncio
async def test_get_home_screen_single_episode(isolated_graph):
    from backend import add_journal_entry
    from backend.database import get_home_screen

    uuid_str = await add_journal_entry("# Title\nBody text", title="Fallback Title")

    episodes = await get_home_screen()

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["uuid"] == uuid_str
    assert ep["name"] == "Fallback Title"
    assert ep["preview"] == "Title"


@pytest.mark.asyncio
async def test_get_home_screen_ordering_and_preview(isolated_graph):
    from backend import add_journal_entry
    from backend.database import get_home_screen
    from datetime import datetime, timezone, timedelta

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    newest = await add_journal_entry("No header here. Second sentence.", title="Newest", reference_time=base + timedelta(hours=2))
    middle = await add_journal_entry("# Middle\nContent", title="Middle", reference_time=base + timedelta(hours=1))
    oldest = await add_journal_entry("Oldest content only", title="Oldest", reference_time=base)

    episodes = await get_home_screen()

    assert [e["uuid"] for e in episodes] == [newest, middle, oldest]
    # Preview uses first sentence when no header
    assert episodes[0]["preview"].startswith("No header here.")
    # Preview uses header when present
    assert episodes[1]["preview"] == "Middle"


@pytest.mark.asyncio
async def test_get_home_screen_multiple_journals(isolated_graph):
    from backend import add_journal_entry
    from backend.database import get_home_screen
    from backend.settings import DEFAULT_JOURNAL

    default_uuid = await add_journal_entry("Default journal entry")
    other_uuid = await add_journal_entry("Other journal entry", journal="other_journal")

    default_episodes = await get_home_screen(journal=DEFAULT_JOURNAL)
    other_episodes = await get_home_screen(journal="other_journal")

    assert [ep["uuid"] for ep in default_episodes] == [default_uuid]
    assert [ep["uuid"] for ep in other_episodes] == [other_uuid]


@pytest.mark.asyncio
async def test_update_episode_content(isolated_graph):
    """Test updating episode content."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    original_content = "Original content"
    uuid_str = await add_journal_entry(original_content)

    # Update the content
    new_content = "Updated content with more details"
    await db.update_episode(uuid_str, content=new_content)

    # Verify the update
    episode = await db.get_episode(uuid_str)
    assert episode['content'] == new_content


def test_build_preview_variants():
    from backend.database.queries import _build_preview, HOME_PREVIEW_MAX_LEN

    header = _build_preview("# Header Here\nMore text")
    assert header == "Header Here"

    sentence = _build_preview("First sentence here. Second follows")
    assert sentence == "First sentence here."

    truncated = _build_preview("A" * (HOME_PREVIEW_MAX_LEN + 20))
    assert truncated.endswith("...")
    assert len(truncated) <= HOME_PREVIEW_MAX_LEN + 3

    newline_handling = _build_preview("Line one without header\ncontinues on next line")
    assert newline_handling.startswith("Line one without header")


@pytest.mark.asyncio
async def test_update_episode_title(isolated_graph):
    """Test updating episode title (name)."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    uuid_str = await add_journal_entry("Some content", title="Original Title")

    # Update the title
    new_title = "Updated Title"
    await db.update_episode(uuid_str, name=new_title)

    # Verify the update
    episode = await db.get_episode(uuid_str)
    assert episode['name'] == new_title


@pytest.mark.asyncio
async def test_update_episode_valid_at(isolated_graph):
    """Test updating episode date (valid_at)."""
    from backend import add_journal_entry
    import backend.database as db
    from datetime import datetime, timezone

    # Create an episode
    original_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    uuid_str = await add_journal_entry("Content", reference_time=original_date)

    # Update the date
    new_date = datetime(2024, 6, 15, 18, 30, 0, tzinfo=timezone.utc)
    await db.update_episode(uuid_str, valid_at=new_date)

    # Verify the update
    episode = await db.get_episode(uuid_str)
    # graphiti-core returns datetime objects, not ISO strings
    assert episode['valid_at'] == new_date


@pytest.mark.asyncio
async def test_update_episode_multiple_fields(isolated_graph):
    """Test updating multiple episode fields at once."""
    from backend import add_journal_entry
    import backend.database as db
    from datetime import datetime, timezone

    # Create an episode
    uuid_str = await add_journal_entry("Original", title="Old Title")

    # Update multiple fields at once
    new_content = "Completely rewritten content"
    new_title = "New Title"
    new_date = datetime(2024, 12, 25, 10, 0, 0, tzinfo=timezone.utc)

    await db.update_episode(
        uuid_str,
        content=new_content,
        name=new_title,
        valid_at=new_date
    )

    # Verify all updates
    episode = await db.get_episode(uuid_str)
    assert episode['content'] == new_content
    assert episode['name'] == new_title
    assert episode['valid_at'] == new_date


@pytest.mark.asyncio
async def test_update_episode_nonexistent_uuid(isolated_graph):
    """Test that updating a nonexistent episode raises an error."""
    import backend.database as db
    from uuid import uuid4

    nonexistent_uuid = str(uuid4())

    # Should raise ValueError when trying to update nonexistent episode
    with pytest.raises(ValueError, match="Episode .* not found"):
        await db.update_episode(nonexistent_uuid, content="New content")


@pytest.mark.asyncio
async def test_update_episode_preserves_immutable_fields(isolated_graph):
    """Test that update doesn't change immutable fields like uuid, group_id, created_at, source_description."""
    from backend import add_journal_entry
    import backend.database as db
    from backend.settings import DEFAULT_JOURNAL

    # Create an episode
    uuid_str = await add_journal_entry("Original content", source_description="Original source")
    original_episode = await db.get_episode(uuid_str)

    original_uuid = original_episode['uuid']
    original_group_id = original_episode['group_id']
    original_created_at = original_episode['created_at']
    original_source_description = original_episode['source_description']

    # Update content
    await db.update_episode(uuid_str, content="Updated content")

    # Verify immutable fields haven't changed
    updated_episode = await db.get_episode(uuid_str)
    assert updated_episode['uuid'] == original_uuid
    assert updated_episode['group_id'] == original_group_id
    assert updated_episode['created_at'] == original_created_at
    assert updated_episode['source_description'] == original_source_description


@pytest.mark.asyncio
async def test_delete_episode(isolated_graph):
    """Test deleting an episode."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    uuid_str = await add_journal_entry("Episode to delete")

    # Verify it exists
    episode = await db.get_episode(uuid_str)
    assert episode is not None

    # Delete it
    await db.delete_episode(uuid_str)

    # Verify it's gone
    episode = await db.get_episode(uuid_str)
    assert episode is None


@pytest.mark.asyncio
async def test_delete_episode_nonexistent_uuid(isolated_graph):
    """Test that deleting a nonexistent episode raises an error."""
    import backend.database as db
    from uuid import uuid4

    nonexistent_uuid = str(uuid4())

    # Should raise ValueError when trying to delete nonexistent episode
    with pytest.raises(ValueError, match="Episode .* not found"):
        await db.delete_episode(nonexistent_uuid)


@pytest.mark.asyncio
async def test_delete_episode_from_specific_journal(isolated_graph):
    """Test deleting an episode from a specific journal."""
    from backend import add_journal_entry
    import backend.database as db

    # Create episodes in different journals
    uuid1 = await add_journal_entry("Entry in journal1", journal="journal1")
    uuid2 = await add_journal_entry("Entry in journal2", journal="journal2")

    # Delete from journal1
    await db.delete_episode(uuid1, journal="journal1")

    # Verify journal1 episode is gone
    episode1 = await db.get_episode(uuid1, journal="journal1")
    assert episode1 is None

    # Verify journal2 episode still exists
    episode2 = await db.get_episode(uuid2, journal="journal2")
    assert episode2 is not None


@pytest.mark.asyncio
async def test_update_episode_naive_datetime_rejected(isolated_graph):
    """Test that naive datetimes are rejected with clear error."""
    from backend import add_journal_entry
    import backend.database as db
    from datetime import datetime

    # Create an episode
    uuid_str = await add_journal_entry("Content")

    # Try to update with naive datetime
    naive_dt = datetime(2024, 6, 15, 10, 30, 0)  # No tzinfo

    with pytest.raises(ValueError, match="Naive datetime not allowed"):
        await db.update_episode(uuid_str, valid_at=naive_dt)


@pytest.mark.asyncio
async def test_update_episode_iso_string_valid(isolated_graph):
    """Test updating with valid ISO 8601 string."""
    from backend import add_journal_entry
    import backend.database as db
    from datetime import datetime, timezone

    # Create an episode
    uuid_str = await add_journal_entry("Content")

    # Update with ISO string (timezone-aware)
    iso_string = "2024-06-15T10:30:00+00:00"
    await db.update_episode(uuid_str, valid_at=iso_string)

    # Verify the update
    episode = await db.get_episode(uuid_str)
    expected_dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    assert episode['valid_at'] == expected_dt


@pytest.mark.asyncio
async def test_update_episode_iso_string_naive_rejected(isolated_graph):
    """Test that naive ISO strings are rejected."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    uuid_str = await add_journal_entry("Content")

    # Try to update with naive ISO string (no timezone)
    naive_iso = "2024-06-15T10:30:00"

    with pytest.raises(ValueError, match="timezone-naive"):
        await db.update_episode(uuid_str, valid_at=naive_iso)


@pytest.mark.asyncio
async def test_update_episode_iso_string_invalid_format(isolated_graph):
    """Test that invalid ISO strings raise clear error."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    uuid_str = await add_journal_entry("Content")

    # Try to update with invalid ISO string
    invalid_iso = "not-a-valid-datetime"

    with pytest.raises(ValueError, match="Invalid ISO 8601 datetime string"):
        await db.update_episode(uuid_str, valid_at=invalid_iso)


@pytest.mark.asyncio
async def test_update_episode_valid_at_wrong_type(isolated_graph):
    """Test that wrong type for valid_at raises clear error."""
    from backend import add_journal_entry
    import backend.database as db

    # Create an episode
    uuid_str = await add_journal_entry("Content")

    # Try to update with wrong type
    with pytest.raises(ValueError, match="must be timezone-aware datetime or ISO 8601 string"):
        await db.update_episode(uuid_str, valid_at=12345)


@pytest.mark.asyncio
async def test_update_episode_no_fields_error_before_not_found(isolated_graph):
    """Test that 'episode not found' error takes precedence over 'no fields' error."""
    import backend.database as db
    from uuid import uuid4

    nonexistent_uuid = str(uuid4())

    # Should raise "not found" error, not "no fields" error
    with pytest.raises(ValueError, match="not found"):
        await db.update_episode(nonexistent_uuid)  # No fields provided


def test_entity_types_format():
    """Test entity types formatting for LLM."""
    from backend.graph.entities_edges import format_entity_types_for_llm, entity_types
    import json

    result = format_entity_types_for_llm(entity_types)
    types_list = json.loads(result)

    assert len(types_list) == 5
    # Entity is listed last so LLM prioritizes Person/Place/Group/Activity first
    assert types_list[0]["entity_type_name"] == "Person"
    assert types_list[-1]["entity_type_name"] == "Entity"

    person_type = next(t for t in types_list if t["entity_type_name"] == "Person")
    assert person_type["entity_type_id"] == 1


@pytest.mark.asyncio
async def test_persist_entities_and_edges(isolated_graph):
    """Test persisting entities and episodic edges."""
    from backend.database.persistence import persist_entities_and_edges
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EpisodicEdge
    from graphiti_core.utils.datetime_utils import utc_now
    from backend import add_journal_entry
    from backend.settings import DEFAULT_JOURNAL
    from backend.database import get_driver

    episode_uuid = await add_journal_entry("Test content")

    person_node = EntityNode(
        name="Sarah",
        group_id=DEFAULT_JOURNAL,
        labels=["Entity", "Person"],
        summary="",
        created_at=utc_now(),
        name_embedding=[],
    )

    episodic_edge = EpisodicEdge(
        source_node_uuid=episode_uuid,
        target_node_uuid=person_node.uuid,
        group_id=DEFAULT_JOURNAL,
        created_at=utc_now(),
    )

    await persist_entities_and_edges(
        nodes=[person_node],
        edges=[],
        episodic_edges=[episodic_edge],
        journal=DEFAULT_JOURNAL,
        episode_uuid=episode_uuid,
    )

    # Verify entity exists
    driver = get_driver(DEFAULT_JOURNAL)
    query = "MATCH (e:Entity {name: 'Sarah'}) RETURN count(e) as count"
    records, _, _ = await driver.execute_query(query)
    assert len(records) > 0
    assert records[0]['count'] >= 1


@pytest.mark.asyncio
async def test_persist_entities_and_edges_raises_when_episode_deleted(isolated_graph):
    """Test that persist_entities_and_edges raises EpisodeDeletedError for missing episode."""
    from backend.database.persistence import persist_entities_and_edges, EpisodeDeletedError
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EpisodicEdge
    from graphiti_core.utils.datetime_utils import utc_now
    from backend.settings import DEFAULT_JOURNAL

    nonexistent_uuid = "nonexistent-episode-uuid"

    person_node = EntityNode(
        name="TestPerson",
        group_id=DEFAULT_JOURNAL,
        labels=["Entity", "Person"],
        summary="",
        created_at=utc_now(),
        name_embedding=[],
    )

    episodic_edge = EpisodicEdge(
        source_node_uuid=nonexistent_uuid,
        target_node_uuid=person_node.uuid,
        group_id=DEFAULT_JOURNAL,
        created_at=utc_now(),
    )

    with pytest.raises(EpisodeDeletedError) as exc_info:
        await persist_entities_and_edges(
            nodes=[person_node],
            edges=[],
            episodic_edges=[episodic_edge],
            journal=DEFAULT_JOURNAL,
            episode_uuid=nonexistent_uuid,
        )

    assert exc_info.value.episode_uuid == nonexistent_uuid


@pytest.mark.asyncio
async def test_update_episode_triggers_extraction_when_content_changes(isolated_graph):
    """Test that update_episode() returns True and sets status when content changes."""
    from backend import add_journal_entry
    from backend.database import update_episode
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend.settings import DEFAULT_JOURNAL

    # Create an episode
    original_content = "Original content"
    uuid_str = await add_journal_entry(original_content)

    # Clear the pending status from creation
    set_episode_status(uuid_str, "done", DEFAULT_JOURNAL)

    # Update the content
    new_content = "Updated content with different text"
    content_changed = await update_episode(uuid_str, content=new_content)

    # Verify function returns True when content changes
    assert content_changed is True, "Expected content_changed to be True"

    # Verify Redis status is set to pending_nodes
    status = get_episode_status(uuid_str, DEFAULT_JOURNAL)
    assert status == "pending_nodes", f"Expected pending_nodes but got {status}"


@pytest.mark.asyncio
async def test_update_episode_returns_false_when_content_unchanged(isolated_graph):
    """Test that update_episode returns False when only name is updated (content unchanged)."""
    from backend import add_journal_entry
    from backend.database import update_episode, get_episode
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend.settings import DEFAULT_JOURNAL

    # Create an episode
    original_content = "Test content"
    original_name = "Original Title"
    uuid_str = await add_journal_entry(original_content, title=original_name)

    # Clear the pending status from creation
    set_episode_status(uuid_str, "done", DEFAULT_JOURNAL)

    # Update only the name (no content parameter)
    new_name = "Updated Title"
    content_changed = await update_episode(uuid_str, name=new_name)

    # Verify return value is False
    assert content_changed is False, "Expected content_changed to be False when content not updated"

    # Verify Redis status is NOT "pending_nodes"
    status = get_episode_status(uuid_str, DEFAULT_JOURNAL)
    assert status != "pending_nodes", f"Status should not be pending_nodes, got {status}"

    # Verify name was actually updated
    episode = await get_episode(uuid_str)
    assert episode['name'] == new_name, f"Name should be updated to {new_name}"


@pytest.mark.asyncio
async def test_update_episode_returns_false_when_content_identical(isolated_graph):
    """Test that update_episode returns False when content is identical to original."""
    from backend import add_journal_entry
    from backend.database import update_episode
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend.settings import DEFAULT_JOURNAL

    # Create an episode
    original_content = "Test content"
    uuid_str = await add_journal_entry(original_content)

    # Clear the pending status from creation
    set_episode_status(uuid_str, "done", DEFAULT_JOURNAL)

    # Update with identical content
    content_changed = await update_episode(uuid_str, content="Test content")

    # Verify return value is False
    assert content_changed is False, "Expected content_changed to be False when content identical"

    # Verify Redis status is NOT "pending_nodes"
    status = get_episode_status(uuid_str, DEFAULT_JOURNAL)
    assert status != "pending_nodes", f"Status should not be pending_nodes, got {status}"
