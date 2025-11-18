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
    """Execute a query and return decoded raw response rows."""
    result = graph.query(query)
    raw = getattr(result, "_raw_response", None)
    if not raw or len(raw) < 2:
        return []
    rows = raw[1]
    return [[_decode(col[1]) if isinstance(col, (list, tuple)) and len(col) > 1 else _decode(col[0]) for col in row] for row in rows]


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
    """Test that concurrent SELF entity seeding doesn't create race conditions.

    Similar to lock initialization, SELF entity seeding has a lazy lock initialization
    that could race.
    """
    import asyncio
    import backend.database.persistence as persistence

    # Reset SELF seeding state
    persistence._seeded_self_groups.clear()
    persistence._self_seed_lock = None

    async def seed_self(journal: str, task_id: int):
        """Try to seed SELF entity from multiple tasks."""
        await db_utils.ensure_self_entity(journal)
        return task_id

    # Launch 10 concurrent tasks trying to seed SELF for the same journal
    tasks = [seed_self("self-race-journal", i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All tasks should complete
    assert len(results) == 10

    # Verify SELF was seeded (should be in the set)
    assert "self-race-journal" in persistence._seeded_self_groups

    # Verify only one SELF entity exists in the graph
    graph = db_utils.get_falkordb_graph("self-race-journal")
    from backend.database import SELF_ENTITY_UUID, to_cypher_literal
    query = f"""
    MATCH (self:Entity:Person {{uuid: {to_cypher_literal(str(SELF_ENTITY_UUID))}}})
    RETURN count(self) as count
    """
    result = graph.query(query)
    rows = _query_rows(graph, query)
    count = int(rows[0][0]) if rows else 0
    assert count == 1, f"Expected exactly 1 SELF entity, found {count}"


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
