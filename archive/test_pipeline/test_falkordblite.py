"""Regression-style FalkorDBLite tests that share the global redis-server.

These tests were ported from the manual verification script in
``falkordblite-build/test_falkordblite.py`` so they can run inside the main
pytest suite and reuse the session-wide FalkorDB Lite process configured in
``tests/test_pipeline/conftest.py``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack

import pytest

import pipeline.falkordblite_driver as db_utils
from pipeline.self_reference import (
    SELF_ENTITY_LABELS,
    SELF_ENTITY_NAME,
    SELF_ENTITY_UUID,
)
from settings import GROUP_ID


def _decode(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def _query_rows(graph, query: str) -> list[list[object]]:
    """Execute a query and return decoded statistics rows."""
    result = graph.query(query)
    return list(db_utils._iter_result_rows(result))


def _rows_from_result(result) -> list[list[object]]:
    return list(db_utils._iter_result_rows(result))


@pytest.fixture
def falkordb_client(falkordb_test_context):
    graph = db_utils._ensure_graph()
    if graph is None or db_utils._db is None:  # pragma: no cover - defensive guard
        pytest.skip("FalkorDB Lite is unavailable in this environment.")
    return db_utils._db


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
    rows = [(row[0], int(row[1])) for row in _rows_from_result(read_only)]
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

    departments: list[tuple[str, int, float, float]] = []
    for name, count, avg_salary, max_salary in _rows_from_result(result):
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
    stored_rows = _rows_from_result(creation)
    assert stored_rows in ([[]], [[[]]]), f"Unexpected storage payload {stored_rows}"

    verification = isolated_graph.query(
        """
        MATCH (n:Entity {uuid: 'vecf32-empty'})
        RETURN n.embedding
        """
    )
    verified_rows = _rows_from_result(verification)
    assert verified_rows in ([[]], [[[]]]), f"Unexpected verification payload {verified_rows}"


@pytest.mark.asyncio
async def test_ensure_graph_ready_idempotent(isolated_graph):
    """ensure_graph_ready can rebuild indices multiple times without corrupting data."""
    await db_utils.ensure_graph_ready(delete_existing=True)
    await db_utils.ensure_graph_ready()

    isolated_graph.query("CREATE (:InitSmoke {uuid: 'init-smoke'})")
    rows = _query_rows(isolated_graph, "MATCH (n:InitSmoke) RETURN count(n)")
    assert rows == [[1]]


@pytest.mark.asyncio
async def test_reset_database_reseeds_self_node(isolated_graph):
    """reset_database now wipes data but reseeds the deterministic author entity "I"."""
    isolated_graph.query(
        """
        CREATE (:Episodic {uuid: 'reset-episode', group_id: 'reset-group'}),
               (:Entity {uuid: 'reset-entity', group_id: 'reset-group'})
        """
    )

    stats_before = await db_utils.get_db_stats()
    assert stats_before["episodes"] >= 1
    assert stats_before["entities"] >= 1

    await db_utils.reset_database()
    stats_after = await db_utils.get_db_stats()
    assert stats_after["episodes"] == 0
    assert stats_after["entities"] == 1

    rows = _query_rows(
        isolated_graph,
        """
        MATCH (self:Entity {uuid: $self_uuid})
        RETURN self.name, self.group_id, self.labels
        """.replace(
            "$self_uuid", db_utils.to_cypher_literal(str(SELF_ENTITY_UUID))
        ),
    )
    assert rows, 'author entity "I" should exist after reset'
    self_name, self_group, raw_labels = rows[0]
    assert self_name == SELF_ENTITY_NAME
    assert self_group == GROUP_ID
    labels = json.loads(raw_labels)
    assert all(label in labels for label in SELF_ENTITY_LABELS)


@pytest.mark.asyncio
async def test_gradio_persistence_workflow(isolated_graph):
    """Simulate the Gradio UI workflow: write -> persist -> reset -> verify.

    This test validates the user's expected behavior:
    1. Save data (persist episode and nodes)
    2. Verify data persists (simulating page refresh)
    3. Clear database
    4. Verify database is empty (simulating another page refresh)
    """
    from graphiti_core.nodes import EpisodicNode, EntityNode, EpisodeType
    from graphiti_core.utils.datetime_utils import utc_now
    from datetime import datetime

    # Step 1: Create and persist data (simulating Gradio "Write to Database")
    reference_time = utc_now()
    episode = EpisodicNode(
        name=f"test_episode_{reference_time.isoformat()}",
        group_id="test-group",
        labels=[],
        source=EpisodeType.text,
        content="Test journal entry for persistence workflow",
        source_description="Test entry",
        created_at=reference_time,
        valid_at=reference_time,
    )

    nodes = [
        EntityNode(
            name="Test Entity 1",
            labels=["Entity", "Person"],
            group_id="test-group",
            created_at=reference_time,
        ),
        EntityNode(
            name="Test Entity 2",
            labels=["Entity", "Location"],
            group_id="test-group",
            created_at=reference_time,
        ),
    ]

    result = await db_utils.persist_episode_and_nodes(episode, nodes)
    assert result["status"] == "persisted"
    assert result["nodes_written"] == 2

    # Step 2: Verify data persists (simulating page refresh - stats should show data)
    stats_after_write = await db_utils.get_db_stats()
    assert stats_after_write["episodes"] >= 1, "Episode should persist after write"
    assert stats_after_write["entities"] >= 2, "Entities should persist after write"

    # Step 3: Reset database (simulating Gradio "Reset Database" button)
    reset_result = await db_utils.reset_database()
    assert "cleared successfully" in reset_result.lower()

    # Step 4: Verify database is empty (simulating another page refresh)
    stats_after_reset = await db_utils.get_db_stats()
    assert stats_after_reset["episodes"] == 0, "Episodes should be 0 after reset"
    assert stats_after_reset["entities"] == 1, 'Only author entity "I" should remain after reset'

    # Step 5: Verify no nodes remain in the database
    rows = _query_rows(
        isolated_graph,
        "MATCH (n:Entity) RETURN count(n)",
    )
    assert rows == [[1]], 'Only author entity "I" should exist after reset'
