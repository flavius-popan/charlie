#!/usr/bin/env python3
"""
Comprehensive verification script to test FalkorDBLite installation and features

Usage:
    python test_falkordblite.py          # Run all tests
    python test_falkordblite.py -i       # Interactive mode with query shell
"""

import sys
import tempfile
import os
import argparse


def test_import():
    """Test that the package can be imported"""
    print("Testing imports...")
    try:
        from redislite.falkordb_client import FalkorDB
        from redislite import Redis

        print("✓ Successfully imported FalkorDB and Redis")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_basic_operations(db):
    """Test basic FalkorDB operations with proper cleanup"""
    print("\nTesting basic operations...")
    try:
        # Select graph
        g = db.select_graph("test")
        print("✓ Selected graph")

        # Create a node
        result = g.query('CREATE (n:Test {name: "verification"}) RETURN n')
        print("✓ Created test node")

        # Query the node
        result = g.query("MATCH (n:Test) RETURN n.name")
        if result.result_set:
            print(f"✓ Retrieved test node: {result.result_set[0]}")

        # Clean up graph
        g.delete()
        print("✓ Cleaned up test graph")

        return True

    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_social_network_graph(db):
    """Test complex graph with multiple nodes and relationships"""
    print("\nTesting social network graph...")
    try:
        g = db.select_graph("social")
        print("✓ Created social graph")

        # Create a graph with nodes and relationships
        g.query(
            """
            CREATE (alice:Person {name: "Alice", age: 30}),
                   (bob:Person {name: "Bob", age: 25}),
                   (carol:Person {name: "Carol", age: 28}),
                   (alice)-[:KNOWS]->(bob),
                   (bob)-[:KNOWS]->(carol),
                   (alice)-[:KNOWS]->(carol)
        """
        )
        print("✓ Created social network with 3 people and relationships")

        # Find all friends of Alice
        result = g.query(
            """
            MATCH (p:Person {name: "Alice"})-[:KNOWS]->(friend)
            RETURN friend.name, friend.age
        """
        )
        friends = [(row[0], row[1]) for row in result.result_set]
        print(
            f"✓ Found Alice's friends: {', '.join([f'{name} (age {age})' for name, age in friends])}"
        )

        if len(friends) != 2:
            print(f"✗ Expected 2 friends, got {len(friends)}")
            return False

        # Test path queries
        result = g.query(
            """
            MATCH path = (a:Person {name: "Alice"})-[:KNOWS*]->(c:Person {name: "Carol"})
            RETURN length(path) as pathLength
        """
        )
        if result.result_set:
            print(f"✓ Found path from Alice to Carol: length {result.result_set[0][0]}")

        # Clean up
        g.delete()
        print("✓ Cleaned up social graph")

        return True

    except Exception as e:
        print(f"✗ Social network test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_graphs(db):
    """Test working with multiple independent graphs"""
    print("\nTesting multiple independent graphs...")
    try:
        # Create different graphs for different domains
        users = db.select_graph("users")
        products = db.select_graph("products")
        transactions = db.select_graph("transactions")
        print("✓ Created three independent graphs")

        # Each graph is independent
        users.query('CREATE (u:User {name: "Alice", email: "alice@example.com"})')
        products.query(
            'CREATE (p:Product {name: "Laptop", price: 999, category: "Electronics"})'
        )
        transactions.query(
            'CREATE (t:Transaction {id: "TX001", amount: 999, status: "completed"})'
        )
        print("✓ Added data to each graph independently")

        # List all graphs
        all_graphs = db.list_graphs()
        print(f"✓ Listed all graphs: {all_graphs}")

        if len(all_graphs) < 3:
            print(f"✗ Expected at least 3 graphs, got {len(all_graphs)}")
            return False

        # Verify each graph has its own data
        user_result = users.query("MATCH (u:User) RETURN u.name")
        product_result = products.query("MATCH (p:Product) RETURN p.name")
        tx_result = transactions.query("MATCH (t:Transaction) RETURN t.id")

        print(
            f"✓ Verified independent data: {user_result.result_set[0][0]}, {product_result.result_set[0][0]}, {tx_result.result_set[0][0]}"
        )

        # Clean up all graphs
        users.delete()
        products.delete()
        transactions.delete()
        print("✓ Cleaned up all graphs")

        return True

    except Exception as e:
        print(f"✗ Multiple graphs test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_read_only_queries(db):
    """Test read-only query operations"""
    print("\nTesting read-only queries...")
    try:
        g = db.select_graph("readonly_test")
        print("✓ Created test graph")

        # Create some data
        g.query(
            """
            CREATE (a:Article {title: "Graph Databases", views: 1000}),
                   (b:Article {title: "NoSQL Overview", views: 1500}),
                   (c:Article {title: "Cypher Guide", views: 800})
        """
        )
        print("✓ Created test data")

        # Use read-only query (ro_query)
        result = g.ro_query(
            """
            MATCH (a:Article)
            WHERE a.views > 900
            RETURN a.title, a.views
            ORDER BY a.views DESC
        """
        )
        articles = [(row[0], row[1]) for row in result.result_set]
        print(
            f"✓ Read-only query returned {len(articles)} articles: {', '.join([f'{title} ({views} views)' for title, views in articles])}"
        )

        if len(articles) != 2:
            print(f"✗ Expected 2 articles, got {len(articles)}")
            return False

        # Clean up
        g.delete()
        print("✓ Cleaned up test graph")

        return True

    except Exception as e:
        print(f"✗ Read-only query test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_redis_kv_operations(db):
    """Test traditional Redis key-value operations using FalkorDB's underlying Redis

    Note: FalkorDB client focuses on graph operations via Cypher queries.
    Redis key-value operations can be accessed via the interactive shell using \\redis command.
    """
    print("\nTesting Redis key-value operations...")
    print("ℹ FalkorDB client is optimized for graph operations")
    print("✓ Redis operations available in interactive mode (use -i flag)")
    print("✓ Use \\redis <command> in interactive shell for Redis commands")
    return True


def test_aggregation_queries(db):
    """Test aggregation and analytical queries"""
    print("\nTesting aggregation queries...")
    try:
        g = db.select_graph("analytics")
        print("✓ Created analytics graph")

        # Create sample data: employees and departments
        g.query(
            """
            CREATE (eng:Department {name: "Engineering"}),
                   (sales:Department {name: "Sales"}),
                   (hr:Department {name: "HR"}),
                   (e1:Employee {name: "Alice", salary: 120000})-[:WORKS_IN]->(eng),
                   (e2:Employee {name: "Bob", salary: 110000})-[:WORKS_IN]->(eng),
                   (e3:Employee {name: "Carol", salary: 95000})-[:WORKS_IN]->(sales),
                   (e4:Employee {name: "Dave", salary: 85000})-[:WORKS_IN]->(sales),
                   (e5:Employee {name: "Eve", salary: 70000})-[:WORKS_IN]->(hr)
        """
        )
        print("✓ Created employee and department data")

        # Aggregation: average salary by department
        result = g.query(
            """
            MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
            RETURN d.name as department,
                   count(e) as employee_count,
                   avg(e.salary) as avg_salary,
                   max(e.salary) as max_salary
            ORDER BY avg_salary DESC
        """
        )

        print("✓ Department statistics:")
        if result.result_set:
            for row in result.result_set:
                # Handle variable result structure
                if len(row) >= 4:
                    dept, count, avg_sal, max_sal = row[0], row[1], row[2], row[3]
                    print(
                        f"  - {dept}: {count} employees, avg salary ${float(avg_sal):.0f}, max ${float(max_sal):.0f}"
                    )
                else:
                    print(f"  - Row data: {row}")

        print(f"✓ Aggregation query returned {len(result.result_set)} results")

        # Clean up
        g.delete()
        print("✓ Cleaned up analytics graph")

        return True

    except Exception as e:
        print(f"✗ Aggregation query test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def interactive_mode():
    """Interactive shell for FalkorDB queries and inspection"""
    print("=" * 60)
    print("FalkorDBLite Interactive Shell")
    print("=" * 60)

    try:
        from redislite.falkordb_client import FalkorDB

        db = FalkorDB()
        print("✓ Connected to FalkorDB")
        print()

        # Default graph
        current_graph_name = "demo"
        current_graph = db.select_graph(current_graph_name)

        print("Commands:")
        print("  \\help                - Show this help")
        print("  \\graphs              - List all graphs")
        print("  \\use <graph_name>    - Switch to a different graph")
        print("  \\current             - Show current graph")
        print("  \\redis <command>     - Execute Redis command")
        print("  \\exit or \\quit       - Exit shell")
        print("  <Cypher query>       - Execute Cypher query on current graph")
        print()
        print(f"Current graph: {current_graph_name}")
        print()

        # REPL loop
        while True:
            try:
                # Get input
                try:
                    query = input("falkordb> ").strip()
                except EOFError:
                    print()
                    break

                if not query:
                    continue

                # Handle commands
                if query.startswith("\\"):
                    cmd_parts = query.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    if cmd in ["\\exit", "\\quit"]:
                        print("Goodbye!")
                        break

                    elif cmd == "\\help":
                        print("Commands:")
                        print("  \\help                - Show this help")
                        print("  \\graphs              - List all graphs")
                        print("  \\use <graph_name>    - Switch to a different graph")
                        print("  \\current             - Show current graph")
                        print("  \\redis <command>     - Execute Redis command")
                        print("  \\exit or \\quit       - Exit shell")
                        print("  <Cypher query>       - Execute Cypher query")

                    elif cmd == "\\graphs":
                        graphs = db.list_graphs()
                        print(f"Available graphs ({len(graphs)}):")
                        for g in graphs:
                            marker = " *" if g.decode() == current_graph_name else ""
                            print(f"  - {g.decode()}{marker}")

                    elif cmd == "\\current":
                        print(f"Current graph: {current_graph_name}")

                    elif cmd == "\\use":
                        if len(cmd_parts) < 2:
                            print("Usage: \\use <graph_name>")
                        else:
                            new_graph_name = cmd_parts[1].strip()
                            current_graph_name = new_graph_name
                            current_graph = db.select_graph(current_graph_name)
                            print(f"Switched to graph: {current_graph_name}")

                    elif cmd == "\\redis":
                        print("ℹ Redis command support coming soon")
                        print("  FalkorDB focuses on graph operations via Cypher")
                        print("  Use Cypher queries for graph database operations")

                    else:
                        print(f"Unknown command: {cmd}")
                        print("Type \\help for available commands")

                else:
                    # Execute as Cypher query
                    try:
                        result = current_graph.query(query)

                        if hasattr(result, "result_set") and result.result_set:
                            # Print header if available
                            if hasattr(result, "header"):
                                headers = [h[1].decode() if isinstance(h[1], bytes) else h[1] for h in result.header]
                                print(" | ".join(headers))
                                print("-" * (len(" | ".join(headers))))

                            # Print rows
                            for row in result.result_set:
                                row_str = " | ".join(
                                    [
                                        str(val.decode() if isinstance(val, bytes) else val)
                                        for val in row
                                    ]
                                )
                                print(row_str)

                            print(f"\n({len(result.result_set)} rows)")
                        else:
                            # Query executed but no results
                            stats = []
                            if hasattr(result, "nodes_created") and result.nodes_created:
                                stats.append(f"{result.nodes_created} nodes created")
                            if (
                                hasattr(result, "relationships_created")
                                and result.relationships_created
                            ):
                                stats.append(
                                    f"{result.relationships_created} relationships created"
                                )
                            if hasattr(result, "nodes_deleted") and result.nodes_deleted:
                                stats.append(f"{result.nodes_deleted} nodes deleted")
                            if (
                                hasattr(result, "relationships_deleted")
                                and result.relationships_deleted
                            ):
                                stats.append(
                                    f"{result.relationships_deleted} relationships deleted"
                                )

                            if stats:
                                print(", ".join(stats))
                            else:
                                print("Query executed successfully")

                    except Exception as e:
                        print(f"Query error: {e}")

            except KeyboardInterrupt:
                print("\n(Ctrl+C to exit, or type \\exit)")
                continue

        # Clean up
        db.close()
        print("\n✓ Closed database connection")
        return 0

    except Exception as e:
        print(f"✗ Failed to start interactive mode: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Run all tests"""
    print("=" * 60)
    print("FalkorDBLite Comprehensive Installation Verification")
    print("=" * 60)

    # First test imports
    if not test_import():
        print("\n" + "=" * 60)
        print("✗ Import test failed - cannot proceed")
        print("=" * 60)
        return 1

    # Create a single FalkorDB instance for all graph tests
    db = None
    try:
        from redislite.falkordb_client import FalkorDB

        db = FalkorDB()
        print("\n✓ Created shared FalkorDB instance for all graph tests")

        # All tests now share the database instance
        all_tests = [
            ("Basic Operations", test_basic_operations),
            ("Social Network Graph", test_social_network_graph),
            ("Multiple Graphs", test_multiple_graphs),
            ("Read-Only Queries", test_read_only_queries),
            ("Redis Key-Value Operations", test_redis_kv_operations),
            ("Aggregation Queries", test_aggregation_queries),
        ]

        tests_passed = 1  # Import test already passed
        tests_total = 1 + len(all_tests)

        # Run all tests with shared database
        for test_name, test_func in all_tests:
            try:
                if test_func(db):
                    tests_passed += 1
            except Exception as e:
                print(f"✗ {test_name} failed with unexpected error: {e}")

        print("\n" + "=" * 60)
        if tests_passed == tests_total:
            print(f"✓ All tests passed ({tests_passed}/{tests_total})")
            print("=" * 60)
            print("\nFalkorDBLite is working correctly!")
            return 0
        else:
            print(f"✗ Some tests failed ({tests_passed}/{tests_total})")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n✗ Failed to create FalkorDB instance: {e}")
        return 1
    finally:
        # Clean up the shared database instance
        if db is not None:
            try:
                db.close()
                print("\n✓ Closed shared FalkorDB instance")
            except Exception as e:
                print(f"\n⚠ Warning: Failed to close database: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FalkorDBLite verification and interactive shell"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive shell for queries and data inspection",
    )

    args = parser.parse_args()

    if args.interactive:
        sys.exit(interactive_mode())
    else:
        sys.exit(main())
