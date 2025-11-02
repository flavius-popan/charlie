"""FalkorDB initialization and utility functions."""
import atexit
import logging
from pathlib import Path
from redislite.falkordb_client import FalkorDB
from settings import DB_PATH, GRAPH_NAME

# Create data directory
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize embedded database (spawns Redis process)
db = FalkorDB(dbfilename=str(DB_PATH))

# Select the graph to use (creates if doesn't exist)
graph = db.select_graph(GRAPH_NAME)


def cleanup_db():
    """Clean shutdown of FalkorDB - CRITICAL before exit."""
    try:
        db.close()
        print("✓ FalkorDB closed successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to close FalkorDB: {e}")


# Register cleanup handler for proper shutdown
atexit.register(cleanup_db)


def get_db_stats() -> dict[str, int]:
    """
    Query database statistics for UI display.

    Returns:
        dict with 'nodes' and 'edges' counts

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # Count nodes
    result = graph.query("MATCH (n) RETURN count(n) as node_count")
    node_count = result.result_set[0][0] if result.result_set else 0

    # Count edges
    result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
    edge_count = result.result_set[0][0] if result.result_set else 0

    return {"nodes": node_count, "edges": edge_count}


def reset_database() -> str:
    """
    Clear all graph data (DESTRUCTIVE - no confirmation in Phase 1).

    Returns:
        Success message

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    graph.query("MATCH (n) DETACH DELETE n")
    return "Database cleared successfully"
