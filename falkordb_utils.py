"""FalkorDB initialization and utility functions."""
import atexit
import json
import logging
from pathlib import Path
from redislite.falkordb_client import FalkorDB
from settings import DB_PATH, GRAPH_NAME
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from typing import Any

# Create data directory
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize embedded database (spawns Redis process)
db = FalkorDB(dbfilename=str(DB_PATH))

# Select the graph to use (creates if doesn't exist)
graph = db.select_graph(GRAPH_NAME)

# Track database cleanup state
_db_closed = False


def cleanup_db():
    """Clean shutdown of FalkorDB - CRITICAL before exit."""
    global _db_closed
    if _db_closed:
        return
    try:
        db.close()
        _db_closed = True
        print("✓ FalkorDB closed successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to close FalkorDB: {e}")


# Register cleanup handler for proper shutdown
atexit.register(cleanup_db)


def to_cypher_literal(value: Any) -> str:
    """
    Convert a Python value to a Cypher literal string.

    Note: FalkorDBLite (embedded version) does not support parameterized queries,
    so we must build queries with literal values.

    Args:
        value: Python value to convert (str, int, float, bool, None, list, dict)

    Returns:
        Cypher-compatible literal string
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape single quotes and wrap in single quotes
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, (list, dict)):
        # Arrays and objects use JSON format
        return json.dumps(value)
    else:
        # Fallback to JSON encoding
        return json.dumps(value)


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


def write_entities_and_edges(
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge]
) -> dict[str, Any]:
    """
    Write entities and relationships to FalkorDB using Graphiti utilities.

    Args:
        entity_nodes: List of EntityNode objects
        entity_edges: List of EntityEdge objects

    Returns:
        dict with counts and UUIDs of created nodes/edges

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # Validate input
    if not entity_nodes:
        logging.warning("write_entities_and_edges called with empty entity_nodes list")
        return {"nodes_created": 0, "edges_created": 0, "node_uuids": [], "edge_uuids": []}

    # 1. Convert EntityNode objects to Cypher-compatible dicts
    node_dicts = []
    for node in entity_nodes:
        node_dict = {
            "uuid": node.uuid,
            "name": node.name,
            "group_id": node.group_id,
            "created_at": node.created_at,
            "labels": node.labels or ["Entity"],
            "name_embedding": node.name_embedding or [],
            "summary": node.summary,
            "attributes": node.attributes or {},
        }
        # Use Graphiti's utility to convert datetimes to ISO strings
        node_dicts.append(convert_datetimes_to_strings(node_dict))

    # 2. Write nodes (individual queries - FalkorDBLite doesn't support parameters)
    nodes_created = 0
    if node_dicts:
        for node in node_dicts:
            # Build property assignments with literal values
            props = {
                'uuid': node['uuid'],
                'name': node['name'],
                'group_id': node['group_id'],
                'created_at': node['created_at'],
                'labels': node['labels'],
                'summary': node['summary'],
                'attributes': node['attributes']
            }

            # Build SET clauses for properties
            set_clause = ', '.join([
                f"n.{key} = {to_cypher_literal(value)}"
                for key, value in props.items()
            ])

            # Build embedding SET clause separately (uses vecf32 function)
            embedding_literal = json.dumps(node['name_embedding'])

            node_query = f"""
            MERGE (n:Entity {{uuid: {to_cypher_literal(node['uuid'])}}})
            SET {set_clause}
            SET n.name_embedding = vecf32({embedding_literal})
            RETURN n.uuid AS uuid
            """

            result = graph.query(node_query)
            if result.result_set:
                nodes_created += 1

        logging.info(f"Created {nodes_created} nodes")

    # 3. Convert EntityEdge objects to Cypher-compatible dicts
    edge_dicts = []
    for edge in entity_edges:
        edge_dict = {
            "uuid": edge.uuid,
            "source_uuid": edge.source_node_uuid,
            "target_uuid": edge.target_node_uuid,
            "name": edge.name,
            "fact": edge.fact,
            "group_id": edge.group_id,
            "created_at": edge.created_at,
            "fact_embedding": edge.fact_embedding or [],
            "episodes": edge.episodes or [],
            "expired_at": edge.expired_at,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
            "attributes": edge.attributes or {},
        }
        # Use Graphiti's utility to convert datetimes to ISO strings
        edge_dicts.append(convert_datetimes_to_strings(edge_dict))

    # 4. Write edges (individual queries - FalkorDBLite doesn't support parameters)
    edges_created = 0
    if edge_dicts:
        for edge in edge_dicts:
            # Build property assignments with literal values
            props = {
                'uuid': edge['uuid'],
                'name': edge['name'],
                'fact': edge['fact'],
                'group_id': edge['group_id'],
                'created_at': edge['created_at'],
                'episodes': edge['episodes'],
                'expired_at': edge['expired_at'],
                'valid_at': edge['valid_at'],
                'invalid_at': edge['invalid_at'],
                'attributes': edge['attributes']
            }

            # Build SET clauses for properties
            set_clause = ', '.join([
                f"r.{key} = {to_cypher_literal(value)}"
                for key, value in props.items()
            ])

            # Build embedding SET clause separately (uses vecf32 function)
            embedding_literal = json.dumps(edge['fact_embedding'])

            edge_query = f"""
            MATCH (source:Entity {{uuid: {to_cypher_literal(edge['source_uuid'])}}})
            MATCH (target:Entity {{uuid: {to_cypher_literal(edge['target_uuid'])}}})
            MERGE (source)-[r:RELATES_TO {{uuid: {to_cypher_literal(edge['uuid'])}}}]->(target)
            SET {set_clause}
            SET r.fact_embedding = vecf32({embedding_literal})
            RETURN r.uuid AS uuid
            """

            result = graph.query(edge_query)
            if result.result_set:
                edges_created += 1

        logging.info(f"Created {edges_created} edges")

    return {
        "nodes_created": len(node_dicts),
        "edges_created": len(edge_dicts),
        "node_uuids": [n["uuid"] for n in node_dicts],
        "edge_uuids": [e["uuid"] for e in edge_dicts]
    }
