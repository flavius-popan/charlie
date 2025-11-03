"""FalkorDB initialization and utility functions.

CRITICAL: FalkorDB Query Result Format
=====================================

FalkorDB's QueryResult object has a non-standard structure that differs from Neo4j/Cypher:

1. **result_set** contains column METADATA, not data:
   - Format: [[column_index, column_name_bytes], ...]
   - Example: [[1, b'n.uuid'], [1, b'n.name']]

2. **result.statistics** contains the actual DATA:
   - Format: [[[type_code, value], [type_code, value], ...], ...]
   - Each outer list element is a row
   - Each inner list element is a column value pair [type_code, actual_value]
   - Example: [[[2, b'uuid-123'], [2, b'Alice']], [[2, b'uuid-456'], [2, b'Bob']]]

3. **Special case for count() queries:**
   - result_set[0][0] happens to work because it extracts the count value from metadata
   - Example: result_set = [[1, b'count(n)']] → result_set[0][0] = 1

4. **Multi-column queries MUST use result.statistics:**
   - Extract values from [type_code, value] pairs: row[col_index][1]
   - Example: uuid = row[0][1], name = row[1][1]

5. **FalkorDBLite (embedded) does NOT support:**
   - Parameterized queries ($param syntax)
   - Multi-query transactions
   - Must use literal values in Cypher queries

See graphviz_utils.py:load_written_entities() for working example.
"""
import atexit
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from redislite.falkordb_client import FalkorDB

from settings import DB_PATH, GRAPH_NAME

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
    node_result = graph.query("MATCH (n) RETURN count(n) as node_count")
    node_count = 0
    if node_result.statistics and node_result.statistics[0]:
        node_column = node_result.statistics[0][0]
        node_value = node_column[1] if len(node_column) > 1 else node_column[0]
        decoded_value = _decode_value(node_value)
        node_count = int(decoded_value) if decoded_value is not None else 0

    # Count edges
    edge_result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
    edge_count = 0
    if edge_result.statistics and edge_result.statistics[0]:
        edge_column = edge_result.statistics[0][0]
        edge_value = edge_column[1] if len(edge_column) > 1 else edge_column[0]
        decoded_value = _decode_value(edge_value)
        edge_count = int(decoded_value) if decoded_value is not None else 0

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
    episode,
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    episodic_edges
) -> dict[str, Any]:
    """
    Write episode, entities, and relationships to FalkorDB using Graphiti utilities.

    Args:
        episode: EpisodicNode object
        entity_nodes: List of EntityNode objects
        entity_edges: List of EntityEdge objects
        episodic_edges: List of EpisodicEdge objects

    Returns:
        dict with counts and UUIDs of created nodes/edges

    Error handling: Exceptions propagate to caller (fail-fast).
    """
    # Validate input
    if not entity_nodes:
            logging.warning("write_entities_and_edges called with empty entity_nodes list")
            return {"nodes_created": 0, "edges_created": 0, "node_uuids": [], "edge_uuids": []}

    # 1. Write EpisodicNode (mirrors bulk_utils.py:217)
    episode_dict = {
        "uuid": episode.uuid,
        "name": episode.name,
        "group_id": episode.group_id,
        "source": episode.source.value,  # Convert enum to string
        "source_description": episode.source_description,
        "content": episode.content,
        "valid_at": episode.valid_at,
        "created_at": episode.created_at,
        "entity_edges": episode.entity_edges,
        "labels": episode.labels or [],
    }
    episode_dict = convert_datetimes_to_strings(episode_dict)

    # Build episode node properties for Cypher
    episode_props = {
        'uuid': episode_dict['uuid'],
        'name': episode_dict['name'],
        'group_id': episode_dict['group_id'],
        'source': episode_dict['source'],
        'source_description': episode_dict['source_description'],
        'content': episode_dict['content'],
        'valid_at': episode_dict['valid_at'],
        'created_at': episode_dict['created_at'],
        'entity_edges': episode_dict['entity_edges'],
        'labels': json.dumps(episode_dict['labels'])  # Serialize list to JSON string
    }

    # Build SET clauses for episode properties
    episode_set_clause = ', '.join([
        f"e.{key} = {to_cypher_literal(value)}"
        for key, value in episode_props.items()
    ])

    episode_query = f"""
    MERGE (e:Episodic {{uuid: {to_cypher_literal(episode_dict['uuid'])}}})
    SET {episode_set_clause}
    RETURN e.uuid AS uuid
    """

    result = graph.query(episode_query)
    if result.result_set:
        logging.info(f"Created episode node: {episode.uuid}")

    # 2. Convert EntityNode objects to Cypher-compatible dicts
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

    # 3. Write nodes (individual queries - FalkorDBLite doesn't support parameters)
    nodes_created = 0
    if node_dicts:
        for node in node_dicts:
            # FalkorDB only supports primitives and arrays of primitives
            # Serialize dicts to JSON strings, keep lists of strings as-is
            props = {
                'uuid': node['uuid'],
                'name': node['name'],
                'group_id': node['group_id'],
                'created_at': node['created_at'],
                'labels': json.dumps(node['labels']),  # Serialize list to JSON string
                'summary': node['summary'],
                'attributes': json.dumps(node['attributes'])  # Serialize dict to JSON string
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

    # 4. Convert EntityEdge objects to Cypher-compatible dicts
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

    # 5. Write edges (individual queries - FalkorDBLite doesn't support parameters)
    edges_created = 0
    if edge_dicts:
        for edge in edge_dicts:
            # FalkorDB only supports primitives and arrays of primitives
            # Serialize dicts to JSON strings, keep lists of strings as-is
            props = {
                'uuid': edge['uuid'],
                'name': edge['name'],
                'fact': edge['fact'],
                'group_id': edge['group_id'],
                'created_at': edge['created_at'],
                'episodes': edge['episodes'],  # Keep as native list
                'expired_at': edge['expired_at'],
                'valid_at': edge['valid_at'],
                'invalid_at': edge['invalid_at'],
                'attributes': json.dumps(edge['attributes'])  # Serialize dict to JSON string
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

    # 6. Write EpisodicEdges (MENTIONS) (mirrors bulk_utils.py:221)
    episodic_edge_dicts = []
    for edge in episodic_edges:
        edge_dict = {
            "uuid": edge.uuid,
            "source_uuid": edge.source_node_uuid,  # Episode UUID
            "target_uuid": edge.target_node_uuid,  # Entity UUID
            "group_id": edge.group_id,
            "created_at": edge.created_at,
        }
        episodic_edge_dicts.append(convert_datetimes_to_strings(edge_dict))

    episodic_edges_created = 0
    if episodic_edge_dicts:
        for edge in episodic_edge_dicts:
            # Build properties for MENTIONS edge
            props = {
                'uuid': edge['uuid'],
                'group_id': edge['group_id'],
                'created_at': edge['created_at']
            }

            # Build SET clauses for properties
            set_clause = ', '.join([
                f"r.{key} = {to_cypher_literal(value)}"
                for key, value in props.items()
            ])

            mentions_query = f"""
            MATCH (episode:Episodic {{uuid: {to_cypher_literal(edge['source_uuid'])}}})
            MATCH (entity:Entity {{uuid: {to_cypher_literal(edge['target_uuid'])}}})
            MERGE (episode)-[r:MENTIONS {{uuid: {to_cypher_literal(edge['uuid'])}}}]->(entity)
            SET {set_clause}
            RETURN r.uuid AS uuid
            """

            result = graph.query(mentions_query)
            if result.result_set:
                episodic_edges_created += 1

        logging.info(f"Created {episodic_edges_created} MENTIONS edges")

    return {
        "episode_uuid": episode.uuid,
        "nodes_created": len(node_dicts),
        "edges_created": len(edge_dicts),
        "episodic_edges_created": len(episodic_edge_dicts),
        "node_uuids": [n["uuid"] for n in node_dicts],
        "edge_uuids": [e["uuid"] for e in edge_dicts],
        "episodic_edge_uuids": [e["uuid"] for e in episodic_edge_dicts]
    }


def _decode_value(value: Any) -> Any:
    """Convert FalkorDB return values (bytes) to Python primitives."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:  # noqa: BLE001
            return value
    return value


def _decode_sequence(value: Any) -> list[Any]:
    """Normalize FalkorDB array values into plain Python lists."""
    if not isinstance(value, list):
        return []

    normalized: list[Any] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            raw = item[1] if len(item) > 1 else item[0]
        else:
            raw = item
        normalized.append(_decode_value(raw))
    return normalized


def _normalize_string_list(values: list[Any]) -> list[str]:
    """Coerce a list of decoded values into strings."""
    normalized: list[str] = []
    for value in values or []:
        normalized.append(str(value))
    return normalized


def _decode_json(value: Any, default):
    """Parse JSON strings stored in FalkorDB properties."""
    decoded = _decode_value(value)
    if decoded in ("", None):
        return default
    if isinstance(decoded, (list, dict)):
        if isinstance(decoded, list):
            seq = _decode_sequence(decoded)
            return seq if seq else default
        return decoded
    try:
        return json.loads(decoded)
    except Exception:  # noqa: BLE001
        return default


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO formatted datetime strings."""
    decoded = _decode_value(value)
    if not decoded:
        return None
    if isinstance(decoded, datetime):
        return decoded
    try:
        return datetime.fromisoformat(decoded)
    except Exception:  # noqa: BLE001
        return None


def _iter_statistics_rows(result) -> Iterable[list[Any]]:
    """Yield decoded rows from a FalkorDB query result."""
    for row in result.statistics or []:
        yield [_decode_value(col[1]) if len(col) > 1 else None for col in row]


def fetch_recent_episodes(
    group_id: str,
    reference_time: datetime,
    limit: int,
) -> list[EpisodicNode]:
    """Fetch the most recent episodes at or before a reference time."""
    reference_literal = to_cypher_literal(reference_time.isoformat())
    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (e:Episodic)
    WHERE e.group_id = {group_literal}
      AND e.valid_at <= {reference_literal}
    RETURN e.uuid, e.name, e.content, e.source_description,
           e.valid_at, e.created_at, e.entity_edges, e.labels,
           e.source
    ORDER BY e.valid_at DESC
    LIMIT {limit}
    """
    result = graph.query(query)
    episodes: list[EpisodicNode] = []
    for (
        uuid,
        name,
        content,
        source_description,
        valid_at,
        created_at,
        entity_edges,
        labels,
        source_value,
    ) in _iter_statistics_rows(result):
        source_literal = _decode_value(source_value) or EpisodeType.text.value
        try:
            source = EpisodeType(source_literal)
        except Exception:  # noqa: BLE001
            source = EpisodeType.text
        episode = EpisodicNode(
            uuid=str(uuid),
            name=str(name),
            content=str(content or ""),
            source_description=str(source_description or ""),
            valid_at=_parse_datetime(valid_at) or reference_time,
            created_at=_parse_datetime(created_at) or reference_time,
            entity_edges=_normalize_string_list(_decode_json(entity_edges, [])),
            labels=_normalize_string_list(_decode_json(labels, [])),
            group_id=group_id,
            source=source,
        )
        episodes.append(episode)
    return episodes


def fetch_entities_by_group(group_id: str) -> dict[str, EntityNode]:
    """Fetch all Entity nodes for a group keyed by UUID."""
    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (n:Entity)
    WHERE n.group_id = {group_literal}
    RETURN n.uuid, n.name, n.summary, n.labels, n.attributes,
           n.created_at
    """
    result = graph.query(query)
    entities: dict[str, EntityNode] = {}
    for uuid, name, summary, labels, attributes, created_at in _iter_statistics_rows(result):
        node = EntityNode(
            uuid=str(uuid),
            name=str(name),
            summary=str(summary or ""),
            labels=_normalize_string_list(_decode_json(labels, ["Entity"])),
            attributes=_decode_json(attributes, {}),
            created_at=_parse_datetime(created_at),
            group_id=group_id,
            name_embedding=[],  # Embeddings handled elsewhere (stubbed for parity phase)
        )
        entities[node.uuid] = node
    return entities


def fetch_entity_edges_by_group(group_id: str) -> dict[str, EntityEdge]:
    """Fetch all RELATES_TO edges for a group keyed by UUID."""
    group_literal = to_cypher_literal(group_id)
    query = f"""
    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
    WHERE r.group_id = {group_literal}
    RETURN r.uuid, source.uuid, target.uuid, r.name, r.fact,
           r.episodes, r.valid_at, r.invalid_at, r.expired_at,
           r.attributes, r.created_at
    """
    result = graph.query(query)
    edges: dict[str, EntityEdge] = {}
    for (
        uuid,
        source_uuid,
        target_uuid,
        name,
        fact,
        episodes,
        valid_at,
        invalid_at,
        expired_at,
        attributes,
        created_at,
    ) in _iter_statistics_rows(result):
        edge = EntityEdge(
            uuid=str(uuid),
            source_node_uuid=str(source_uuid),
            target_node_uuid=str(target_uuid),
            name=str(name),
            fact=str(fact or ""),
            episodes=_normalize_string_list(_decode_json(episodes, [])),
            valid_at=_parse_datetime(valid_at),
            invalid_at=_parse_datetime(invalid_at),
            expired_at=_parse_datetime(expired_at),
            attributes=_decode_json(attributes, {}),
            created_at=_parse_datetime(created_at),
            group_id=group_id,
            fact_embedding=[],  # Embeddings handled elsewhere (stubbed for parity phase)
        )
        edges[edge.uuid] = edge
    return edges
