"""Core utilities for FalkorDB operations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

JOURNAL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
MAX_JOURNAL_NAME_LENGTH = 64

STOPWORDS = [
    'a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but',
    'by', 'for', 'if', 'in', 'into', 'it', 'no', 'not', 'of', 'on',
    'or', 'such', 'that', 'their', 'then', 'there', 'these', 'they',
    'this', 'to', 'was', 'will', 'with',
]

SELF_ENTITY_UUID = UUID("11111111-1111-1111-1111-111111111111")
SELF_ENTITY_NAME = "Self"
SELF_ENTITY_LABELS = ["Entity", "Person"]


def validate_journal_name(journal: str) -> None:
    """Validate journal name for use as FalkorDB graph name.

    Args:
        journal: Journal name to validate

    Raises:
        ValueError: If name is invalid (empty, too long, or contains invalid characters)
    """
    if not journal:
        raise ValueError("Journal name cannot be empty")
    if len(journal) > MAX_JOURNAL_NAME_LENGTH:
        raise ValueError(
            f"Journal name too long: {len(journal)} chars (max {MAX_JOURNAL_NAME_LENGTH})"
        )
    if not JOURNAL_NAME_PATTERN.match(journal):
        raise ValueError(
            f"Invalid journal name '{journal}'. "
            "Use only: letters, numbers, underscores, hyphens"
        )


def to_cypher_literal(value: Any) -> str:
    """Convert Python values into Cypher literals (FalkorDB lacks parameters).

    Properly escapes special characters in strings for Cypher queries:
    - Backslashes must be escaped first (before quotes)
    - Single quotes are then escaped
    - This prevents malformed queries and ensures content integrity
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # CRITICAL: Escape backslashes FIRST, then quotes
        # This prevents content like "C:\Users\" from creating malformed queries
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return json.dumps(value)


SELF_UUID_LITERAL = to_cypher_literal(str(SELF_ENTITY_UUID))


def _decode_value(value: Any) -> Any:
    """Decode FalkorDB result cells to Python primitives.

    FalkorDB returns values as [type_id, value] pairs:
    - type 1: null
    - type 2: string (bytes)
    - type 3: integer
    - type 4: boolean
    - type 5: double
    - type 6: array (list of [type_id, value] pairs)
    - type 7: edge
    - type 8: node
    - type 9: path
    """
    # Handle [type_id, value] pair format
    if isinstance(value, (list, tuple)) and len(value) == 2:
        type_id, data = value
        if isinstance(type_id, int):
            if type_id == 1:  # null
                return None
            elif type_id == 2:  # string
                return data.decode("utf-8") if isinstance(data, bytes) else data
            elif type_id == 3:  # integer
                return int(data)
            elif type_id == 4:  # boolean
                return bool(data)
            elif type_id == 5:  # double
                return float(data)
            elif type_id == 6:  # array
                if isinstance(data, list):
                    return [_decode_value(item) for item in data]
                return []
            # Types 7-9 (edge, node, path) handled by default return

    # Fallback for non-typed values
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    return value


class FalkorGraph(Protocol):
    """Type protocol for FalkorDB graph instances.

    This protocol defines the interface expected from FalkorDB graph objects
    returned by select_graph(). Used for type hints without tight coupling
    to the redislite.falkordb_client implementation.
    """

    def query(self, cypher: str, params: Any = None) -> Any:
        """Execute a Cypher query and return results."""
        ...

    def ro_query(self, cypher: str, params: Any = None) -> Any:
        """Execute a read-only Cypher query."""
        ...

    def delete(self) -> None:
        """Delete this graph from the database."""
        ...


def _to_cypher_array(items: list) -> str:
    """Convert Python list to Cypher array literal for FalkorDB.

    Args:
        items: List of values to convert

    Returns:
        Cypher array literal string (e.g., "['a', 'b', 'c']" or "[]")
    """
    if not items:
        return "[]"
    # Convert each item to a Cypher literal and join with commas
    elements = [to_cypher_literal(item) for item in items]
    return f"[{', '.join(elements)}]"


def _merge_episode_sync(graph: FalkorGraph, episode: dict[str, Any]) -> None:
    """Synchronous episode merge for use in asyncio.to_thread.

    Args:
        graph: FalkorDB graph instance
        episode: Episode dict (graphiti-core converts EpisodicNode to dict before calling)

    Note:
        Uses FalkorDB native array types for entity_edges and labels.
    """
    # graphiti-core converts EpisodicNode to dict and removes 'labels' field (bulk_utils.py line 160-162)
    episode.setdefault('entity_edges', [])
    episode.setdefault('labels', [])

    # Convert arrays to native FalkorDB array syntax
    entity_edges_array = _to_cypher_array(episode['entity_edges'])
    labels_array = _to_cypher_array(episode['labels'])

    # Handle source - could be EpisodeType enum or string
    source_val = episode['source']
    if hasattr(source_val, 'value'):
        source_val = source_val.value
    elif isinstance(source_val, str):
        pass  # Already a string
    else:
        source_val = str(source_val)

    # Handle datetime fields - could be datetime objects or ISO strings
    valid_at = episode['valid_at']
    if hasattr(valid_at, 'isoformat'):
        valid_at = valid_at.isoformat()

    created_at = episode['created_at']
    if hasattr(created_at, 'isoformat'):
        created_at = created_at.isoformat()

    query = f"""
    MERGE (e:Episodic {{uuid: {to_cypher_literal(episode['uuid'])}}})
    SET e.name = {to_cypher_literal(episode['name'])},
        e.group_id = {to_cypher_literal(episode['group_id'])},
        e.content = {to_cypher_literal(episode['content'])},
        e.source = {to_cypher_literal(source_val)},
        e.source_description = {to_cypher_literal(episode['source_description'])},
        e.valid_at = {to_cypher_literal(valid_at)},
        e.created_at = {to_cypher_literal(created_at)},
        e.entity_edges = {entity_edges_array},
        e.labels = {labels_array}
    RETURN e.uuid
    """

    try:
        graph.query(query)
    except Exception as exc:
        logger.exception("Failed to merge episode")
        raise RuntimeError(f"Failed to persist episode {episode['uuid']}") from exc


__all__ = [
    "SELF_ENTITY_UUID",
    "SELF_ENTITY_NAME",
    "SELF_ENTITY_LABELS",
    "SELF_UUID_LITERAL",
    "STOPWORDS",
    "FalkorGraph",
    "validate_journal_name",
    "to_cypher_literal",
    "_decode_value",
    "_merge_episode_sync",
]
