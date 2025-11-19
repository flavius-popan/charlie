"""FalkorDB Lite database module.

Public API for episode persistence and database management.
See README.md for module organization.
"""

from backend.database.driver import FalkorLiteDriver, get_driver
from backend.database.lifecycle import (
    disable_tcp_server,
    enable_tcp_server,
    get_falkordb_graph,
    get_tcp_server_endpoint,
    get_tcp_server_password,
    reset_lifecycle_state,
)
from backend.database.persistence import (
    delete_episode,
    ensure_database_ready,
    ensure_graph_ready,
    ensure_self_entity,
    persist_episode,
    reset_persistence_state,
    update_episode,
)
from backend.database.queries import get_episode
from backend.database.utils import (
    SELF_ENTITY_LABELS,
    SELF_ENTITY_NAME,
    SELF_ENTITY_UUID,
    to_cypher_literal,
    validate_journal_name,
)


def shutdown_database():
    """Manual database shutdown for testing.

    Resets all global state to allow clean database reinitialization.
    """
    from backend.database.lifecycle import _close_db

    _close_db()
    reset_lifecycle_state()
    reset_persistence_state()


__all__ = [
    # Persistence operations
    "ensure_database_ready",
    "ensure_graph_ready",
    "ensure_self_entity",
    "persist_episode",
    "update_episode",
    "delete_episode",
    # Query operations
    "get_episode",
    # Driver access
    "FalkorLiteDriver",
    "get_driver",
    "get_falkordb_graph",
    # SELF entity constants
    "SELF_ENTITY_UUID",
    "SELF_ENTITY_NAME",
    "SELF_ENTITY_LABELS",
    # Utilities
    "to_cypher_literal",
    "validate_journal_name",
    # Lifecycle management
    "enable_tcp_server",
    "disable_tcp_server",
    "get_tcp_server_endpoint",
    "get_tcp_server_password",
    "shutdown_database",
]
