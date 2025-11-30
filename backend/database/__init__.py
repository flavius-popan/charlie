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
from backend.database.queries import (
    episode_exists,
    get_entity_browser_data,
    get_entry_entities,
    get_entry_entities_with_counts,
    get_episode,
    get_home_screen,
    get_n_plus_one_neighbors,
    get_period_entities,
    truncate_quote,
)
from backend.database.redis_ops import (
    get_active_episode_uuid,
    get_episode_status,
    get_processing_status,
    redis_ops,
)
from backend.database.utils import (
    SELF_ENTITY_LABELS,
    SELF_ENTITY_NAME,
    SELF_ENTITY_UUID,
    is_self_entity_name,
    to_cypher_literal,
    validate_journal_name,
)


def shutdown_database():
    """Shut down the database cleanly.

    Stops the embedded Redis process. The shutdown flag remains set
    so tasks know shutdown is in progress. For test re-initialization,
    call reset_lifecycle_state() and reset_persistence_state() explicitly.
    """
    from backend.database.lifecycle import _close_db

    _close_db()
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
    "episode_exists",
    "get_entity_browser_data",
    "get_entry_entities",
    "get_entry_entities_with_counts",
    "get_episode",
    "get_home_screen",
    "get_n_plus_one_neighbors",
    "get_period_entities",
    "truncate_quote",
    # Redis operations
    "get_active_episode_uuid",
    "get_episode_status",
    "get_processing_status",
    "redis_ops",
    # Driver access
    "FalkorLiteDriver",
    "get_driver",
    "get_falkordb_graph",
    # SELF entity constants
    "SELF_ENTITY_UUID",
    "SELF_ENTITY_NAME",
    "SELF_ENTITY_LABELS",
    "is_self_entity_name",
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
