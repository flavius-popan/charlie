"""Database initialization, shutdown, and process lifecycle management."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from threading import Lock
from typing import Any

from backend.database.utils import FalkorGraph, validate_journal_name
from backend.settings import DB_PATH, REDIS_TCP_ENABLED, TCP_HOST, TCP_PASSWORD, TCP_PORT

try:
    from redislite.falkordb_client import FalkorDB
except Exception:
    FalkorDB = None  # type: ignore[assignment]

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Global database state
_db_lock = Lock()
_db = None
_graphs: dict[str, FalkorGraph] = {}
_graph_locks: dict[str, Lock] = {}
_db_unavailable = False
_shutdown_requested = False
_redis_dir: Path | None = None  # Track redislite's temp directory for cleanup

# TCP server configuration
_tcp_server = {
    "enabled": REDIS_TCP_ENABLED and TCP_PORT > 0,
    "host": TCP_HOST,
    "port": TCP_PORT,
    "password": TCP_PASSWORD,
}


def _tcp_server_active() -> bool:
    """Check if TCP server is configured and enabled."""
    port = _tcp_server["port"]
    return bool(_tcp_server["enabled"] and port and port > 0)


def _build_serverconfig() -> dict[str, str] | None:
    """Build FalkorDB server configuration for TCP endpoint."""
    if not _tcp_server_active():
        return None

    config: dict[str, str] = {"port": str(int(_tcp_server["port"]))}
    if _tcp_server["host"]:
        config["bind"] = _tcp_server["host"]
    if _tcp_server["password"]:
        config["requirepass"] = _tcp_server["password"]
    return config


def get_tcp_server_endpoint() -> tuple[str, int] | None:
    """Return (host, port) if TCP server is configured, otherwise None."""
    if not _tcp_server_active():
        return None
    host = _tcp_server["host"] or "127.0.0.1"
    return host, int(_tcp_server["port"])


def get_tcp_server_password() -> str | None:
    """Return the configured password for the TCP server (if any)."""
    if not _tcp_server_active():
        return None
    return _tcp_server["password"]


def enable_tcp_server(
    *,
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
) -> None:
    """Expose FalkorDB Lite on a TCP port in addition to the default unix socket."""
    global _tcp_server
    if port is not None:
        if port <= 0:
            logger.warning("Ignoring TCP enable request with invalid port %s", port)
            return
        _tcp_server["port"] = port
    if host:
        _tcp_server["host"] = host
    if password is not None:
        _tcp_server["password"] = password or None

    _tcp_server["enabled"] = bool(_tcp_server["port"])
    if _db is not None:
        logger.warning(
            "FalkorDB Lite TCP configuration updated after initialization; "
            "restart the process to apply the change."
        )


def disable_tcp_server() -> None:
    """Disable the TCP listener (unix socket access remains available)."""
    _tcp_server["enabled"] = False


def _pid_exists(pid: int) -> bool:
    """Check if process with given PID exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Return True when the process exits within timeout seconds."""
    if pid <= 0:
        return True

    deadline = time.time() + timeout
    if psutil is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return True
        while time.time() < deadline:
            if not process.is_running():
                return True
            try:
                process.wait(0.1)
                return True
            except psutil.TimeoutExpired:
                continue
        return False

    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)
    return False


def _send_signal(pid: int, sig: signal.Signals) -> None:
    """Send signal to process with given PID."""
    if pid <= 0:
        return
    if psutil is not None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        try:
            process.send_signal(sig)
        except Exception:
            logger.debug(
                "Failed to send %s to redis pid %s", sig.name, pid, exc_info=True
            )
        return
    try:
        os.kill(pid, sig.value)
    except OSError:
        logger.debug("os.kill failed for pid %s signal %s", pid, sig, exc_info=True)


def _ensure_redis_stopped(redis_client) -> None:
    """Best-effort shutdown of the embedded redis-server process.

    Uses direct SIGTERM instead of the blocking shutdown() command to avoid
    a known redis-py client timeout issue that causes 5-10 second delays.
    See: https://github.com/redis/redis-py/issues/3704

    Also cleans up the redis temp directory (socket file, pidfile) to match
    what redislite's _cleanup() does, but without the blocking shutdown call.
    """
    global _redis_dir

    if redis_client is None:
        return

    pid = getattr(redis_client, "pid", 0)
    if pid <= 0:
        return

    try:
        redis_client.connection_pool.disconnect()
    except Exception:
        logger.debug(
            "Failed to disconnect redis connection pool cleanly", exc_info=True
        )

    _send_signal(pid, signal.SIGTERM)
    if not _wait_for_exit(pid, timeout=3.0):
        logger.warning("redis-server pid %s ignored SIGTERM; sending SIGKILL", pid)
        _send_signal(pid, signal.SIGKILL)
        _wait_for_exit(pid, timeout=1.0)

    # Clean up the redis temp directory (contains socket file, pidfile, etc.)
    # This matches redislite's cleanup behavior but without the blocking shutdown()
    if _redis_dir and _redis_dir.exists():
        try:
            shutil.rmtree(_redis_dir, ignore_errors=True)
            logger.debug("Cleaned up redis temp directory: %s", _redis_dir)
        except OSError:
            logger.debug("Failed to clean up redis temp directory", exc_info=True)
    _redis_dir = None


def _init_db() -> None:
    """Initialize the FalkorDB database connection."""
    global _db, _db_unavailable, _redis_dir

    if FalkorDB is None:
        logger.debug("FalkorDB client not available")
        _db_unavailable = True
        raise RuntimeError("FalkorDB Lite is unavailable")

    # Ensure database directory exists (deferred from module level to avoid
    # creating production directory when tests override DB_PATH)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        serverconfig = _build_serverconfig()
        kwargs: dict[str, Any] = {}
        if serverconfig:
            kwargs["serverconfig"] = serverconfig
        _db = FalkorDB(dbfilename=str(DB_PATH), **kwargs)

        # Capture redis temp directory path for cleanup (contains socket, pidfile)
        client = getattr(_db, "client", None)
        if client and hasattr(client, "redis_dir") and client.redis_dir:
            _redis_dir = Path(client.redis_dir)

        if serverconfig:
            endpoint = get_tcp_server_endpoint()
            if endpoint:
                host, port = endpoint
                logger.info(
                    "FalkorDB Lite TCP debug endpoint exposed on %s:%d",
                    host,
                    port,
                )
    except Exception as exc:
        logger.warning("Failed to initialize FalkorDB Lite: %s", exc)
        _db_unavailable = True
        raise RuntimeError("FalkorDB Lite is unavailable") from exc


def _ensure_graph(journal: str) -> tuple[FalkorGraph, Lock]:
    """Get or create a FalkorDB graph for the specified journal.

    Thread safety: Uses _db_lock to protect graph cache initialization.
    Multiple threads can safely call this function concurrently. The lock
    ensures that exactly one graph instance and one lock are created per
    journal, even under concurrent access.

    Args:
        journal: Journal name (used as graph name)

    Returns:
        Tuple of (graph instance, lock for this journal)

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If FalkorDB is unavailable
    """
    validate_journal_name(journal)

    global _db, _graphs, _graph_locks, _db_unavailable

    if _db_unavailable:
        raise RuntimeError("FalkorDB Lite is unavailable")

    with _db_lock:
        # Return cached graph and lock if exists
        if journal in _graphs:
            return _graphs[journal], _graph_locks[journal]

        # Initialize database if needed
        if _db is None:
            _init_db()

        # Select and cache the graph for this journal
        _graphs[journal] = _db.select_graph(journal)
        _graph_locks[journal] = Lock()
        logger.debug("Created graph for journal '%s'", journal)
        return _graphs[journal], _graph_locks[journal]


def _close_db():
    """Ensure the embedded FalkorDB process shuts down cleanly.

    Skips db.close() because it calls client.shutdown() which has a
    known blocking timeout issue. We handle shutdown directly via SIGTERM.

    Sets shutdown flag to fail-fast reject any new operations attempting to
    start during shutdown (no waiting - operations fail immediately).
    """
    global _db, _graphs, _graph_locks, _shutdown_requested, _redis_dir

    # Set shutdown flag FIRST (before lock) for immediate fail-fast
    _shutdown_requested = True

    with _db_lock:
        if _db is None:
            return
        client = getattr(_db, "client", None)
        try:
            _ensure_redis_stopped(client)
        finally:
            _db = None
            _graphs.clear()
            _graph_locks.clear()
            _redis_dir = None


atexit.register(_close_db)


def get_falkordb_graph(journal: str) -> FalkorGraph:
    """Return a FalkorDB graph instance for the specified journal.

    Args:
        journal: Journal name

    Returns:
        Graph instance for the journal

    Raises:
        ValueError: If journal name is invalid
        RuntimeError: If FalkorDB is unavailable
    """
    graph, _lock = _ensure_graph(journal)
    return graph


def is_shutdown_requested() -> bool:
    """Check if database shutdown has been requested."""
    return _shutdown_requested


def reset_lifecycle_state() -> None:
    """Reset lifecycle state flags (for testing)."""
    global _db_unavailable, _shutdown_requested, _redis_dir
    _db_unavailable = False
    _shutdown_requested = False
    _redis_dir = None


__all__ = [
    "_ensure_graph",
    "get_falkordb_graph",
    "enable_tcp_server",
    "disable_tcp_server",
    "get_tcp_server_endpoint",
    "get_tcp_server_password",
    "is_shutdown_requested",
    "reset_lifecycle_state",
]
