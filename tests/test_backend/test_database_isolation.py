"""Verify test database isolation.

This test ensures tests never write to production data/charlie.db.
"""
from pathlib import Path
import backend.settings as backend_settings


def test_database_path_is_isolated():
    """Verify tests use an isolated database, not production.

    Backend tests may use either:
    - tests/data/charlie-test.db (root conftest default)
    - A temp directory (when falkordb_test_context fixture runs)

    The key requirement is that it's NOT the production path.
    """
    db_path_str = str(backend_settings.DB_PATH)
    production_path = "data/charlie.db"

    # Primary check: NOT the production database
    assert production_path not in db_path_str or "tests/data" in db_path_str, (
        f"Tests must not use production database path, "
        f"but DB_PATH is {backend_settings.DB_PATH}. "
        f"Check tests/conftest.py database path override."
    )

    # Verify it's a test database (either tests/data or pytest temp dir)
    is_test_data_dir = "tests/data" in db_path_str
    is_pytest_temp_dir = "pytest-" in db_path_str or "/tmp/" in db_path_str
    is_backend_test_db = "backend-tests.db" in db_path_str

    assert is_test_data_dir or is_pytest_temp_dir or is_backend_test_db, (
        f"Test database should be in tests/data/ or a pytest temp directory, "
        f"but DB_PATH is {backend_settings.DB_PATH}"
    )


def test_production_database_not_used():
    """Verify production database path is NOT in use during tests."""
    production_path = Path("data/charlie.db").resolve()
    test_path = backend_settings.DB_PATH

    # Resolve test path only if it exists (temp paths may not exist yet)
    test_path_resolved = str(test_path.resolve() if test_path.exists() else test_path)

    assert test_path_resolved != str(production_path), (
        "Tests are configured to use PRODUCTION DATABASE! "
        "This is a critical test isolation failure. "
        "Check tests/conftest.py configuration."
    )


def test_redis_directory_cleaned_up_on_shutdown(falkordb_test_context):
    """Verify redis temp directory is removed when database shuts down.

    This ensures proper cleanup matching redislite's behavior, preventing
    stale socket files from causing connection errors in subsequent tests.

    Note: This test uses falkordb_test_context directly instead of isolated_graph
    because we intentionally destroy the database and need to reinitialize it.
    """
    from backend.database import lifecycle
    from backend.settings import DEFAULT_JOURNAL

    # First shutdown any existing database to start fresh
    lifecycle._close_db()
    lifecycle.reset_lifecycle_state()

    # Initialize a fresh database
    lifecycle._ensure_graph(DEFAULT_JOURNAL)

    # Verify redis_dir is tracked (only set during _init_db)
    redis_dir = lifecycle._redis_dir
    assert redis_dir is not None, "Redis directory should be tracked after fresh initialization"
    assert redis_dir.exists(), f"Redis directory should exist at {redis_dir}"

    # Verify the directory contains expected files (socket, pid, etc.)
    files_in_dir = list(redis_dir.iterdir())
    assert len(files_in_dir) > 0, "Redis directory should contain files (socket, pid, etc.)"

    # Capture path before shutdown
    captured_dir = redis_dir

    # Shutdown the database - this cleans up the directory
    lifecycle._close_db()
    lifecycle.reset_lifecycle_state()

    # Verify entire directory is cleaned up (including socket file)
    assert not captured_dir.exists(), (
        f"Redis temp directory should be cleaned up after shutdown: {captured_dir}"
    )

    # Reinitialize the database for subsequent tests
    # (falkordb_test_context fixture will handle final cleanup)
    lifecycle._ensure_graph(DEFAULT_JOURNAL)
