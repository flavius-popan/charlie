"""Verify test database isolation.

This test ensures tests never write to production data/charlie.db.
"""
from pathlib import Path
import backend.settings as backend_settings


def test_database_path_is_isolated():
    """Verify tests use tests/data/charlie-test.db, not production database."""
    assert "tests/data" in str(backend_settings.DB_PATH), (
        f"Tests must use tests/data/ directory for database, "
        f"but DB_PATH is {backend_settings.DB_PATH}. "
        f"Check tests/conftest.py database path override."
    )

    assert "charlie-test.db" in str(backend_settings.DB_PATH), (
        f"Test database should be charlie-test.db, "
        f"but DB_PATH is {backend_settings.DB_PATH}"
    )


def test_production_database_not_used():
    """Verify production database path is NOT in use during tests."""
    production_path = Path("data/charlie.db")

    assert backend_settings.DB_PATH != production_path, (
        "Tests are configured to use PRODUCTION DATABASE! "
        "This is a critical test isolation failure. "
        "Check tests/conftest.py configuration."
    )
