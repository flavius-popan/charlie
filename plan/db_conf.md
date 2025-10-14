# Database Configuration Refactoring Plan

## Objective

Enable `run.py` to accept a `--db` flag to toggle between various Kuzu databases (e.g., test.kuzu, charlie.kuzu) and eliminate all hard-coded database names throughout the codebase.

## Current State

### Hard-coded Database References

**Core Application Files:**
- `run.py:18` - `"brain/charlie.kuzu"`
- `load_journals.py:34` - `KUZU_DB_PATH = "brain/charlie.kuzu"`
- `app/services/kuzu_service.py:41` - `def __init__(self, db_path: str = "brain/charlie.kuzu")`
- `app/routes/episodes.py:22` - `kuzu_service = KuzuService()`
- `app/routes/entities.py:20` - `kuzu_service = KuzuService()`
- `app/routes/communities.py:23` - `kuzu_service = KuzuService()`
- `kuzu_explorer.sh:3` - `KUZU_FILE=charlie.kuzu`

**Test Files (18+ references):**
- `tests/test_infrastructure.py` - Multiple hardcoded `"brain/charlie.kuzu"` paths
- `tests/test_communities.py` - Multiple hardcoded `"brain/charlie.kuzu"` paths

**Legacy Files (not critical):**
- `legacy/charlie_local.py:28`
- `legacy/search_examples.py:89`

## Implementation Plan

### 1. Create `app/settings.py`

**Location:** `app/settings.py`

**Purpose:** Central configuration module for application-wide settings

**Content:**
```python
"""
Charlie Application Settings

Central configuration for database paths and other application-wide settings.
"""

# Database configuration
DB_PATH: str = "brain/charlie.kuzu"
BRAIN_DIR: str = "brain"

# Journal loading defaults
JOURNAL_FILE_PATH: str = "raw_data/Journal.json"
SOURCE_DESCRIPTION: str = "journal entry"
```

**Notes:**
- `DB_PATH` is mutable and can be overridden by `run.py` before app startup
- All other scripts import from this module for consistency

### 2. Update `run.py`

**Changes:**
1. Add `--db` command-line argument
2. Import and modify `app.settings.DB_PATH` before starting uvicorn
3. Update `check_database()` to use configured path

**Implementation:**
```python
import argparse
from app import settings

def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Charlie - Interactive Graph Explorer"
    )
    parser.add_argument(
        "--db",
        default="charlie.kuzu",
        help="Database filename in brain/ directory (default: charlie.kuzu)",
    )
    args = parser.parse_args()

    # Configure database path
    settings.DB_PATH = f"{settings.BRAIN_DIR}/{args.db}"

    logger.info("=" * 60)
    logger.info("Charlie - Interactive Graph Explorer")
    logger.info("=" * 60)
    logger.info(f"Database: {settings.DB_PATH}")

    # Check database
    check_database()

    # Start the server
    logger.info("\nStarting development server...")
    logger.info("Server will be available at: http://localhost:8080")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60 + "\n")

    try:
        import uvicorn
        uvicorn.run(
            "app.main:app", host="0.0.0.0", port=8080, reload=True, log_level="info"
        )
    except ImportError:
        logger.error("uvicorn not installed. Install dependencies with:")
        logger.error("  uv pip install -e .")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        logger.info("Goodbye!")


def check_database():
    """Check if the Kuzu database exists."""
    from app import settings

    if not os.path.exists(settings.DB_PATH):
        logger.warning(f"Database not found at {settings.DB_PATH}")
        logger.info("You may need to load journal entries first:")
        logger.info("  python load_journals.py load --skip-verification")
        return False
    return True
```

**Usage Examples:**
```bash
# Use default database (charlie.kuzu)
python run.py

# Use test database
python run.py --db test.kuzu

# Use custom database
python run.py --db experiment.kuzu
```

### 3. Update `app/services/kuzu_service.py`

**Changes:**
- Import from `app.settings`
- Change default parameter to use `settings.DB_PATH`

**Implementation:**
```python
from app import settings

class KuzuService:
    """
    Service for querying Charlie's Kuzu knowledge graph.

    Provides async methods for retrieving nodes and relationships
    with proper connection management.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize KuzuService with database path.

        Args:
            db_path: Path to the Kuzu database directory (uses settings.DB_PATH if None)
        """
        self.db_path = db_path if db_path is not None else settings.DB_PATH
        self._db: Optional[kuzu.Database] = None
        self._conn: Optional[kuzu.Connection] = None
        logger.info(f"KuzuService initialized with database: {self.db_path}")
```

### 4. Update Route Modules

**Files to modify:**
- `app/routes/episodes.py:22`
- `app/routes/entities.py:20`
- `app/routes/communities.py:23`

**Changes:**
All three files instantiate `KuzuService()` at module level. These don't need changes since `KuzuService` now defaults to `settings.DB_PATH`.

**Optional improvement** (for clarity):
```python
from app import settings
from app.services.kuzu_service import KuzuService

# Explicitly pass settings for clarity
kuzu_service = KuzuService(db_path=settings.DB_PATH)
```

### 5. Update `load_journals.py`

**Changes:**
1. Remove `KUZU_DB_PATH` constant
2. Import from `app.settings`
3. Ensure `--db-path` argument still works

**Implementation:**
```python
from app import settings

# Remove this line:
# KUZU_DB_PATH = "brain/charlie.kuzu"

# Update default in argparse
load_parser.add_argument(
    "--db-path",
    default=settings.DB_PATH,
    help=f"Path to Kuzu database (default: {settings.DB_PATH})",
)

communities_parser.add_argument(
    "--db-path",
    default=settings.DB_PATH,
    help=f"Path to Kuzu database (default: {settings.DB_PATH})",
)
```

**Usage remains the same:**
```bash
# Use default database
python load_journals.py load --skip-verification

# Use test database
python load_journals.py load --db-path brain/test.kuzu --skip-verification

# Build communities for test database
python load_journals.py build-communities --db-path brain/test.kuzu
```

### 6. Update Test Files

**Files to modify:**
- `tests/test_infrastructure.py`
- `tests/test_communities.py`

**Strategy:**
Create a pytest fixture that sets up a test database path, either by:
1. Creating a temporary test database
2. Using a dedicated `brain/test.kuzu` database

**Implementation Option 1 - Override settings:**
```python
import pytest
from app import settings

@pytest.fixture(autouse=True)
def use_test_database():
    """Automatically use test database for all tests."""
    original_path = settings.DB_PATH
    settings.DB_PATH = "brain/test.kuzu"
    yield
    settings.DB_PATH = original_path
```

**Implementation Option 2 - Explicit test database:**
```python
import pytest
from pathlib import Path
from app.services.kuzu_service import KuzuService

@pytest.fixture
def test_db_service():
    """Provide KuzuService instance using test database."""
    db_path = Path("brain/test.kuzu")
    assert db_path.exists(), "Test database must exist at brain/test.kuzu"
    return KuzuService(db_path=str(db_path))

def test_kuzu_service_connection(test_db_service):
    """Test KuzuService can connect to test database."""
    service = test_db_service
    service.connect()
    assert service._conn is not None
    service.close()
```

**Changes needed in test files:**
- Replace all `Path("brain/charlie.kuzu")` with `Path(settings.DB_PATH)` or use fixture
- Update assertions that check for specific database path

### 7. Update `kuzu_explorer.sh` (Optional)

**Changes:**
Make the script accept a parameter for which database to explore

**Implementation:**
```bash
#!/bin/bash

# Default to charlie.kuzu if no argument provided
KUZU_FILE=${1:-charlie.kuzu}

docker run -p 8000:8000 \
           -v ./brain:/database \
           -e KUZU_FILE="$KUZU_FILE" \
           --rm kuzudb/explorer:latest
```

**Usage:**
```bash
# Explore default database
./kuzu_explorer.sh

# Explore test database
./kuzu_explorer.sh test.kuzu

# Explore custom database
./kuzu_explorer.sh experiment.kuzu
```

## Implementation Checklist

- [ ] Create `app/settings.py` with default configuration
- [ ] Update `run.py` to accept `--db` flag and configure settings
- [ ] Update `app/services/kuzu_service.py` to use settings
- [ ] Verify route modules work with new settings (likely no changes needed)
- [ ] Update `load_journals.py` to use settings module
- [ ] Create test fixture for database configuration
- [ ] Update `tests/test_infrastructure.py` to use settings or fixture
- [ ] Update `tests/test_communities.py` to use settings or fixture
- [ ] Update `kuzu_explorer.sh` to accept database parameter
- [ ] Test with multiple databases:
  - [ ] `python run.py` (default charlie.kuzu)
  - [ ] `python run.py --db test.kuzu`
  - [ ] `python load_journals.py load --db-path brain/test.kuzu`
- [ ] Verify no hardcoded database paths remain (use grep)

## Verification Commands

After implementation, verify no hardcoded paths remain:

```bash
# Search for hardcoded charlie.kuzu references
grep -r "charlie\.kuzu" --include="*.py" --exclude-dir=legacy .

# Search for hardcoded test.kuzu references
grep -r "test\.kuzu" --include="*.py" --exclude-dir=legacy .

# Search for hardcoded brain/ paths
grep -r '"brain/' --include="*.py" --exclude-dir=legacy .
```

All references should either:
- Import from `app.settings`
- Accept path as parameter
- Be in legacy files (can ignore)

## Benefits

1. **Single source of truth** - All configuration in `app/settings.py`
2. **Easy testing** - Switch databases via command-line flag
3. **No environment variable management** - Simple Python module
4. **Maintainable** - Clear import chain from settings module
5. **Flexible** - Easy to add other configuration values later

## Edge Cases & Considerations

1. **Reload mode in uvicorn**: The `reload=True` flag may cause issues if settings are modified during runtime. This shouldn't be a problem since settings are only set once at startup.

2. **Multiple KuzuService instances**: If different parts of the app instantiate their own KuzuService, they'll all use the same `settings.DB_PATH` (desired behavior).

3. **Test isolation**: Tests must ensure they either share a test database or properly clean up between tests.

4. **Path validation**: Consider adding validation to ensure database path exists or can be created.

## Future Enhancements

- Add more settings to `app/settings.py` (API keys, model configs, etc.)
- Support for `.env` file overrides if needed
- Database path validation with helpful error messages
- Multiple database connections if needed for data migration
