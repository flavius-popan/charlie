# Database Isolation Plan

**Goal:** Prevent tests from writing to production `data/charlie.db` by using `tests/data/charlie-test.db` for all test runs.

**Approach:** Configure database path override in conftest.py files BEFORE any backend imports. Keep settings.py clean.

---

## Implementation Tasks

### Task 1: Update AGENTS.md Documentation

**File:** `AGENTS.md`
**Action:** Add new section after line 9 (after "## NO GIT PUSH, EVER")

**Insert:**
```markdown
## DATABASE ISOLATION - NEVER WRITE TO PRODUCTION DB

**CRITICAL RULE: Agents and tests must NEVER write to `data/charlie.db`**

The test suite automatically uses `tests/data/charlie-test.db` via conftest.py fixtures.

**Verification after running tests:**
```bash
# Production database should NEVER change during tests
git status data/charlie.db  # Should show "nothing to commit"
```

If `data/charlie.db` was modified, tests have a critical isolation bug that must be fixed immediately.

**For manual testing/development:**
```bash
# Use environment variable to override database location
export CHARLIE_DB_PATH="tests/data/manual-test.db"
python charlie.py
```
```

---

### Task 2: Update backend/settings.py for Environment Variable Support

**File:** `backend/settings.py`
**Current line 8:** `DB_PATH = Path("data/charlie.db")`

**Replace with:**
```python
import os

DB_PATH = Path(os.getenv("CHARLIE_DB_PATH", "data/charlie.db"))
```

**Rationale:** Keeps settings.py clean. Environment variable allows manual testing override. Test isolation handled entirely in conftest.py.

---

### Task 3: Update Root conftest.py

**File:** `tests/conftest.py`
**Action:** Add database path override at the TOP of the file (before line 1, before any other imports)

**Insert at line 1:**
```python
"""Root test configuration.

CRITICAL: Database path MUST be overridden before any backend imports.
"""
from pathlib import Path

# Override database path BEFORE importing backend modules
import backend.settings as backend_settings
backend_settings.DB_PATH = Path("tests/data/charlie-test.db")

# Now safe to import other modules
```

**Then update existing content:**
- Current imports start at line 1
- They should now start after the DB_PATH override
- Add comment at line ~30 (after TCP endpoint disable) explaining isolation:

```python
# Database isolation: DB_PATH overridden to tests/data/charlie-test.db at top of file
# This prevents all tests from writing to production data/charlie.db
```

---

### Task 4: Clean Up tests/data/ Directory

**Actions:**
1. Remove `tests/data/charlie.db` (polluted test artifact)
2. Remove `tests/data/charlie.db.settings` (if exists)
3. Verify `tests/data/.gitignore` contains:
   ```
   *
   !.gitignore
   ```

**Files to delete:**
- `tests/data/charlie.db`
- `tests/data/charlie.db.settings`

**File to verify:** `tests/data/.gitignore`

---

### Task 5: Verify Backend Test Conftest

**File:** `tests/test_backend/conftest.py`
**Current:** Uses `falkordb_test_context` fixture with temporary databases (lines 18-39)

**Action:** No changes needed. Backend tests use per-test temporary databases for maximum isolation. This is more thorough than the shared `tests/data/charlie-test.db` and should be kept as-is.

**Verification:** Ensure backend tests don't import backend modules at module level before fixtures run.

---

### Task 6: Verify Frontend Test Patterns

**File:** `tests/test_frontend/test_charlie.py`
**Current:** Uses `mock_database` fixture that patches high-level functions

**Issue:** If any code imports backend.database modules at module level (before fixtures), writes could bypass mocks and hit the database.

**Action:** Review import statements at top of file (lines 1-20). Ensure no backend.database imports at module level.

**If backend imports exist at module level:** Move them inside test functions or fixtures.

**Example fix if needed:**
```python
# BAD - imports at module level
from backend.database.persistence import persist_episode

def test_something():
    # test code

# GOOD - imports inside function/fixture
def test_something():
    from backend.database.persistence import persist_episode
    # test code
```

---

### Task 7: Add Verification Test

**File:** `tests/test_database_isolation.py` (NEW)
**Action:** Create new test file to verify isolation

**Content:**
```python
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
```

---

### Task 8: Reset Production Database (Optional)

**Current state:** `data/charlie.db` is 87KB (polluted with test data)

**Options:**
1. Delete and let it recreate clean: `rm data/charlie.db`
2. Restore from backup (if available)
3. Keep polluted version for forensics, rename: `mv data/charlie.db data/charlie.db.polluted`

**Recommendation:** After implementing all fixes and verifying tests pass, delete `data/charlie.db` and let the application create a fresh one on first real use.

---

## Verification Steps

After implementing all tasks:

1. **Verify isolation test passes:**
   ```bash
   pytest tests/test_database_isolation.py -v
   ```

2. **Check production DB before tests:**
   ```bash
   ls -lh data/charlie.db
   # Note the size
   ```

3. **Run full test suite:**
   ```bash
   pytest tests/ -v
   ```

4. **Check production DB after tests:**
   ```bash
   ls -lh data/charlie.db
   # Size should be UNCHANGED
   ```

5. **Verify test DB was created:**
   ```bash
   ls -lh tests/data/charlie-test.db
   # Should exist and contain test data
   ```

6. **Check git status:**
   ```bash
   git status data/charlie.db
   # Should show no modifications
   ```

---

## Expected Results

- ✅ Production `data/charlie.db` never modified by tests
- ✅ Tests use `tests/data/charlie-test.db` automatically
- ✅ New verification test passes
- ✅ All existing tests continue to pass
- ✅ Settings.py remains clean (only env var support added)
- ✅ All isolation logic contained in conftest.py

---

## Rollback Plan

If issues arise:

1. Revert `tests/conftest.py` changes (remove DB_PATH override)
2. Revert `backend/settings.py` changes (remove env var support)
3. Tests will temporarily write to production DB again
4. Debug and re-apply with fixes

**Note:** This should not be necessary if tasks are followed in order.
