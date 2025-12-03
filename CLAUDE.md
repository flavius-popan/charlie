# ASK ME QUESTIONS

Unless I request a simple change, any discussion of architecture, design, feature building, debugging, etc. should include clarifying questions asked ONE AT A TIME so we may align our understanding of the task & high level goals. Prompt me back!

## NO GIT PUSH, EVER

Git add & commit are fine for feature branches but NEVER for `main`!

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

## Comment Rules

- Add code comments sparingly.
- Focus on *why* something is done, especially for complex logic, rather than *what* is done.
- Only add high-value comments if necessary for clarity or if requested by the user.
- Do not edit comments that are separate from the code you are changing. *NEVER* talk to the user or describe your changes through comments.
- Ensure all comments are evergreen and don't reference line numbers or bugs that were previously fixed, espcially for tests.

## Testing Textual Applications

**CRITICAL:** Never run the Textual app directly (e.g., `python charlie.py`) in a non-interactive shell or agent environment. It will hang or output raw terminal control codes.

**Always use headless testing:**
- Run tests via `pytest` using Textual's `run_test()` API
- Use `app.query_one()` to assert widget state, not stdout
- Use `pytest-textual-snapshot` for visual testing (generates SVG files you can read)

**See `tests/test_frontend/test_charlie.py` for comprehensive testing patterns and examples.**

## Testing LLM Inference

**Mark all tests that make real LLM inference calls with `@pytest.mark.inference`:**
- Default behavior: `pytest` excludes inference tests (fast test runs)
- Run inference tests: `pytest -m inference`
- Run all tests: `pytest -m ""`
- Use the `require_llm` fixture to skip if no LLM configured
- Reuse the session-configured LLM (`dspy.settings.lm`) for all inference tests. Only tests that explicitly verify load/unload behavior should instantiate or unload models; everything else should use the shared model seeded by `configure_dspy_for_backend`/`reuse_session_lm`.

**Pattern:**
```python
@pytest.mark.inference
@pytest.mark.asyncio
async def test_with_llm_call(isolated_graph, require_llm):
    # Test automatically skips if dspy.settings.lm is None
    # Test code that makes LLM calls via DSPy
```

## Querying the Knowledge Graph (FalkorDB)

Use inline Python to query entities and episodes. The graph uses Cypher syntax (no regex support).

**Basic entity query:**
```python
python3 -c "
import asyncio
from backend.database.driver import get_driver

async def query():
    driver = get_driver()  # defaults to 'default' journal
    records, _, _ = await driver.execute_query('''
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS 'search_term'
        RETURN e.name as name, e.uuid as uuid
        LIMIT 20
    ''')
    for r in records:
        print(r['name'])

asyncio.run(query())
"
```

**Check entity types from Redis cache:**
```python
python3 -c "
import asyncio, json
from backend.database.driver import get_driver
from backend.database.redis_ops import redis_ops

async def check_types():
    driver = get_driver()
    records, _, _ = await driver.execute_query('MATCH (ep:Episodic) RETURN ep.uuid as uuid')

    with redis_ops() as r:
        for rec in records:
            nodes_json = r.hget(f'journal:default:{rec[\"uuid\"]}', 'nodes')
            if nodes_json:
                for n in json.loads(nodes_json.decode()):
                    print(f'{n.get(\"name\"):30} -> {n.get(\"type\")}')

asyncio.run(check_types())
"
```

**Key patterns:**
- `MATCH (e:Entity)` - query entities
- `MATCH (ep:Episodic)` - query episodes/entries
- `MATCH (ep:Episodic)-[:MENTIONS]->(e:Entity)` - episodes mentioning entities
- Redis cache key: `journal:{journal_name}:{episode_uuid}` with fields: `nodes`, `status`, `mentions_edges`

### Misc. Things to Remember

- Use the respective conftest.py file for a given test suite for fixtures.
- Add imports to the top of the module unless necessary for a given feature.
- Opt for using hardcoded constants unless I explicitly ask to check for an env var.
- Always use the project's uv-managed virtual environment (e.g., `uv add ...`) instead of the system Python for installing dependencies and running tests.
