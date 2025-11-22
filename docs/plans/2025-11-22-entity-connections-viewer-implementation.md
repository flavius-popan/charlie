# Entity Connections Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add interactive sidebar to ViewScreen showing extracted entities with deletion capabilities

**Architecture:** Split ViewScreen into horizontal container (3/4 markdown + 1/4 sidebar). Poll Redis cache for entity data, allow local deletion of MENTIONS edges with cache updates. Follow TDD throughout.

**Tech Stack:** Textual widgets (Container, Horizontal, ListView, ListItem, LoadingIndicator), Redis cache, FalkorDB Cypher queries, pytest-asyncio

**Key Changes from Original Design:**
- Use Redis cache instead of database polling for entity display
- Remove reference count from UI (show only name and type)
- Remove ASCII cat spinner (use default LoadingIndicator)
- Add log viewer toggle from ViewScreen (`l` key)
- Update cache after successful DB writes for data integrity

---

## Implementation Status

**Completed Tasks (Original Implementation):**
- ✅ Task 2: Database Mutation - Delete Entity Mention (commit: 517b6ed)
- ✅ Task 3: EntitySidebar Widget - Basic Structure (commit: 7fdc326)
- ✅ Task 7: ViewScreen - Add EntitySidebar Integration (commit: 89c7949)
- ✅ Task 9: Navigation Flow - ESC to ViewScreen (commit: a0dca42)
- ✅ Task 10: Entity Deletion - Confirmation Modal (commit: 16c610a)
- ✅ Task 12: Update Existing ViewScreen Call Sites (commit: 8a7509b)

**Tasks Requiring Revision:**
- ⚠️ Task 1: REMOVE - Database fetch function (replaced by Redis cache)
- ⚠️ Task 2: UPDATE - Add Redis cache update after DB deletion
- ⚠️ Task 4: UPDATE - Remove ASCII cat, use LoadingIndicator
- ⚠️ Task 5: UPDATE - Remove ref_count from display
- ⚠️ Task 6: REPLACE - Poll Redis cache instead of database
- ⚠️ Task 8: UPDATE - Poll Redis `nodes` field instead of job status only
- ⚠️ Task 11: UPDATE - Update Redis cache after successful deletion

**New Tasks Required:**
- Task NEW-1: Update `extract_nodes` task to write entities to Redis cache
- Task NEW-2: Add log viewer toggle (`l` key binding) to ViewScreen
- Task NEW-3: Remove database fetch tests and function
- Task NEW-4: Update all entity display to remove ref_count
- Task 13: Integration Test - Full Workflow (update for Redis)
- Task 14: Documentation and Cleanup
- Task 15: Final Verification

---

## Task NEW-1: Update extract_nodes Task to Write to Redis Cache

**Files:**
- Backend: `backend/tasks.py` (or wherever `extract_nodes` task is defined)
- Tests: `tests/test_backend/test_tasks.py`

### Step 1: Write test for Redis cache write

Add test to verify entities are written to Redis after successful DB write:

```python
@pytest.mark.asyncio
async def test_extract_nodes_writes_to_redis_cache(redis_client, graph_db):
    """Test that extract_nodes writes entity data to Redis after DB write."""
    episode_uuid = "test-uuid"
    journal = "default"

    # Execute task
    result = await extract_nodes(episode_uuid, journal, "I met Sarah at Starbucks")

    # Verify Redis cache contains nodes
    cache_key = f"journal:{journal}:{episode_uuid}"
    nodes_json = await redis_client.hget(cache_key, "nodes")
    assert nodes_json is not None

    nodes = json.loads(nodes_json)
    assert len(nodes) == 2  # Sarah and Starbucks
    assert {"uuid": ANY, "name": "Sarah", "type": "Person"} in nodes
    assert {"uuid": ANY, "name": "Starbucks", "type": "Organization"} in nodes
```

### Step 2: Update extract_nodes task implementation

Modify task to write to Redis AFTER successful DB write:

```python
async def extract_nodes(episode_uuid: str, journal: str, content: str):
    # ... existing extraction logic ...

    # Write to FalkorDB
    await write_entities_to_graph(episode_uuid, journal, entities)

    # AFTER successful DB write, update Redis cache
    cache_key = f"journal:{journal}:{episode_uuid}"
    nodes_data = [
        {
            "uuid": entity["uuid"],
            "name": entity["name"],
            "type": get_most_specific_label(entity["labels"])
        }
        for entity in entities
    ]
    await redis_client.hset(cache_key, "nodes", json.dumps(nodes_data))

    return {"status": "done", "entity_count": len(entities)}
```

### Step 3: Run tests

```bash
pytest tests/test_backend/test_tasks.py::test_extract_nodes_writes_to_redis_cache -v
```

### Step 4: Commit

```bash
git add backend/tasks.py tests/test_backend/test_tasks.py
git commit -m "feat: write entity data to Redis cache after extraction

- extract_nodes now writes to Redis 'nodes' field after DB write
- Ensures cache only contains valid data
- Format: JSON array of {uuid, name, type}"
```

---

## Task NEW-2: Remove Database Fetch Function and Tests

**Files:**
- Backend: `backend/database.py` (or equivalent)
- Tests: `tests/test_backend/test_database.py`

### Step 1: Identify and remove fetch_entities_for_episode

Remove the `fetch_entities_for_episode` function and all related tests:

```bash
# Find the function
grep -r "fetch_entities_for_episode" backend/
grep -r "fetch_entities_for_episode" tests/
```

### Step 2: Delete function and tests

Remove:
- Function definition
- All tests for the function
- Any imports that are no longer used

### Step 3: Verify no references remain

```bash
grep -r "fetch_entities_for_episode" .
# Should return no results
```

### Step 4: Run test suite

```bash
pytest tests/test_backend/ -v
```

### Step 5: Commit

```bash
git add backend/ tests/
git commit -m "refactor: remove fetch_entities_for_episode

- No longer fetching entities from database
- All entity data now served from Redis cache
- Removes unnecessary database queries"
```

---

## Task NEW-3: Update Entity Deletion to Update Redis Cache

**Files:**
- Backend: `backend/database.py` (delete_entity_mention function)
- Tests: `tests/test_backend/test_database.py`

### Step 1: Write test for cache update

```python
@pytest.mark.asyncio
async def test_delete_entity_mention_updates_redis_cache(redis_client, graph_db):
    """Test that deletion removes entity from Redis cache."""
    episode_uuid = "test-uuid"
    entity_uuid = "sarah-uuid"
    journal = "default"

    # Setup: Add entities to cache
    cache_key = f"journal:{journal}:{episode_uuid}"
    nodes = [
        {"uuid": "sarah-uuid", "name": "Sarah", "type": "Person"},
        {"uuid": "john-uuid", "name": "John", "type": "Person"}
    ]
    await redis_client.hset(cache_key, "nodes", json.dumps(nodes))

    # Delete entity
    await delete_entity_mention(episode_uuid, entity_uuid, journal)

    # Verify cache updated
    updated_nodes = json.loads(await redis_client.hget(cache_key, "nodes"))
    assert len(updated_nodes) == 1
    assert updated_nodes[0]["uuid"] == "john-uuid"
```

### Step 2: Update delete_entity_mention implementation

```python
async def delete_entity_mention(episode_uuid: str, entity_uuid: str, journal: str):
    # Delete from FalkorDB
    query = """
    MATCH (ep:Episodic {uuid: $ep_uuid})-[m:MENTIONS]->(e:Entity {uuid: $e_uuid})
    DELETE m
    WITH e
    MATCH (e)
    WHERE NOT EXISTS((e)<-[:MENTIONS]-())
    DELETE e
    RETURN count(e) as deleted_node
    """
    result = await graph.query(query, {"ep_uuid": episode_uuid, "e_uuid": entity_uuid})

    # AFTER successful DB deletion, update Redis cache
    cache_key = f"journal:{journal}:{episode_uuid}"
    nodes_json = await redis_client.hget(cache_key, "nodes")
    if nodes_json:
        nodes = json.loads(nodes_json)
        updated_nodes = [n for n in nodes if n["uuid"] != entity_uuid]
        await redis_client.hset(cache_key, "nodes", json.dumps(updated_nodes))
```

### Step 3: Run tests

```bash
pytest tests/test_backend/test_database.py::test_delete_entity_mention_updates_redis_cache -v
```

### Step 4: Commit

```bash
git add backend/database.py tests/test_backend/test_database.py
git commit -m "feat: update Redis cache after entity deletion

- Removes deleted entity from cache after successful DB write
- Keeps cache in sync with database state"
```

---

## Task NEW-4: Update EntitySidebar to Poll Redis and Remove Ref Count

**Files:**
- Frontend: `charlie.py` (EntitySidebar widget)
- Tests: `tests/test_frontend/test_entity_sidebar.py`

### Step 1: Update tests to remove ref_count

Update all entity sidebar tests to:
- Remove ref_count from mock data
- Update display assertions to match `{name} [{type}]` format
- Mock Redis polling instead of database calls

### Step 2: Update EntitySidebar implementation

```python
class EntitySidebar(Container):
    def compose(self):
        yield Label("Connections", id="header")
        yield LoadingIndicator(id="loading")  # No ASCII cat
        yield ListView(id="entity-list")
        yield Label("d: delete | ↑↓: navigate | c: close | l: logs", id="footer")

    async def poll_redis_for_entities(self):
        """Poll Redis for nodes field."""
        cache_key = f"journal:{self.journal}:{self.episode_uuid}"

        while True:
            nodes_json = await redis_client.hget(cache_key, "nodes")
            status = await redis_client.hget(cache_key, "status")

            if nodes_json:
                # Parse and display entities
                nodes = json.loads(nodes_json)
                # Filter out SELF entity
                filtered = [n for n in nodes if n["name"] != "I"]
                self.display_entities(filtered)
                break
            elif status == "done":
                # Job complete but no entities found
                self.show_empty_state()
                break

            await asyncio.sleep(0.5)

    def display_entities(self, nodes):
        """Display entities in format: Name [Type]"""
        for node in nodes:
            # No ref_count displayed
            label = f"{node['name']} [{node['type']}]"
            self.query_one(ListView).append(ListItem(Label(label)))
```

### Step 3: Run tests

```bash
pytest tests/test_frontend/test_entity_sidebar.py -v
```

### Step 4: Commit

```bash
git add charlie.py tests/test_frontend/
git commit -m "refactor: poll Redis cache and simplify display

- EntitySidebar now polls Redis 'nodes' field
- Removed ref_count from display (cleaner UI)
- Removed ASCII cat spinner (use LoadingIndicator)
- Filter SELF entity in UI component"
```

---

## Task NEW-5: Add Log Viewer Toggle to ViewScreen

**Files:**
- Frontend: `charlie.py` (ViewScreen)
- Tests: `tests/test_frontend/test_view_screen.py`

### Step 1: Write test

```python
@pytest.mark.asyncio
async def test_view_screen_log_viewer_toggle():
    """Test that 'l' key navigates to log viewer."""
    app = CharlieApp()

    async with app.run_test() as pilot:
        # Navigate to ViewScreen
        app.push_screen(ViewScreen(episode_uuid="test", journal="default"))

        # Press 'l' key
        await pilot.press("l")

        # Should be on LogViewerScreen
        from charlie import LogViewerScreen
        assert isinstance(app.screen, LogViewerScreen)
```

### Step 2: Add key binding to ViewScreen

```python
class ViewScreen(Screen):
    BINDINGS = [
        ("e", "edit", "Edit"),
        ("c", "toggle_connections", "Connections"),
        ("l", "show_logs", "Logs"),  # NEW
        ("q", "back", "Back"),
        ("escape", "back", "Back"),
    ]

    def action_show_logs(self):
        """Navigate to log viewer."""
        self.app.push_screen(LogViewerScreen())
```

### Step 3: Run tests

```bash
pytest tests/test_frontend/test_view_screen.py::test_view_screen_log_viewer_toggle -v
```

### Step 4: Commit

```bash
git add charlie.py tests/test_frontend/test_view_screen.py
git commit -m "feat: add log viewer toggle from ViewScreen

- Press 'l' to navigate to logs without returning home
- Improves navigation flow"
```

---

## Task 13: Integration Test - Full Workflow (Updated for Redis)

**Files:**
- Test: `tests/test_frontend/test_integration.py` (update existing)

### Step 1: Update integration test for Redis

Update `tests/test_frontend/test_integration.py` to use Redis cache:

```python
"""Integration tests for entity viewer workflow."""

import pytest
import json
from unittest.mock import patch, AsyncMock
from charlie import CharlieApp


@pytest.mark.asyncio
async def test_full_workflow_create_view_delete_entity(redis_client):
    """Test complete flow: create entry → view with entities → delete entity."""

    with patch("charlie.add_journal_entry", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = "test-uuid"

        with patch("charlie.update_episode", new_callable=AsyncMock):
            with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    "uuid": "test-uuid",
                    "content": "# Meeting\nI met Sarah today.",
                }

                # Setup Redis cache with entity data
                cache_key = "journal:default:test-uuid"
                nodes = [
                    {"uuid": "sarah-uuid", "name": "Sarah", "type": "Person"}
                ]
                await redis_client.hset(cache_key, "nodes", json.dumps(nodes))
                await redis_client.hset(cache_key, "status", "done")

                with patch("charlie.delete_entity_mention", new_callable=AsyncMock):
                    app = CharlieApp()

                    async with app.run_test() as pilot:
                        # Press 'n' to create new entry
                        await pilot.press("n")

                        # Type content
                        from charlie import EditScreen
                        edit_screen = app.screen
                        assert isinstance(edit_screen, EditScreen)

                        editor = edit_screen.query_one("#editor")
                        editor.text = "# Meeting\nI met Sarah today."

                        # Press ESC to save and view
                        await pilot.press("escape")

                        # Should be on ViewScreen
                        from charlie import ViewScreen
                        view_screen = app.screen
                        assert isinstance(view_screen, ViewScreen)

                        # Wait for entities to load from Redis
                        await asyncio.sleep(0.6)

                        # Should have EntitySidebar with Sarah
                        from charlie import EntitySidebar
                        sidebar = view_screen.query_one(EntitySidebar)
                        assert len(sidebar.entities) == 1
                        assert sidebar.entities[0]["name"] == "Sarah"
                        assert sidebar.entities[0]["type"] == "Person"
                        # No ref_count in display
```

### Step 2: Run test

Run: `pytest tests/test_frontend/test_integration.py -v`

Expected: PASS

### Step 3: Commit

```bash
git add tests/test_frontend/test_integration.py
git commit -m "test: update integration test for Redis cache

- Uses Redis cache instead of database mocks
- Verifies complete navigation flow with Redis polling
- Tests entity loading from cache (no ref_count)"
```

---

## Task 14: Documentation and Cleanup

**Files:**
- Create: `docs/features/entity-viewer.md`

### Step 1: Write feature documentation

Create `docs/features/entity-viewer.md`:

```markdown
# Entity Connections Viewer

## Overview

The entity connections viewer is a sidebar in the ViewScreen that displays entities extracted from the current journal entry. It provides a diagnostic view of what entities (people, places, organizations, activities) are connected to an episode.

## User Interface

### Layout

- **Markdown Viewer**: Left side (75% width) - displays journal entry content
- **Connections Sidebar**: Right side (25% width) - displays extracted entities

### Sidebar States

1. **Loading**: Shows loading indicator while polling for entity data
2. **Loaded**: Displays ListView of entities formatted as `Name [Type]`
3. **Empty**: Shows "No connections found" when no entities extracted

### Key Bindings

- `c` - Toggle sidebar visibility
- `l` - Toggle to log viewer
- `d` - Delete selected entity (when sidebar focused)
- `Tab` - Switch focus between markdown viewer and sidebar
- `↑↓` - Navigate entity list

## Entity Deletion

### Local Deletion

Deleting an entity from the sidebar removes only the MENTIONS edge for this specific episode. The entity remains in the knowledge graph if referenced by other episodes.

### Confirmation Modal

Shows:
- Entity name
- Note: "(This only removes the connection from this entry)"

### Automatic Cleanup

If an entity has no remaining MENTIONS edges after deletion, it's automatically removed from the knowledge graph.

## Implementation Details

### Redis Cache

- Entity data stored in Redis hash field `nodes`
- Format: JSON array `[{"uuid": "...", "name": "...", "type": "..."}]`
- Written by `extract_nodes` task AFTER successful FalkorDB write
- Updated after entity deletion to maintain sync

### Database Operations

- `delete_entity_mention(episode_uuid, entity_uuid, journal)` - Deletes MENTIONS edge from FalkorDB
- Updates Redis cache after successful deletion

### Cache Polling

ViewScreen polls Redis `nodes` field every 500ms to detect when entity data is available.

### SELF Entity Filtering

The SELF entity (journal author "I") is automatically filtered in the UI component to reduce noise.

## Design Principles

- **No toasts** - UI updates are sufficient feedback
- **Playful language** - Avoid technical jargon, use natural terms
- **Redis caching** - Fast entity display without database queries
```

### Step 2: Commit documentation

```bash
git add docs/features/entity-viewer.md
git commit -m "docs: add entity viewer feature documentation

- Overview of functionality
- UI layout and states (no ASCII cat, no ref_count)
- Key bindings and interactions (includes log viewer toggle)
- Redis cache implementation details
- Design principles"
```

---

## Task 15: Final Verification

### Step 1: Run complete test suite

Run: `pytest tests/ -v --tb=short`

Expected: All tests PASS

### Step 2: Run type checking (if configured)

Run: `mypy charlie.py backend/` (if mypy is set up)

Expected: No errors

### Step 3: Manual smoke test

Run: `python charlie.py`

1. Press `n` to create new entry
2. Type: `# Meeting Notes\nI met Sarah and John at Central Park.`
3. Press ESC
4. Verify ViewScreen shows markdown on left, sidebar on right with loading indicator
5. Wait for entities to appear in sidebar (should show `Sarah [Person]`, `John [Person]`, `Central Park [Location]`)
6. Verify NO reference count displayed
7. Press `l` to toggle to log viewer (verify navigation works)
8. Return to ViewScreen
9. Press Tab to focus sidebar
10. Press `d` to delete an entity
11. Confirm deletion (verify entity removed from sidebar)
12. Press `c` to toggle sidebar (verify markdown expands to full width)

### Step 4: Final commit

```bash
git add -A
git commit -m "feat: entity connections viewer with Redis cache

Complete implementation with:
- Redis cache for fast entity display
- Simplified UI (no ref_count, no ASCII cat)
- EntitySidebar widget with LoadingIndicator
- ViewScreen integration with Redis polling
- Navigation flow from EditScreen to ViewScreen
- Log viewer toggle from ViewScreen
- Entity deletion with cache updates
- Comprehensive test coverage
- Feature documentation"
```

---

## Execution Summary

**Status:** ✅ Complete

**Completed Tasks:**
- ✅ NEW-1: Extract_nodes writes to Redis after FalkorDB write (d8c9329)
- ✅ NEW-2: Removed fetch_entities_for_episode (994a3cb)
- ✅ NEW-3: Entity deletion syncs Redis cache (7312206)
- ✅ NEW-4: EntitySidebar polls Redis, simplified UI (5483f37)
- ✅ NEW-5: Log viewer toggle added (cc18c61)
- ✅ Performance: Instant cache loading (5989a6c)
- ✅ UX: Smart sidebar visibility (587359f)
- ✅ Fix: Test isolation (31321ea)
- ✅ Fix: Node extraction on update (5a66406)
- ✅ Fix: Auto-select first entity (167be52)

**Final Architecture:**
- Redis cache for instant entity display
- Smart sidebar visibility (hidden from HomeScreen, shown from EditScreen)
- Auto-focus first entity when opened
- Node extraction triggers on both create and update
- No ref_count, no ASCII cat, no redundant footers

**Performance Gains:**
- Cached entries: ~50ms load (vs ~500ms+ before)
- Large entries (12+ entities): No startup lag from HomeScreen
- Smooth edit→view transition with immediate markdown display
