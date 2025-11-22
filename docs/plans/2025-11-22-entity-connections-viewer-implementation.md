# Entity Connections Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add interactive sidebar to ViewScreen showing extracted entities with deletion capabilities

**Architecture:** Split ViewScreen into horizontal container (3/4 markdown + 1/4 sidebar). Poll Huey job status on mount, fetch entities when complete, allow local deletion of MENTIONS edges. Follow TDD throughout.

**Tech Stack:** Textual widgets (Container, Horizontal, ListView, ListItem), Neo4j Cypher queries, pytest-asyncio

---

## Implementation Status

**Completed Tasks:**
- ✅ Task 1: Database Query - Fetch Entities for Episode (commit: 07a302f)
- ✅ Task 2: Database Mutation - Delete Entity Mention (commit: 517b6ed)
- ✅ Task 3: EntitySidebar Widget - Basic Structure (commit: 7fdc326)
- ✅ Task 4: EntitySidebar - Loading State Display (commit: fe37f07)
- ✅ Task 5: EntitySidebar - Entity List Display (commit: 33a51cc)
- ✅ Task 6: EntitySidebar - Fetch Entities from Database (commit: 3c0e699)
- ✅ Task 7: ViewScreen - Add EntitySidebar Integration (commit: 89c7949)
- ✅ Task 8: ViewScreen - Job Status Polling (commit: 367fb4c)
- ✅ Task 9: Navigation Flow - ESC to ViewScreen (commit: a0dca42)
- ✅ Task 10: Entity Deletion - Confirmation Modal (commit: 16c610a)
- ✅ Task 11: Entity Deletion - Database Integration (commit: 2c3161d)
- ✅ Task 12: Update Existing ViewScreen Call Sites (commit: 8a7509b)
- ✅ **Bug Fix:** Fixed async callback issue in ViewScreen polling (changed to sync with run_worker)
- ✅ **Bug Fix:** Updated status check to use "pending_edges"/"done" instead of "completed"
- ✅ **Enhancement:** Improved entity formatting (ref_count only shown if > 1, better label handling)
- ✅ **Enhancement:** Auto-focus sidebar ListView when entities load for keyboard navigation

**Remaining:**
- Task 13: Integration Test - Full Workflow
- Task 14: Documentation and Cleanup
- Task 15: Final Verification

---

## Task 13: Integration Test - Full Workflow

**Files:**
- Test: `tests/test_frontend/test_integration.py` (new file)

### Step 1: Write integration test

Create `tests/test_frontend/test_integration.py`:

```python
"""Integration tests for entity viewer workflow."""

import pytest
from unittest.mock import patch, AsyncMock
from charlie import CharlieApp


@pytest.mark.asyncio
async def test_full_workflow_create_view_delete_entity():
    """Test complete flow: create entry → view with entities → delete entity."""

    with patch("charlie.add_journal_entry", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = "test-uuid"

        with patch("charlie.update_episode", new_callable=AsyncMock):
            with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    "uuid": "test-uuid",
                    "content": "# Meeting\nI met Sarah today.",
                }

                with patch("charlie.fetch_entities_for_episode", new_callable=AsyncMock) as mock_fetch:
                    mock_fetch.return_value = [
                        {"uuid": "sarah-uuid", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 1},
                    ]

                    with patch("charlie.delete_entity_mention", new_callable=AsyncMock):
                        with patch("charlie.get_episode_status") as mock_status:
                            mock_status.return_value = "pending_edges"

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

                                # Wait for entities to load
                                await asyncio.sleep(0.6)

                                # Should have EntitySidebar with Sarah
                                from charlie import EntitySidebar
                                sidebar = view_screen.query_one(EntitySidebar)
                                assert len(sidebar.entities) == 1
                                assert sidebar.entities[0]["name"] == "Sarah"
```

### Step 2: Run test

Run: `pytest tests/test_frontend/test_integration.py -v`

Expected: PASS

### Step 3: Commit

```bash
git add tests/test_frontend/test_integration.py
git commit -m "test: add integration test for full workflow

- Create entry → ESC → ViewScreen with entities
- Verifies complete navigation flow
- Verifies entity loading and display"
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

The entity connections viewer is a sidebar in the ViewScreen that displays entities extracted from the current journal entry. It provides a diagnostic view of what entities (people, places, organizations, activities) are attached to an episode node.

## User Interface

### Layout

- **Markdown Viewer**: Left side (75% width) - displays journal entry content
- **Connections Sidebar**: Right side (25% width) - displays extracted entities

### Sidebar States

1. **Loading**: Shows animated ASCII cat spinner with "Slinging yarn..." message
2. **Loaded**: Displays ListView of entities formatted as `Name [Type] (RefCount)`
3. **Empty**: Shows "No connections found" when no entities extracted

### Key Bindings

- `c` - Toggle sidebar visibility
- `d` - Delete selected entity (when sidebar focused)
- `Tab` - Switch focus between markdown viewer and sidebar
- `↑↓` - Navigate entity list

## Entity Deletion

### Local Deletion

Deleting an entity from the sidebar removes only the MENTIONS edge for this specific episode. The entity remains in the knowledge graph if referenced by other episodes.

### Confirmation Modal

Shows:
- Entity name
- Reference count hint:
  - `(Will remain in N other entries)` if ref_count > 1
  - `(Will be removed entirely)` if ref_count == 1

### Automatic Cleanup

If an entity has no remaining MENTIONS edges after deletion, it's automatically removed from the knowledge graph.

## Implementation Details

### Database Queries

- `fetch_entities_for_episode(episode_uuid, journal)` - Returns entities with ref counts
- `delete_entity_mention(episode_uuid, entity_uuid, journal)` - Deletes MENTIONS edge

### Job Polling

ViewScreen polls Huey job status every 500ms to detect when entity extraction completes, then fetches and displays entities.

### SELF Entity Filtering

The SELF entity (journal author "I") is automatically filtered from display to reduce noise.

## Design Principles

- **No toasts** - UI updates are sufficient feedback
- **Playful language** - Avoid technical jargon, use natural terms
- **Quirky cat theme** - Subtle personality without being overwhelming
```

### Step 2: Commit documentation

```bash
git add docs/features/entity-viewer.md
git commit -m "docs: add entity viewer feature documentation

- Overview of functionality
- UI layout and states
- Key bindings and interactions
- Implementation details
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

### Step 3: Manual smoke test (optional)

Run: `python charlie.py`

1. Press `n` to create new entry
2. Type: `# Meeting Notes\nI met Sarah and John at Central Park.`
3. Press ESC
4. Verify ViewScreen shows markdown on left, sidebar on right
5. Wait for entities to appear in sidebar
6. Press Tab to focus sidebar
7. Press `d` to delete an entity
8. Confirm deletion
9. Press `c` to toggle sidebar

### Step 4: Final commit

```bash
git add -A
git commit -m "feat: entity connections viewer complete

Complete implementation with:
- Database queries for fetch and delete
- EntitySidebar widget with loading/loaded states
- ViewScreen integration with job polling
- Navigation flow from EditScreen to ViewScreen
- Entity deletion with confirmation
- Comprehensive test coverage
- Feature documentation"
```

---

## Execution Complete

All tasks implemented following TDD:
- Database layer (fetch, delete)
- UI components (EntitySidebar, ViewScreen updates)
- Navigation flow changes
- Entity deletion workflow
- Integration tests
- Documentation

Ready for code review with superpowers:requesting-code-review.
