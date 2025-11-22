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

**In Progress:**
- None

**Remaining:**
- Task 3: EntitySidebar Widget - Basic Structure
- Task 4: EntitySidebar - Loading State Display
- Task 5: EntitySidebar - Entity List Display
- Task 6: EntitySidebar - Fetch Entities from Database
- Task 7: ViewScreen - Add EntitySidebar Integration
- Task 8: ViewScreen - Job Status Polling
- Task 9: Navigation Flow - ESC to ViewScreen
- Task 10: Entity Deletion - Confirmation Modal
- Task 11: Entity Deletion - Database Integration
- Task 12: Update Existing ViewScreen Call Sites
- Task 13: Integration Test - Full Workflow
- Task 14: Documentation and Cleanup
- Task 15: Final Verification

---

## Task 1: Database Query - Fetch Entities for Episode

**Files:**
- Test: `tests/test_backend/test_queries.py` (new file)
- Implement: `backend/database/queries.py`

### Step 1: Write failing test for fetch_entities_for_episode

Create `tests/test_backend/test_queries.py`:

```python
"""Tests for database query functions."""

import pytest
from backend.database.queries import fetch_entities_for_episode
from backend.database.utils import SELF_ENTITY_UUID
from backend import add_journal_entry
from backend.graph.extract_nodes import extract_nodes


@pytest.mark.asyncio
async def test_fetch_entities_for_episode_returns_entities(episode_factory):
    """Should return all entities mentioned in episode, excluding SELF."""
    # Create episode that will extract entities
    episode_uuid = await add_journal_entry(
        content="I met Sarah at Central Park for coffee.",
        journal="test_journal"
    )

    # Extract entities (this creates MENTIONS edges)
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    # Fetch entities
    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")

    # Should have entities but not SELF
    assert len(entities) > 0
    entity_names = [e["name"] for e in entities]
    assert "Sarah" in entity_names or "Central Park" in entity_names

    # SELF should be filtered out
    entity_uuids = [e["uuid"] for e in entities]
    assert str(SELF_ENTITY_UUID) not in entity_uuids


@pytest.mark.asyncio
async def test_fetch_entities_for_episode_empty(episode_factory):
    """Should return empty list when episode has no entity mentions."""
    episode_uuid = await add_journal_entry(
        content="Just some text.",
        journal="test_journal"
    )

    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")

    assert entities == []
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_backend/test_queries.py::test_fetch_entities_for_episode_returns_entities -v`

Expected: FAIL with "ImportError: cannot import name 'fetch_entities_for_episode'"

### Step 3: Implement fetch_entities_for_episode

In `backend/database/queries.py`, add at the end:

```python
async def fetch_entities_for_episode(
    episode_uuid: str, journal: str = DEFAULT_JOURNAL
) -> list[dict]:
    """Fetch all entities mentioned in a specific episode, excluding SELF.

    Args:
        episode_uuid: Episode UUID to query
        journal: Journal name

    Returns:
        List of entity dicts with keys: uuid, name, labels, ref_count
        where ref_count is total mentions across all episodes
    """
    from backend.database.driver import get_driver
    from backend.database.utils import SELF_ENTITY_UUID, to_cypher_literal

    driver = get_driver(journal)

    episode_literal = to_cypher_literal(episode_uuid)
    self_literal = to_cypher_literal(str(SELF_ENTITY_UUID))

    query = f"""
    MATCH (ep:Episodic {{uuid: {episode_literal}}})-[:MENTIONS]->(e:Entity)
    WHERE e.uuid <> {self_literal}
    WITH e
    OPTIONAL MATCH (e)<-[r:MENTIONS]-()
    RETURN e.uuid as uuid, e.name as name, e.labels as labels, count(r) as ref_count
    ORDER BY e.name
    """

    result = await driver.graph.query(query)

    entities = []
    for record in result.result_set:
        entities.append({
            "uuid": record[0],
            "name": record[1],
            "labels": record[2] if record[2] else ["Entity"],
            "ref_count": record[3] if record[3] else 0,
        })

    return entities
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_backend/test_queries.py -v`

Expected: PASS (both tests)

### Step 5: Commit

```bash
git add tests/test_backend/test_queries.py backend/database/queries.py
git commit -m "feat: add fetch_entities_for_episode query

- Query entities mentioned in episode via MENTIONS edges
- Filter out SELF entity
- Include ref_count across all episodes
- Tests cover happy path and empty case"
```

---

## Task 2: Database Mutation - Delete Entity Mention

**Files:**
- Test: `tests/test_backend/test_queries.py`
- Implement: `backend/database/queries.py`

### Step 1: Write failing test for delete_entity_mention

Add to `tests/test_backend/test_queries.py`:

```python
from backend.database.queries import delete_entity_mention


@pytest.mark.asyncio
async def test_delete_entity_mention_orphaned(episode_factory):
    """Should delete entity when it's only mentioned in this episode."""
    # Create episode with entity
    episode_uuid = await add_journal_entry(
        content="I met Sarah at the park.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=episode_uuid, journal="test_journal")

    # Get entity UUID
    entities = await fetch_entities_for_episode(episode_uuid, "test_journal")
    assert len(entities) > 0
    entity_uuid = entities[0]["uuid"]

    # Delete mention
    was_deleted = await delete_entity_mention(episode_uuid, entity_uuid, "test_journal")

    # Entity should be fully deleted (was orphaned)
    assert was_deleted is True

    # Verify entity no longer appears
    entities_after = await fetch_entities_for_episode(episode_uuid, "test_journal")
    assert entity_uuid not in [e["uuid"] for e in entities_after]


@pytest.mark.asyncio
async def test_delete_entity_mention_shared(episode_factory):
    """Should only remove MENTIONS edge when entity referenced elsewhere."""
    # Create two episodes mentioning same entity
    ep1_uuid = await add_journal_entry(
        content="I met Sarah today.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep1_uuid, journal="test_journal")

    ep2_uuid = await add_journal_entry(
        content="Sarah came over again.",
        journal="test_journal"
    )
    await extract_nodes(episode_uuid=ep2_uuid, journal="test_journal")

    # Get Sarah's UUID from first episode
    entities_ep1 = await fetch_entities_for_episode(ep1_uuid, "test_journal")
    sarah = next((e for e in entities_ep1 if "Sarah" in e["name"]), None)
    assert sarah is not None
    assert sarah["ref_count"] >= 2  # Mentioned in at least 2 episodes

    # Delete from first episode only
    was_deleted = await delete_entity_mention(ep1_uuid, sarah["uuid"], "test_journal")

    # Entity should NOT be fully deleted (still referenced)
    assert was_deleted is False

    # Should not appear in ep1 anymore
    entities_ep1_after = await fetch_entities_for_episode(ep1_uuid, "test_journal")
    assert sarah["uuid"] not in [e["uuid"] for e in entities_ep1_after]

    # Should still appear in ep2
    entities_ep2 = await fetch_entities_for_episode(ep2_uuid, "test_journal")
    assert sarah["uuid"] in [e["uuid"] for e in entities_ep2]
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_backend/test_queries.py::test_delete_entity_mention_orphaned -v`

Expected: FAIL with "ImportError: cannot import name 'delete_entity_mention'"

### Step 3: Implement delete_entity_mention

Add to `backend/database/queries.py`:

```python
async def delete_entity_mention(
    episode_uuid: str, entity_uuid: str, journal: str = DEFAULT_JOURNAL
) -> bool:
    """Delete MENTIONS edge between episode and entity, and entity if orphaned.

    Args:
        episode_uuid: Episode UUID
        entity_uuid: Entity UUID to remove mention of
        journal: Journal name

    Returns:
        True if entity was fully deleted (orphaned), False if only edge removed
    """
    from backend.database.driver import get_driver
    from backend.database.utils import to_cypher_literal

    driver = get_driver(journal)

    episode_literal = to_cypher_literal(episode_uuid)
    entity_literal = to_cypher_literal(entity_uuid)

    # Delete the MENTIONS edge, then delete entity if orphaned
    query = f"""
    MATCH (ep:Episodic {{uuid: {episode_literal}}})-[r:MENTIONS]->(ent:Entity {{uuid: {entity_literal}}})
    DELETE r
    WITH ent
    OPTIONAL MATCH (ent)<-[remaining:MENTIONS]-()
    WITH ent, count(remaining) as remaining_refs
    WHERE remaining_refs = 0
    DETACH DELETE ent
    RETURN remaining_refs = 0 as was_deleted
    """

    result = await driver.graph.query(query)

    if result.result_set:
        return bool(result.result_set[0][0])
    return False
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_backend/test_queries.py -v`

Expected: PASS (all 4 tests)

### Step 5: Commit

```bash
git add tests/test_backend/test_queries.py backend/database/queries.py
git commit -m "feat: add delete_entity_mention mutation

- Delete MENTIONS edge between episode and entity
- Auto-delete entity if no remaining references
- Return boolean indicating full deletion
- Tests cover orphaned and shared entity cases"
```

---

## Task 3: EntitySidebar Widget - Basic Structure

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py` (new file)
- Implement: `charlie.py`

### Step 1: Write failing test for EntitySidebar structure

Create `tests/test_frontend/test_entity_sidebar.py`:

```python
"""Tests for EntitySidebar widget."""

import pytest
from textual.widgets import Label, ListView
from charlie import EntitySidebar


@pytest.mark.asyncio
async def test_entity_sidebar_shows_loading_initially():
    """Should show loading state on mount."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test():
        # Should be in loading state
        assert sidebar.loading is True


@pytest.mark.asyncio
async def test_entity_sidebar_has_header():
    """Should have 'Connections' header."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test():
        labels = sidebar.query("Label")
        header_texts = [label.renderable for label in labels]
        assert "Connections" in header_texts
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_shows_loading_initially -v`

Expected: FAIL with "ImportError: cannot import name 'EntitySidebar'"

### Step 3: Implement EntitySidebar basic structure

In `charlie.py`, add after imports and before the `extract_title` function:

```python
from textual.containers import Container, Horizontal
from textual.reactive import reactive


class EntitySidebar(Container):
    """Sidebar showing entities connected to current episode."""

    DEFAULT_CSS = """
    EntitySidebar {
        width: 1fr;
        border-left: solid $accent;
        padding: 1;
    }

    EntitySidebar .sidebar-header {
        text-style: bold;
        margin-bottom: 1;
    }

    EntitySidebar .sidebar-footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    episode_uuid: reactive[str] = reactive("")
    journal: reactive[str] = reactive("")
    loading: reactive[bool] = reactive(True)
    entities: reactive[list[dict]] = reactive([])

    def __init__(self, episode_uuid: str, journal: str, **kwargs):
        super().__init__(**kwargs)
        self.episode_uuid = episode_uuid
        self.journal = journal

    def compose(self) -> ComposeResult:
        yield Label("Connections", classes="sidebar-header")
        yield Container(id="entity-content")
        yield Label("d: delete | ↑↓: navigate | c: close", classes="sidebar-footer")
```

Update imports at top of file to include:
```python
from textual.containers import Container, Horizontal
from textual.reactive import reactive
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (both tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: add EntitySidebar widget structure

- Container with header, content area, footer
- Reactive properties for episode, loading, entities
- Basic CSS styling for sidebar layout
- Tests verify structure and initial state"
```

---

## Task 4: EntitySidebar - Loading State Display

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py`
- Implement: `charlie.py`

### Step 1: Write failing test for loading state rendering

Add to `tests/test_frontend/test_entity_sidebar.py`:

```python
@pytest.mark.asyncio
async def test_entity_sidebar_shows_cat_spinner_when_loading():
    """Should show ASCII cat spinner and 'Slinging yarn...' when loading."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test():
        sidebar.loading = True
        await sidebar._render_content()

        content = sidebar.query_one("#entity-content")
        labels = content.query("Label")

        # Should have loading message
        assert any("Slinging yarn" in str(label.renderable) for label in labels)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_shows_cat_spinner_when_loading -v`

Expected: FAIL with "AttributeError: 'EntitySidebar' object has no attribute '_render_content'"

### Step 3: Implement loading state rendering

In `charlie.py`, add to `EntitySidebar` class:

```python
    CAT_SPINNER_FRAMES = [">^.^<", "^.^", ">^.^<", "^.^"]

    def __init__(self, episode_uuid: str, journal: str, **kwargs):
        super().__init__(**kwargs)
        self.episode_uuid = episode_uuid
        self.journal = journal
        self._spinner_index = 0
        self._spinner_timer = None

    def on_mount(self) -> None:
        """Start spinner animation when mounted."""
        if self.loading:
            self._spinner_timer = self.set_interval(0.3, self._update_spinner)
        self._render_content()

    def watch_loading(self, loading: bool) -> None:
        """Reactive: swap between loading indicator and entity list."""
        if loading and self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.3, self._update_spinner)
        elif not loading and self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

        self._render_content()

    def _update_spinner(self) -> None:
        """Update spinner animation frame."""
        self._spinner_index = (self._spinner_index + 1) % len(self.CAT_SPINNER_FRAMES)
        if self.loading:
            self._render_content()

    def _render_content(self) -> None:
        """Render either loading state or entity list."""
        content_container = self.query_one("#entity-content", Container)
        content_container.remove_children()

        if self.loading:
            cat_frame = self.CAT_SPINNER_FRAMES[self._spinner_index]
            content_container.mount(Label(f"{cat_frame} Slinging yarn..."))
        else:
            # Entity list will be rendered here later
            content_container.mount(Label("No connections found"))
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: add loading state with ASCII cat spinner

- Animated ASCII cat looking left/right
- 'Slinging yarn...' loading message
- Spinner updates every 300ms
- Reactive watch on loading state
- Tests verify spinner display"
```

---

## Task 5: EntitySidebar - Entity List Display

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py`
- Implement: `charlie.py`

### Step 1: Write failing test for entity list rendering

Add to `tests/test_frontend/test_entity_sidebar.py`:

```python
@pytest.mark.asyncio
async def test_entity_sidebar_displays_entities():
    """Should display entities in ListView when loaded."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test():
        # Set entities
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
            {"uuid": "uuid-2", "name": "Central Park", "labels": ["Entity", "Place"], "ref_count": 1},
        ]
        sidebar.loading = False

        # Should have ListView
        list_view = sidebar.query_one(ListView)
        assert list_view is not None

        # Should have 2 items
        items = list(list_view.children)
        assert len(items) == 2


@pytest.mark.asyncio
async def test_entity_sidebar_formats_entity_labels():
    """Should format entities as 'Name [Type] (RefCount)'."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test():
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
            {"uuid": "uuid-2", "name": "Park", "labels": ["Entity"], "ref_count": 1},
        ]
        sidebar.loading = False

        list_view = sidebar.query_one(ListView)
        items = list(list_view.children)

        # First item: show most specific type (Person, not Entity)
        assert "Sarah" in str(items[0])
        assert "[Person]" in str(items[0])
        assert "(3)" in str(items[0])

        # Second item: show Entity when it's the only type
        assert "Park" in str(items[1])
        assert "[Entity]" in str(items[1])
        assert "(1)" in str(items[1])
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_displays_entities -v`

Expected: FAIL with "LookupError: No nodes match"

### Step 3: Implement entity list rendering

In `charlie.py`, update `_render_content` method in `EntitySidebar`:

```python
    def watch_entities(self, entities: list[dict]) -> None:
        """Reactive: re-render when entities change."""
        if not self.loading:
            self._render_content()

    def _render_content(self) -> None:
        """Render either loading state or entity list."""
        content_container = self.query_one("#entity-content", Container)
        content_container.remove_children()

        if self.loading:
            cat_frame = self.CAT_SPINNER_FRAMES[self._spinner_index]
            content_container.mount(Label(f"{cat_frame} Slinging yarn..."))
        elif not self.entities:
            content_container.mount(Label("No connections found"))
        else:
            list_view = ListView()
            for entity in self.entities:
                formatted_label = self._format_entity_label(entity)
                list_view.append(ListItem(Label(formatted_label)))
            content_container.mount(list_view)

    def _format_entity_label(self, entity: dict) -> str:
        """Format entity as 'Name [Type] (RefCount)'."""
        name = entity["name"]
        labels = entity["labels"]
        ref_count = entity["ref_count"]

        # Filter out "Entity" if there's a more specific type
        specific_labels = [l for l in labels if l != "Entity"]
        entity_type = specific_labels[0] if specific_labels else "Entity"

        return f"{name} [{entity_type}] ({ref_count})"
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (all 5 tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: display entity list in sidebar

- ListView with formatted entity items
- Format: 'Name [Type] (RefCount)'
- Filter Entity label when more specific type exists
- Empty state shows 'No connections found'
- Tests verify formatting and display"
```

---

## Task 6: EntitySidebar - Fetch Entities from Database

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py`
- Implement: `charlie.py`

### Step 1: Write failing test for refresh_entities

Add to `tests/test_frontend/test_entity_sidebar.py`:

```python
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_entity_sidebar_refresh_entities():
    """Should fetch entities from database and update state."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    mock_entities = [
        {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 2},
    ]

    with patch("charlie.fetch_entities_for_episode", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_entities

        async with sidebar.app.run_test():
            await sidebar.refresh_entities()

            # Should have called fetch with correct args
            mock_fetch.assert_called_once_with("test-uuid", "test")

            # Should update state
            assert sidebar.loading is False
            assert sidebar.entities == mock_entities
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_refresh_entities -v`

Expected: FAIL with "AttributeError: 'EntitySidebar' object has no attribute 'refresh_entities'"

### Step 3: Implement refresh_entities

In `charlie.py`, add import at top:
```python
from backend.database.queries import fetch_entities_for_episode
```

Add to `EntitySidebar` class:

```python
    async def refresh_entities(self) -> None:
        """Fetch and display entities for current episode."""
        try:
            raw_entities = await fetch_entities_for_episode(self.episode_uuid, self.journal)
            self.entities = raw_entities
            self.loading = False
        except Exception as e:
            logger.error(f"Failed to fetch entities: {e}", exc_info=True)
            self.loading = False
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (all 6 tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: implement refresh_entities method

- Fetch entities from database via query function
- Update entities and loading state
- Error handling with logging
- Test verifies database call and state update"
```

---

## Task 7: ViewScreen - Add EntitySidebar Integration

**Files:**
- Test: `tests/test_frontend/test_view_screen.py` (new file)
- Implement: `charlie.py`

### Step 1: Write failing test for ViewScreen with sidebar

Create `tests/test_frontend/test_view_screen.py`:

```python
"""Tests for ViewScreen with entity sidebar."""

import pytest
from unittest.mock import patch, AsyncMock
from textual.widgets import Markdown
from charlie import ViewScreen, EntitySidebar


@pytest.mark.asyncio
async def test_view_screen_has_sidebar():
    """Should contain EntitySidebar in horizontal layout."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        screen = ViewScreen(episode_uuid="test-uuid", journal="test")

        async with screen.app.run_test():
            # Should have EntitySidebar
            sidebar = screen.query_one(EntitySidebar)
            assert sidebar is not None
            assert sidebar.episode_uuid == "test-uuid"
            assert sidebar.journal == "test"


@pytest.mark.asyncio
async def test_view_screen_has_markdown_viewer():
    """Should still have Markdown widget for journal content."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test\nContent",
        }

        screen = ViewScreen(episode_uuid="test-uuid", journal="test")

        async with screen.app.run_test():
            markdown = screen.query_one(Markdown)
            assert markdown is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_view_screen.py::test_view_screen_has_sidebar -v`

Expected: FAIL with "TypeError: ViewScreen.__init__() got an unexpected keyword argument 'journal'"

### Step 3: Modify ViewScreen to include sidebar

In `charlie.py`, update `ViewScreen` class:

```python
class ViewScreen(Screen):
    """Screen for viewing a journal entry in read-only mode with entity sidebar.

    WARNING: Do NOT use recompose() in this screen - Markdown widget
    has internal state (scroll position) that would be lost.
    """

    BINDINGS = [
        ("e", "edit_entry", "Edit", show=True),
        ("c", "toggle_connections", "Connections", show=True),
        ("q", "back", "Back", show=True),
        ("escape", "back", "Back", show=False),
        ("space", "back", "Back", show=False),
        ("enter", "back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    ViewScreen Horizontal {
        height: 100%;
    }

    ViewScreen #journal-content {
        width: 3fr;
        padding: 1 2;
    }
    """

    def __init__(self, episode_uuid: str, journal: str = DEFAULT_JOURNAL):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.episode = None
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield Horizontal(
            Markdown("Loading...", id="journal-content"),
            EntitySidebar(
                episode_uuid=self.episode_uuid,
                journal=self.journal,
                id="entity-sidebar"
            ),
        )
        yield Footer()

    def action_toggle_connections(self) -> None:
        """Toggle sidebar visibility."""
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        sidebar.display = not sidebar.display
```

Update imports to include DEFAULT_JOURNAL:
```python
from backend.settings import DEFAULT_JOURNAL
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_frontend/test_view_screen.py -v`

Expected: PASS (both tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_view_screen.py charlie.py
git commit -m "feat: integrate EntitySidebar into ViewScreen

- Horizontal layout with Markdown (3fr) + EntitySidebar (1fr)
- Accept journal parameter in ViewScreen.__init__
- Add 'c' key binding to toggle sidebar visibility
- CSS for layout proportions
- Tests verify sidebar presence and layout"
```

---

## Task 8: ViewScreen - Job Status Polling

**Files:**
- Test: `tests/test_frontend/test_view_screen.py`
- Implement: `charlie.py`

### Step 1: Write failing test for job polling

Add to `tests/test_frontend/test_view_screen.py`:

```python
from backend.database.redis_ops import set_episode_status


@pytest.mark.asyncio
async def test_view_screen_polls_job_status():
    """Should poll for job completion and refresh entities."""
    with patch("charlie.get_episode", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "uuid": "test-uuid",
            "content": "# Test",
        }

        screen = ViewScreen(episode_uuid="test-uuid", journal="test")

        async with screen.app.run_test():
            # Set job to pending initially
            set_episode_status("test-uuid", "pending_nodes")

            # Poll timer should be running
            assert screen._poll_timer is not None

            # Complete the job
            set_episode_status("test-uuid", "completed")

            # Wait for poll to detect completion
            await asyncio.sleep(0.6)  # Longer than poll interval

            # Timer should be stopped
            # (Note: Can't directly assert timer.stop() was called, but can check side effects)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_view_screen.py::test_view_screen_polls_job_status -v`

Expected: FAIL with "AssertionError: assert None is not None"

### Step 3: Implement job status polling

In `charlie.py`, add import:
```python
from backend.database.redis_ops import get_episode_status
```

Update `ViewScreen.on_mount`:

```python
    async def on_mount(self):
        await self.load_episode()
        # Start polling for extraction job completion
        self._poll_timer = self.set_interval(0.5, self._check_job_status)

    async def _check_job_status(self) -> None:
        """Poll Huey for job completion, then refresh entities."""
        status = get_episode_status(self.episode_uuid)

        # Job is complete when status is "completed" or None (old episodes)
        if status in ("completed", None):
            if self._poll_timer:
                self._poll_timer.stop()
                self._poll_timer = None

            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            await sidebar.refresh_entities()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_view_screen.py -v`

Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_view_screen.py charlie.py
git commit -m "feat: poll job status and refresh entities

- Start polling on mount (500ms interval)
- Check episode status via Redis
- Stop polling when job complete
- Trigger sidebar refresh when ready
- Test verifies polling lifecycle"
```

---

## Task 9: Navigation Flow - ESC to ViewScreen

**Files:**
- Test: `tests/test_frontend/test_charlie.py`
- Implement: `charlie.py`

### Step 1: Write failing test for ESC navigation from EditScreen

Add to `tests/test_frontend/test_charlie.py`:

```python
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_edit_screen_esc_goes_to_view_screen():
    """ESC from EditScreen should navigate to ViewScreen, not HomeScreen."""
    from charlie import EditScreen, ViewScreen

    with patch("charlie.add_journal_entry", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = "new-uuid"

        with patch("charlie.update_episode", new_callable=AsyncMock):
            screen = EditScreen(episode_uuid=None)  # New entry

            async with screen.app.run_test() as pilot:
                # Type some content
                editor = screen.query_one("#editor")
                editor.text = "# Test Entry\nSome content"

                # Press ESC
                await pilot.press("escape")

                # Should navigate to ViewScreen, not pop to HomeScreen
                assert isinstance(screen.app.screen, ViewScreen)
                assert screen.app.screen.episode_uuid == "new-uuid"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_charlie.py::test_edit_screen_esc_goes_to_view_screen -v`

Expected: FAIL with "AssertionError: assert False" (goes to wrong screen)

### Step 3: Modify EditScreen to navigate to ViewScreen

In `charlie.py`, update `EditScreen.save_entry`:

```python
    async def save_entry(self):
        try:
            editor = self.query_one("#editor", TextArea)
            content = editor.text

            if not content.strip():
                self.app.pop_screen()
                return

            title = extract_title(content)

            if self.is_new_entry:
                uuid = await add_journal_entry(content=content)
                if title:
                    await update_episode(uuid, name=title)
                # Navigate to ViewScreen instead of popping
                self.app.pop_screen()  # Pop EditScreen
                self.app.push_screen(ViewScreen(uuid, DEFAULT_JOURNAL))
            else:
                if title:
                    await update_episode(self.episode_uuid, content=content, name=title)
                else:
                    await update_episode(self.episode_uuid, content=content)
                # Navigate to ViewScreen for existing entries too
                self.app.pop_screen()  # Pop EditScreen
                self.app.push_screen(ViewScreen(self.episode_uuid, DEFAULT_JOURNAL))

        except Exception as e:
            logger.error(f"Failed to save entry: {e}", exc_info=True)
            self.notify("Failed to save entry", severity="error")
            raise
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_charlie.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add tests/test_frontend/test_charlie.py charlie.py
git commit -m "feat: navigate to ViewScreen after editing

- ESC from EditScreen now goes to ViewScreen
- Shows journal content with entity sidebar
- Works for both new and existing entries
- Test verifies navigation flow"
```

---

## Task 10: Entity Deletion - Confirmation Modal

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py`
- Implement: `charlie.py`

### Step 1: Write failing test for deletion confirmation

Add to `tests/test_frontend/test_entity_sidebar.py`:

```python
from textual.screen import ModalScreen


@pytest.mark.asyncio
async def test_entity_sidebar_shows_delete_confirmation():
    """Pressing 'd' on entity should show confirmation modal."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    async with sidebar.app.run_test() as pilot:
        sidebar.entities = [
            {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
        ]
        sidebar.loading = False

        # Focus list and select first item
        list_view = sidebar.query_one(ListView)
        list_view.focus()

        # Press 'd' for delete
        await pilot.press("d")

        # Should show confirmation modal
        modal = sidebar.app.screen
        assert isinstance(modal, ModalScreen)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_shows_delete_confirmation -v`

Expected: FAIL with "AssertionError: assert False"

### Step 3: Implement delete confirmation modal

In `charlie.py`, add new modal class before `EntitySidebar`:

```python
from textual.widgets import Button


class DeleteEntityModal(ModalScreen):
    """Confirmation modal for entity deletion."""

    DEFAULT_CSS = """
    DeleteEntityModal {
        align: center middle;
    }

    #delete-dialog {
        width: 60;
        height: auto;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }

    #delete-dialog Button {
        margin: 1 2 0 0;
    }
    """

    def __init__(self, entity: dict):
        super().__init__()
        self.entity = entity

    def compose(self) -> ComposeResult:
        name = self.entity["name"]
        ref_count = self.entity["ref_count"]

        message = f"Remove {name} from this entry?"
        if ref_count > 1:
            hint = f"(Will remain in {ref_count - 1} other entries)"
        else:
            hint = "(Will be removed entirely)"

        yield Vertical(
            Label("Remove Connection?", id="delete-title"),
            Label(message),
            Label(hint),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Remove", id="remove", variant="error"),
            ),
            id="delete-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "remove":
            self.dismiss(True)
        else:
            self.dismiss(False)
```

Update imports to include Button and Vertical:
```python
from textual.widgets import Button, Vertical
```

Add to `EntitySidebar` class:

```python
    BINDINGS = [
        ("d", "delete_entity", "Delete", show=False),
    ]

    def action_delete_entity(self) -> None:
        """Show delete confirmation for selected entity."""
        list_view = self.query_one(ListView)
        if list_view.index is None or list_view.index < 0:
            return

        entity = self.entities[list_view.index]
        self.app.push_screen(DeleteEntityModal(entity), self._handle_delete_result)

    async def _handle_delete_result(self, confirmed: bool) -> None:
        """Handle deletion confirmation result."""
        if confirmed:
            # Deletion logic will be implemented next
            pass
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (all tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: add entity deletion confirmation modal

- Modal shows entity name and ref count hint
- Cancel/Remove buttons with appropriate styling
- 'd' key binding on EntitySidebar
- Test verifies modal appears on delete action"
```

---

## Task 11: Entity Deletion - Database Integration

**Files:**
- Test: `tests/test_frontend/test_entity_sidebar.py`
- Implement: `charlie.py`

### Step 1: Write failing test for actual deletion

Add to `tests/test_frontend/test_entity_sidebar.py`:

```python
@pytest.mark.asyncio
async def test_entity_sidebar_deletes_entity():
    """Confirming deletion should remove entity from list."""
    sidebar = EntitySidebar(episode_uuid="test-uuid", journal="test")

    with patch("charlie.delete_entity_mention", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = False  # Not fully deleted

        async with sidebar.app.run_test() as pilot:
            sidebar.entities = [
                {"uuid": "uuid-1", "name": "Sarah", "labels": ["Entity", "Person"], "ref_count": 3},
                {"uuid": "uuid-2", "name": "Park", "labels": ["Entity", "Place"], "ref_count": 1},
            ]
            sidebar.loading = False

            list_view = sidebar.query_one(ListView)
            list_view.focus()
            list_view.index = 0  # Select Sarah

            # Press 'd' and confirm
            await pilot.press("d")
            modal = sidebar.app.screen
            remove_button = modal.query_one("#remove", Button)
            remove_button.press()

            await asyncio.sleep(0.1)  # Let deletion process

            # Should have called delete
            mock_delete.assert_called_once_with("test-uuid", "uuid-1", "test")

            # Sarah should be removed from list
            assert len(sidebar.entities) == 1
            assert sidebar.entities[0]["name"] == "Park"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_frontend/test_entity_sidebar.py::test_entity_sidebar_deletes_entity -v`

Expected: FAIL with "AssertionError: assert 2 == 1"

### Step 3: Implement deletion logic

In `charlie.py`, add import:
```python
from backend.database.queries import delete_entity_mention
```

Update `_handle_delete_result` in `EntitySidebar`:

```python
    async def _handle_delete_result(self, confirmed: bool) -> None:
        """Handle deletion confirmation result."""
        if not confirmed:
            return

        list_view = self.query_one(ListView)
        if list_view.index is None or list_view.index < 0:
            return

        entity = self.entities[list_view.index]

        try:
            # Delete from database
            await delete_entity_mention(
                self.episode_uuid,
                entity["uuid"],
                self.journal
            )

            # Remove from local state
            new_entities = [e for e in self.entities if e["uuid"] != entity["uuid"]]
            self.entities = new_entities

        except Exception as e:
            logger.error(f"Failed to delete entity mention: {e}", exc_info=True)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_frontend/test_entity_sidebar.py -v`

Expected: PASS (all tests)

### Step 5: Commit

```bash
git add tests/test_frontend/test_entity_sidebar.py charlie.py
git commit -m "feat: implement entity deletion

- Call delete_entity_mention on confirmation
- Remove entity from local state
- Error handling with logging
- Test verifies database call and state update"
```

---

## Task 12: Update Existing ViewScreen Call Sites

**Files:**
- Implement: `charlie.py`

### Step 1: Find all ViewScreen instantiations

Run: `grep -n "ViewScreen(" charlie.py`

Expected output showing line numbers where ViewScreen is created

### Step 2: Update HomeScreen to pass journal parameter

In `charlie.py`, update `HomeScreen.action_view_entry`:

```python
    def action_view_entry(self):
        list_view = self.query_one("#entries-list", ListView)
        if list_view.index is not None and list_view.index >= 0:
            episode = self.episodes[list_view.index]
            self.app.push_screen(ViewScreen(episode["uuid"], DEFAULT_JOURNAL))
```

### Step 3: Verify no other call sites need updates

Check EditScreen already updated in Task 9.

### Step 4: Run all tests to verify nothing broke

Run: `pytest tests/ -v`

Expected: PASS (all tests)

### Step 5: Commit

```bash
git add charlie.py
git commit -m "fix: update ViewScreen call sites with journal param

- HomeScreen.action_view_entry passes DEFAULT_JOURNAL
- EditScreen already updated in previous task
- All ViewScreen instantiations now consistent"
```

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
                            mock_status.return_value = "completed"

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
