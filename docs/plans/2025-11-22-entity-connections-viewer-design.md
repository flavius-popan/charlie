# Entity Connections Viewer Design

**Date:** 2025-11-22
**Status:** Approved
**Context:** Per-journal diagnostic view for extracted entities with deletion capabilities

## Overview

Add a "Connections" sidebar to the ViewScreen that displays entities extracted from the current journal entry. This is a diagnostic view showing what's attached to a specific episode node, allowing users to view and delete entity mentions.

## User Requirements

- View entities extracted from a journal entry
- Delete entity mentions from the current entry (local deletion)
- See real-time updates as extraction completes
- No obstruction to markdown reading experience
- Playful, end-user friendly interface (quirky cat theme)

## Component Architecture

### ViewScreen Modifications

The current `ViewScreen` displays journal markdown in a full-width `Markdown` widget. Modified to contain:

1. **Horizontal Container** (`Horizontal` widget from Textual)
   - Left: `Markdown` widget (75% width)
   - Right: `EntitySidebar` custom widget (25% width, visible by default)

2. **New EntitySidebar Widget** (custom composite widget)
   - Header: "Connections" title
   - Body: Either loading state OR entity list
   - Loading state: ASCII cat spinner + "Slinging yarn..."
     - Spinner frames: `['>^.^<', '^.^', '>^.^<', '^.^']` (cat looking left/center/right)
   - Loaded state: `ListView` containing entity items
   - Footer: Action hints (delete key bindings)

### Widget Hierarchy

```
ViewScreen
├── Horizontal
│   ├── Markdown (journal content)
│   └── EntitySidebar (custom widget)
│       ├── Label (header "Connections")
│       ├── [Loading | ListView] (conditional rendering)
│       └── Label (footer with key hints)
```

### Key Relationships

- `ViewScreen` owns the episode UUID and journal name
- `ViewScreen` manages the polling timer (reactive var or set_interval)
- `EntitySidebar` receives episode UUID as parameter, fetches entities
- `EntitySidebar` exposes method `refresh_entities()` called after job completes

## Data Flow & Reactive Updates

### On ViewScreen Mount (after ESC from edit)

1. ViewScreen receives episode UUID and journal name from EditScreen
2. Start polling for extract_nodes job status (every 500ms)
3. EntitySidebar shows loading state: `">^.^< Slinging yarn..."`
4. Poll checks Huey queue for job completion using episode UUID as key

### When Job Completes

5. Stop polling timer
6. Call new database function: `fetch_entities_for_episode(episode_uuid, journal)`
7. Filter out SELF entity from results
8. Transform entities to display format: `{name} [{type}]` or `{name} [{type}] ({ref_count})` if > 1
   - Type filtering: Show most specific label (e.g., "Person" not "Entity, Person")
   - Only show "Entity" if it's the only label
   - Ref count format: "(4)" for 4 mentions - only shown when ref_count > 1 to reduce clutter
9. Populate ListView with entity items
10. Footer shows: "d: delete | ↑↓: navigate | c: close"

### Database Query Function (new)

```python
async def fetch_entities_for_episode(episode_uuid: str, journal: str) -> list[dict]:
    # Cypher: MATCH (ep:Episodic {uuid: $uuid})-[:MENTIONS]->(e:Entity)
    # WHERE e.uuid != $SELF_UUID
    # RETURN e.uuid, e.name, e.labels, count mentions across all episodes
```

### Reactive State

- Use Textual reactive variable to track loading/loaded state
- When state changes from loading → loaded, swap Loading widget for ListView
- No manual refresh needed - state change triggers UI update automatically

## User Interactions & Deletion Flow

### Navigation & Toggling

- **ESC from EditScreen** → Navigate to ViewScreen with sidebar open by default
- **`c` key** (connections) → Toggle sidebar visibility (collapse to full-width markdown / expand to split view)
- **`q` or ESC from ViewScreen** → Return to HomeScreen
- **Arrow keys (↑↓)** → Navigate entity list when sidebar has focus
- **Tab** → Switch focus between markdown viewer and entity list

### Entity Selection & Deletion

When an entity is highlighted in the ListView:

1. **`d` key** → Trigger deletion confirmation
2. **Confirmation modal appears:**
   - Title: "Remove Connection?"
   - Message: `"Remove Sarah from this entry?"`
   - If ref_count > 1: `"(Will remain in {ref_count - 1} other entries)"`
   - If ref_count == 1: `"(Will be removed entirely)"`
   - Buttons: `[Cancel] [Remove]` (Cancel focused by default)
3. **On confirm:**
   - Delete MENTIONS edge: `(episode)-[:MENTIONS]->(entity)`
   - If entity has no remaining MENTIONS edges → delete entity node
   - Remove item from ListView (no toast notification)

### Multi-select (future enhancement)

- Could use `Space` to multi-select entities
- `d` deletes all selected
- But start with single-delete for simplicity

### Empty States

- No entities extracted: Show message `"No connections found"` in sidebar
- Job failed: Show `"Extraction failed"` with option to retry

## Design Principles & Implementation Notes

### UI Philosophy

- **No toasts/notifications** - They're annoying and disrupt flow. UI updates (like removing an item from the list) are sufficient feedback.
- **Playful, end-user language** - Avoid technical jargon ("knowledge graph", "nodes", "edges"). Use natural terms ("connections", "entries", "mentioned in").
- **Quirky cat theme** - Subtle personality (ASCII cat, "slinging yarn") without being overwhelming.

## Implementation Details

### New Database Functions (`backend/database/queries.py`)

```python
async def fetch_entities_for_episode(episode_uuid: str, journal: str) -> list[dict]:
    """Fetch all entities mentioned in a specific episode, excluding SELF."""
    # Returns: [
    #   {
    #     'uuid': str,
    #     'name': str,
    #     'labels': list[str],  # e.g., ['Entity', 'Person']
    #     'ref_count': int      # total mentions across all episodes
    #   },
    #   ...
    # ]

async def delete_entity_mention(episode_uuid: str, entity_uuid: str) -> bool:
    """Delete MENTIONS edge and entity if orphaned. Returns True if entity deleted."""
    # 1. DELETE (ep)-[r:MENTIONS]->(ent) WHERE ep.uuid = $episode_uuid AND ent.uuid = $entity_uuid
    # 2. If entity has no remaining MENTIONS edges, DETACH DELETE entity
    # 3. Return whether entity was fully deleted (for confirmation message)
```

### New Widget Class (`charlie.py`)

```python
class EntitySidebar(Container):
    """Sidebar showing entities connected to current episode."""

    episode_uuid: reactive[str] = reactive("")
    journal: reactive[str] = reactive("")
    loading: reactive[bool] = reactive(True)
    entities: reactive[list[dict]] = reactive([])

    def compose(self) -> ComposeResult:
        yield Label("Connections", classes="sidebar-header")
        yield Container(id="entity-content")  # Swaps between loading/list
        yield Label("d: delete | ↑↓: navigate | c: close", classes="sidebar-footer")

    def watch_loading(self, loading: bool) -> None:
        """Reactive: swap between loading indicator and entity list."""
        # Clear content container, render appropriate widget

    async def refresh_entities(self) -> None:
        """Fetch and display entities for current episode."""
        raw_entities = await fetch_entities_for_episode(self.episode_uuid, self.journal)
        self.entities = raw_entities
        self.loading = False
        # Note: ListView auto-focuses after rendering via call_after_refresh in _render_content
```

### ViewScreen Modifications

```python
class ViewScreen(Screen):
    BINDINGS = [
        ("c", "toggle_connections", "Connections"),
        ("escape", "go_back", "Back"),
        # ... existing bindings
    ]

    def __init__(self, episode_uuid: str, journal: str):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Markdown(id="journal-content"),  # 3fr
            EntitySidebar(
                episode_uuid=self.episode_uuid,
                journal=self.journal,
                id="entity-sidebar"
            ),  # 1fr
        )

    def on_mount(self) -> None:
        """Start polling for extraction job completion."""
        self._poll_timer = self.set_interval(0.5, self._check_job_status)

    def _check_job_status(self) -> None:
        """Poll Huey for job completion, then refresh entities."""
        status = get_episode_status(self.episode_uuid)

        # Job complete when status is "pending_edges", "done", or None (old episodes)
        if status in ("pending_edges", "done", None):
            if self._poll_timer:
                self._poll_timer.stop()
                self._poll_timer = None

            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            self.run_worker(sidebar.refresh_entities(), exclusive=True)
```

### CSS Styling (proportions)

```css
ViewScreen Horizontal {
    height: 100%;
}

ViewScreen #journal-content {
    width: 3fr;
    padding: 1 2;
}

ViewScreen #entity-sidebar {
    width: 1fr;
    border-left: solid $accent;
}
```

## Test Coverage

### Database Function Tests (`test_queries.py`)

```python
async def test_fetch_entities_for_episode():
    # Setup: Create episode with MENTIONS to Person, Place, Organization
    # Assert: Returns all entities except SELF
    # Assert: ref_count accurate when entity mentioned in multiple episodes
    # Assert: Type filtering (show "Person" not "Entity, Person")

async def test_fetch_entities_for_episode_filters_self():
    # Setup: Episode mentions SELF and Sarah
    # Assert: Only Sarah returned

async def test_fetch_entities_for_episode_empty():
    # Setup: Episode with no MENTIONS edges
    # Assert: Returns empty list

async def test_delete_entity_mention_orphaned():
    # Setup: Entity mentioned only in this episode (ref_count=1)
    # Act: Delete mention
    # Assert: MENTIONS edge deleted AND entity node deleted
    # Assert: Returns True (entity fully deleted)

async def test_delete_entity_mention_shared():
    # Setup: Entity mentioned in 3 episodes (ref_count=3)
    # Act: Delete mention from one episode
    # Assert: MENTIONS edge deleted, entity node remains
    # Assert: Returns False (entity still exists)

async def test_delete_entity_mention_nonexistent():
    # Setup: Invalid episode or entity UUID
    # Assert: Handles gracefully, returns False
```

### Widget Tests (`test_entity_sidebar.py`)

```python
async def test_entity_sidebar_loading_state():
    # Assert: Shows ">^.^< Slinging yarn..." when loading=True
    # Assert: Shows ListView when loading=False

async def test_entity_sidebar_formats_entities():
    # Setup: Raw entity data from database
    # Assert: Formats as "Sarah [Person] (3)"
    # Assert: Filters "Entity" label when more specific type exists

async def test_entity_sidebar_empty_state():
    # Setup: entities = []
    # Assert: Shows "No connections found"

async def test_entity_sidebar_refresh():
    # Mock: fetch_entities_for_episode returns data
    # Act: Call refresh_entities()
    # Assert: loading switches to False
    # Assert: ListView populated with formatted entities

async def test_entity_deletion_confirmation():
    # Act: Press 'd' on highlighted entity
    # Assert: Modal appears with correct message
    # Assert: Shows ref_count hint
    # Act: Confirm deletion
    # Assert: delete_entity_mention called
    # Assert: Entity removed from list
```

### Integration Tests (`test_view_screen.py`)

```python
async def test_view_screen_polls_job_status():
    # Setup: Mount ViewScreen with episode_uuid
    # Assert: Polling timer started (0.5s interval)
    # Mock: Job completes
    # Assert: Polling stops, refresh_entities called

async def test_view_screen_toggle_sidebar():
    # Act: Press 'c' key
    # Assert: Sidebar hidden, markdown full width
    # Act: Press 'c' again
    # Assert: Sidebar visible, markdown 75% width

async def test_view_screen_tab_focus():
    # Act: Press Tab
    # Assert: Focus switches markdown → sidebar
    # Act: Press Tab again
    # Assert: Focus switches sidebar → markdown

async def test_navigation_from_edit_to_view():
    # Setup: EditScreen with new entry
    # Act: Press ESC
    # Assert: ViewScreen mounted with correct episode_uuid
    # Assert: Sidebar visible by default
    # Assert: Polling started
```

### Edge Case Tests

```python
async def test_extraction_job_failure():
    # Mock: Job fails with error
    # Assert: Shows "Extraction failed" message
    # Assert: Polling stops

async def test_concurrent_deletion():
    # Setup: Two ViewScreens viewing same entity
    # Act: Delete from one screen
    # Assert: Other screen handles missing entity gracefully

async def test_entity_with_multiple_types():
    # Setup: Entity with labels ["Entity", "Person", "Organization"]
    # Assert: Shows only first non-Entity type "Person"
```

## Implementation Notes & Bug Fixes

### Bug Fixes Applied (2025-11-22)

1. **Async Callback Issue**
   - Problem: `_check_job_status` was async but `set_interval` doesn't await callbacks
   - Fix: Changed to synchronous function, use `run_worker()` to schedule async `refresh_entities()`

2. **Status Check Mismatch**
   - Problem: Polling checked for "completed" status but tasks set "pending_edges"/"done"
   - Fix: Updated check to `if status in ("pending_edges", "done", None)`

3. **Entity Formatting Issues**
   - Problem: Labels showing as `[[]]`, ref_count always showing `(1)` causing clutter
   - Fix: Handle nested/malformed labels gracefully, only show ref_count if > 1

4. **Missing Keyboard Focus**
   - Problem: Sidebar required mouse to focus for keyboard navigation
   - Fix: Auto-focus ListView after rendering using `call_after_refresh()`

## Future Enhancements

- Multi-select deletion (Space to select, d to delete all selected)
- View entity details (summaries, attributes when implemented)
- View edges between entities
- Global entity management view (separate from per-journal view)
- Retry button for failed extraction jobs
