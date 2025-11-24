# Reactive Architecture Refactor for Entity Deletion Bug

## Problem

The timer-based polling (`_poll_timer`) in ViewScreen creates race conditions when deleting the last entity:
1. User sets `sidebar.entities = []` and `sidebar.status = "done"`
2. Poll timer fires 0.5s later (even after user action completes)
3. `_check_job_status()` reads stale Redis cache: `status = "pending_nodes"`
4. Overwrites sidebar.status back to "pending_nodes"
5. UI shows "Awaiting processing..." instead of "No connections found" (flashing bug)

**Root cause:** Polling has same priority as user actions; no mechanism to prevent override.

## Solution: Event-Driven Reactive Design

Replace timer-based polling with Textual reactive attributes + background worker + watch methods.

### Key Principle from Textual Docs

Textual reactive attributes automatically handle:
- **Watch methods**: Called when reactive attributes change (via `watch_<attribute>` methods)
- **Smart refresh**: Only one refresh per change cycle even if multiple reactives updated
- **Automatic propagation**: Changes to one widget's reactive can be propagated to children via watch methods

## Architecture Overview

```
User deletes last entity
  ↓
EntitySidebar._handle_delete_result() sets:
  • self.user_override = True  (PREVENTS POLLING OVERRIDE)
  • self.status = "done"
  • self.loading = False
  ↓
watch_status() fires → updates UI
  BUT: checks user_override flag and respects local state
  ↓
Background worker _poll_until_complete() still running:
  • Polls Redis every 0.5s
  • Reads status = "pending_nodes" from cache
  • Updates self.status (ViewScreen reactive)
  • But EntitySidebar.watch_status() sees user_override = True
  • And returns early WITHOUT overwriting UI
  ↓
Result: UI stays at "No connections found" ✓
```

## Implementation

### 1. Convert ViewScreen to Reactive Attributes

**File:** `charlie.py` (lines ~730)

Currently ViewScreen uses instance variables:
```python
self.status = status
self.inference_enabled = inference_enabled
self.active_processing = active_processing
```

Change to reactive attributes in class definition (similar to EntitySidebar lines 231-237):
```python
class ViewScreen(Screen):
    # Make these reactive for automatic syncing to sidebar
    status: reactive[str | None] = reactive(None)
    inference_enabled: reactive[bool] = reactive(True)
    active_processing: reactive[bool] = reactive(False)

    def __init__(self, episode_uuid, journal, ..., status=None, ...):
        super().__init__()
        self.episode_uuid = episode_uuid  # NOT reactive (never changes)
        self.journal = journal              # NOT reactive (never changes)
        self.episode = None
        self.from_edit = from_edit
        # Initialize reactives (will trigger watchers after mount)
        self.status = status
        self.inference_enabled = inference_enabled
        self.active_processing = active_processing
```

### 2. Add Watch Methods to ViewScreen (NEW)

**File:** `charlie.py` (after line 830, before `on_mount`)

These auto-sync ViewScreen changes to EntitySidebar:
```python
def watch_status(self, status: str | None) -> None:
    """When ViewScreen.status changes, sync to sidebar."""
    if not self.is_mounted:
        return
    try:
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        sidebar.status = status
    except NoMatches:
        pass

def watch_inference_enabled(self, enabled: bool) -> None:
    """When inference toggle changes, sync to sidebar."""
    if not self.is_mounted:
        return
    try:
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        sidebar.inference_enabled = enabled
    except NoMatches:
        pass

def watch_active_processing(self, active: bool) -> None:
    """When processing state changes, sync to sidebar."""
    if not self.is_mounted:
        return
    try:
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        sidebar.active_processing = active
    except NoMatches:
        pass
```

### 3. Replace Timer Polling with Worker

**File:** `charlie.py` lines 790-830 (on_mount section) and 889-903 (_check_job_status)

**Remove:**
- Line 808: `self.set_interval(0.5, self._check_job_status)`
- Lines 889-903: Delete `_check_job_status()` method entirely

**Replace with:**

In `on_mount()` (after line 812):
```python
# Start status polling worker (not a timer)
if sidebar.loading:
    sidebar.active_processing = True
    self.run_worker(self._poll_until_complete(), exclusive=True, name="status-poll")
```

New method to replace `_check_job_status()`:
```python
async def _poll_until_complete(self) -> None:
    """Poll Redis until extraction completes (non-blocking background task)."""
    while True:
        try:
            # Run blocking I/O in thread to keep UI responsive
            status = await asyncio.to_thread(
                get_episode_status,
                self.episode_uuid,
                self.journal
            )

            # Update reactive (watch_status will sync to sidebar)
            self.status = status

            # When extraction complete, refresh and stop
            if status in ("pending_edges", "done", None):
                self.active_processing = False
                sidebar = self.query_one("#entity-sidebar", EntitySidebar)
                await sidebar.refresh_entities()
                break

            # Wait before next check
            await asyncio.sleep(0.5)
        except Exception as exc:
            logger.exception(f"Status poll error: {exc}")
            self.active_processing = False
            break
```

### 4. Add User Override Flag to EntitySidebar

**File:** `charlie.py` line 237 (after `active_processing` reactive)

Add new reactive attribute:
```python
user_override: reactive[bool] = reactive(False)
```

This flag prevents background polling from overwriting user-initiated state changes.

### 5. Update EntitySidebar.watch_status()

**File:** `charlie.py` line 267-269 (modify existing watch_status)

Currently:
```python
def watch_status(self, status: str | None) -> None:
    """Reactive: re-render when status changes."""
    self._update_content()
```

Change to respect user override:
```python
def watch_status(self, status: str | None) -> None:
    """Reactive: re-render when status changes.

    Note: If user_override is True, don't update.
    This prevents background polling from overwriting
    user-initiated changes like entity deletion.
    """
    if self.user_override:
        return
    if not self.loading:
        self._update_content()
```

### 6. Set User Override on Deletion

**File:** `charlie.py` line 391-397 (in _handle_delete_result)

Currently:
```python
if len(new_entities) == 0:
    self.status = "done"
    self.loading = False

self.entities = new_entities
```

Change to:
```python
if len(new_entities) == 0:
    self.user_override = True  # PREVENT POLLING OVERRIDE
    self.status = "done"
    self.loading = False

self.entities = new_entities
```

### 7. Clean Up Worker on Screen Exit

**File:** `charlie.py` line ~860 (in action_back or on_pop_screen)

Add worker cancellation:
```python
def action_back(self) -> None:
    """Go back to home screen."""
    self.workers.cancel_group(self, "status-poll")
    self.app.pop_screen()
```

Or if using `on_pop_screen()` lifecycle:
```python
def on_pop_screen(self):
    """Called when screen is popped."""
    self.workers.cancel_group(self, "status-poll")
```

## Key Concepts from Textual Reactivity Guide

### Watch Methods
- Syntax: `watch_<attribute_name>(self, new_value)` or `watch_<attribute_name>(self, old_value, new_value)`
- Called automatically when reactive attribute changes
- Only called if value actually changes (use `always_update=True` to override)
- Can return early to skip updates (like we do with `user_override`)

### Reactive Attributes
- Syntax: `name: reactive[type] = reactive(default_value)`
- Changes automatically trigger refresh (render) and watchers
- Can be accessed/assigned like normal attributes: `self.status = "done"`
- Multiple changes in batch_update() trigger only one refresh

### Workers vs Timers
- **Workers**: Long-running async tasks (like polling)
  - Can be cancelled: `self.workers.cancel_group(self, name)`
  - Better lifecycle: stop when done, clean cancel on exit
  - Non-blocking I/O with `await asyncio.to_thread()`

- **Timers**: Repeated fixed-interval tasks (`set_interval`)
  - Harder to stop cleanly
  - Can fire during user actions (creates race conditions)
  - Block UI if work takes longer than interval

## Testing

Update `tests/test_frontend/test_entity_sidebar.py`:

1. Verify `user_override` flag prevents polling override
2. Verify watch methods propagate ViewScreen → Sidebar
3. Verify deletion sets `user_override = True`
4. Verify "No connections found" persists after deletion

```python
@pytest.mark.asyncio
async def test_user_override_prevents_polling():
    """User override flag prevents polling from overwriting status."""
    # Set user_override = True
    # Change status in background
    # Verify sidebar.status doesn't change
    pass
```

## Benefits

| Aspect | Before (Timer) | After (Reactive+Worker) |
|--------|---|---|
| Race conditions | HIGH (timer vs user actions) | LOW (user override flag) |
| UI thread safety | Blocking Redis calls | Non-blocking `asyncio.to_thread()` |
| State sync | Manual `sidebar.status = status` | Automatic watch methods |
| Lifecycle | Hard (timer cleanup) | Clean (worker auto-cancel) |
| Code clarity | Scattered timing logic | Declarative flow |

## Files Modified

- `charlie.py`: ViewScreen and EntitySidebar classes
  - Lines ~230-237: Add `user_override` reactive
  - Lines ~267-278: Update watch_status() to respect override
  - Lines ~730: Convert to reactive attributes
  - Lines ~780-830: Update on_mount, add watch methods
  - Lines ~391-397: Set user_override on deletion
  - Lines ~840-860: Add worker cleanup
  - Lines ~889-903: Replace _check_job_status() with _poll_until_complete()

- `tests/test_frontend/test_entity_sidebar.py`:
  - Verify user_override behavior
  - Verify watch method propagation

## References

- Textual Reactivity Guide: https://textual.textualize.io/guide/reactivity/
- `asyncio.to_thread()`: Non-blocking I/O in async context
- `self.run_worker()`: Run async background task
- `self.workers.cancel_group()`: Cancel all workers with given name
