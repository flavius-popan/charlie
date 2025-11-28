# Home Screen State Machine Refactoring Design

## Overview

Refactor the `HomeScreen` state management to use a proper state machine for processing state, simplifying the complex reactive property interactions and improving tractability.

## Current Problems

The `home_screen.py` has 9 reactive properties with interconnected watchers:
- `active_episode_uuid`, `queue_count`, `model_state`, `inference_enabled` all affect processing pane
- `watch_active_episode_uuid`, `watch_model_state`, `watch_queue_count` all update the same processing dots
- `_refresh_connections_pane()` called from 4 different watchers
- Race condition guards scattered throughout
- `batch_update()` calls sprinkled to prevent cascade renders

## Solution: Hybrid Approach

### Processing State Machine

A single `ProcessingStateMachine` handles model lifecycle and processing pane:

**States:**
- `idle` - Model not loaded, no active processing. Processing pane hidden.
- `loading` - Model is loading into memory. Pane visible, shows "Loading model..."
- `inferring` - Model is actively extracting from an entry. Pane visible, shows "Extracting:" with entry name and animated dot.
- `unloading` - Model is unloading from memory. Pane visible, shows "Unloading model..."

**Internal data:**
- `active_episode_uuid: str | None` - Which episode is being inferred
- `queue_count: int` - Number of entries waiting
- `inference_enabled: bool` - Whether inference is enabled

**Events:**

| Event | Description | Transitions |
|-------|-------------|-------------|
| `status_idle` | Model state is idle | any → idle |
| `status_loading` | Model state is loading | any → loading |
| `status_inferring` | Model state is inferring | any → inferring |
| `status_unloading` | Model state is unloading | any → unloading |

The machine is a **passive follower** of the worker's state - it doesn't initiate transitions, it just reflects what `get_processing_status()` reports. This is simpler than modeling events like "model_ready" which would require the machine to predict transitions.

**`apply_status()` method:**

```python
def apply_status(self, status: dict) -> ProcessingOutput:
    """Apply polling status and return output.

    Args:
        status: Dict from get_processing_status() with keys:
            - model_state: "idle", "loading", "inferring", "unloading"
            - active_uuid: str | None
            - pending_count: int
            - inference_enabled: bool

    Returns:
        ProcessingOutput with all computed display values
    """
    # Update internal data with defensive defaults (handle malformed input)
    self._active_episode_uuid = status.get("active_uuid")
    self._queue_count = status.get("pending_count", 0)
    self._inference_enabled = status.get("inference_enabled", True)

    # Route to appropriate state based on model_state (default to idle)
    model_state = status.get("model_state", "idle")
    if model_state == "idle":
        self.send("status_idle")
    elif model_state == "loading":
        self.send("status_loading")
    elif model_state == "inferring":
        self.send("status_inferring")
    elif model_state == "unloading":
        self.send("status_unloading")
    else:
        self.send("status_idle")  # Unknown state defaults to idle

    return self.output
```

**Self-transitions for idempotent polling:**

Each state event must include self-transitions to avoid `TransitionNotAllowed` errors during repeated polling (following `SidebarStateMachine` pattern at lines 92-96):

```python
status_idle = idle.to(idle) | loading.to(idle) | inferring.to(idle) | unloading.to(idle)
status_loading = idle.to(loading) | loading.to(loading) | inferring.to(loading) | unloading.to(loading)
status_inferring = idle.to(inferring) | loading.to(inferring) | inferring.to(inferring) | unloading.to(inferring)
status_unloading = idle.to(unloading) | loading.to(unloading) | inferring.to(unloading) | unloading.to(unloading)
```

**Output properties** (computed from state + data):
- `pane_visible: bool` - Whether processing pane should display
- `status_text: str` - "Loading model...", "Extracting:", "Finishing extracting:", "Unloading model..."
- `queue_text: str` - "Queue: N remaining" or empty
- `show_dot: bool` - Whether to show animated ProcessingDot
- `poll_interval: float` - How fast to poll (0.3s active, 2.0s idle)
- `is_inferring: bool` - Whether currently in inferring state
- `active_episode_uuid: str | None` - For connections pane / entry dots / name resolution
- `inference_enabled: bool` - For status_text and downstream consumers

Note: `entry_name` is NOT a machine output - HomeScreen resolves it from `active_episode_uuid` using `self.episodes`.

**Output computation logic:**

| Property | Computation |
|----------|-------------|
| `pane_visible` | `state != idle` |
| `status_text` | See status_text table below |
| `queue_text` | `f"Queue: {queue_count} remaining"` if `queue_count > 0`, else `""` |
| `show_dot` | `state in (loading, inferring, unloading)` |
| `poll_interval` | See poll_interval table below |
| `is_inferring` | `state == inferring` |

**status_text logic:**

| State | inference_enabled | Text |
|-------|------------------|------|
| idle | any | `""` |
| loading | any | `"Loading model..."` |
| inferring | True | `"Extracting:"` |
| inferring | False | `"Finishing extracting:"` |
| unloading | any | `"Unloading model..."` |

Note: "Finishing extracting:" indicates user disabled inference while extraction was in progress. The current episode will complete before unloading.

**entry_name resolution (NOT in machine):**

The machine outputs `active_episode_uuid` but does NOT resolve `entry_name`. Name resolution stays in HomeScreen because the machine has no access to `self.episodes`:

```python
# In HomeScreen._update_processing_pane(), not in machine
def _resolve_entry_name(self, uuid: str | None) -> str:
    if not uuid:
        return ""
    for ep in self.episodes:
        if ep["uuid"] == uuid:
            return get_display_title(ep)
    return ""  # Entry was deleted
```

**Poll interval logic:**

| Condition | Interval | Rationale |
|-----------|----------|-----------|
| `state in (loading, inferring, unloading)` | 0.3s | Active state changes, update UI quickly |
| `state == idle` | 2.0s | Nothing active, conserve resources |

Simple two-tier approach: fast (0.3s) when active, slow (2.0s) when idle.

### Redis Key Naming

**Keep existing key structure** - no migration needed:

Worker/Huey state (set by background tasks):
- `task:model_state` - Hash: {state, started_at} - "idle", "loading", "inferring", "unloading"
- `task:active_episode` - Hash: {uuid, journal} - Currently processing episode

App config (set by UI):
- `app:inference_enabled` - String: "true"/"false"

Queue (managed by task system):
- `pending:nodes:{journal}` - Sorted set with episode UUIDs (count derived via ZCARD)

### Inference Disabled Handling

The machine doesn't need a separate "disabled" state. The existing flow handles it:

1. User disables inference → `app:inference_enabled` = "false"
2. Running task checks this at checkpoints, exits early
3. Orchestrator calls `cleanup_if_no_work()` which:
   - Sets `task:model_state` = "unloading" (if models loaded)
   - Unloads models
   - Clears `task:model_state` (back to "idle")

The machine simply follows `task:model_state` from polling. When inference is disabled:
- If model was loaded: idle → unloading → idle (pane shows "Unloading model...", then hides)
- If model wasn't loaded: stays idle (pane stays hidden)

The `inference_enabled` flag in output is passed through for downstream consumers (connections pane "Awaiting processing..." vs "Inference disabled" messaging) but doesn't affect the machine's state.

### Temporal Pane Refresh on Processing Complete

The temporal pane needs to refresh when an episode finishes processing (to update period aggregates). The current `watch_active_episode_uuid` handles this.

**Solution:** Detect completion in `watch_processing_output` by comparing old and new outputs:

```python
def watch_processing_output(self, old: ProcessingOutput, new: ProcessingOutput) -> None:
    """Single watcher handles all processing pane updates."""
    self._update_processing_pane(new)
    self._update_entry_processing_dots(new.active_episode_uuid if new.is_inferring else None)

    # Refresh connections pane if active episode changed
    if old.active_episode_uuid != new.active_episode_uuid:
        self._refresh_connections_pane()

    # Refresh temporal pane if an episode just finished processing
    # Check is_inferring to avoid false "finished" when processing was cancelled
    finished_uuid = None
    if old.is_inferring and not new.is_inferring and old.active_episode_uuid:
        finished_uuid = old.active_episode_uuid

    if finished_uuid:
        # Refresh connections if finished entry is selected
        if finished_uuid == self.selected_entry_uuid:
            self.run_worker(self._fetch_entry_entities(finished_uuid), ...)

        # Refresh temporal if finished entry is in current period
        if self._is_episode_in_current_period(finished_uuid):
            period = self.periods[self.selected_period_index]
            self.run_worker(self._fetch_period_stats(period["start"], period["end"]), ...)
```

This preserves the existing behavior while consolidating it into the single watcher.

### Connections Pane Simplification

Remove the "loading on return" flash. No loading state for fast database queries.

**States (no state machine needed):**
1. Actively processing this entry → Loading spinner
2. Queued but not yet processing → "Awaiting processing..."
3. Has entities → Show entity list
4. No entry selected → "Select an entry"
5. Default → "No connections"

```python
def _refresh_connections_pane(self) -> None:
    # Actively processing this entry? → Spinner
    if (self.processing_output.is_inferring and
        self.processing_output.active_episode_uuid == self.selected_entry_uuid):
        show_spinner()
        return

    # Queued but not yet processing? → "Awaiting processing..."
    if self.selected_entry_status in ("pending_nodes", "pending_edges"):
        show_message("Awaiting processing...")
        return

    # Has entities? → Show list
    if self.entry_entities:
        show_entity_list()
        return

    # No entry selected? → "Select an entry"
    if not self.selected_entry_uuid:
        show_message("Select an entry")
        return

    # Default → "No connections"
    show_message("No connections")
```

### Temporal Pane

Left as-is with reactive properties. It's straightforward enough that a state machine would be over-engineering.

## HomeScreen Integration

**Reactive properties to remove:**
- `active_episode_uuid` → moved to machine
- `queue_count` → moved to machine
- `model_state` → replaced by machine state
- `inference_enabled` → moved to machine

**Watchers to remove:**
- `watch_active_episode_uuid`
- `watch_queue_count`
- `watch_model_state`
- `watch_inference_enabled`

### Call Site Migration

All existing references to removed reactive properties must be updated:

| Location | Current Code | Migration |
|----------|--------------|-----------|
| `connections_loading` property | `self.model_state == "inferring" and self.active_episode_uuid` | `self.processing_output.is_inferring and self.processing_output.active_episode_uuid` |
| `_update_entry_processing_dots()` | Reads `self.model_state`, `self.active_episode_uuid` internally | Change to accept `active_uuid: str \| None` parameter; remove internal model_state check |
| `watch_active_episode_uuid` | Direct watcher on reactive | Merge logic into `watch_processing_output` |
| `watch_queue_count` | Direct watcher on reactive | Delete - visibility handled by `watch_processing_output` |
| `watch_model_state` | Direct watcher on reactive | Merge logic into `watch_processing_output` |
| `watch_inference_enabled` | Direct watcher on reactive | Merge logic into `watch_processing_output` |
| `_update_processing_pane_content()` | Takes `active_uuid, pending_count, model_state` | Rename to `_update_processing_pane(output: ProcessingOutput)` |
| `load_episodes` entry dot init | `self.active_episode_uuid` | `self.processing_output.active_episode_uuid` |
| Entry name lookup in pane | Inline loop over `self.episodes` | Extract to `_resolve_entry_name()` helper, call from `_update_processing_pane()` |

**Updated `_update_entry_processing_dots` signature:**

Current signature reads reactive properties internally. New signature receives computed value from watcher:

```python
# OLD: Method reads self.model_state and self.active_episode_uuid internally
def _update_entry_processing_dots(self) -> None:
    should_show_dot = (
        self.model_state == "inferring"
        and self.active_episode_uuid is not None
    )
    target_uuid = self.active_episode_uuid if should_show_dot else None
    ...

# NEW: Watcher computes the value, method just applies it
def _update_entry_processing_dots(self, active_uuid: str | None) -> None:
    """Update processing dots. Watcher passes None when not inferring."""
    with self.app.batch_update():
        for entry_label in self.query(EntryLabel):
            entry_label.set_processing(entry_label.episode_uuid == active_uuid)
```

**Updated `_update_processing_pane` with entry name resolution:**

```python
def _update_processing_pane(self, output: ProcessingOutput) -> None:
    """Update processing pane display based on machine output."""
    processing_pane = self.query_one("#processing-pane", Container)
    status_widget = self.query_one("#processing-status", Static)
    entry_widget = self.query_one("#processing-entry", Static)
    queue_widget = self.query_one("#processing-queue", Static)

    with self.app.batch_update():
        # Toggle .active class for ProcessingDot visibility
        if output.show_dot:
            processing_pane.add_class("active")
        else:
            processing_pane.remove_class("active")

        # Status text from machine output
        status_widget.update(output.status_text)

        # Entry name resolved HERE, not in machine
        entry_name = self._resolve_entry_name(output.active_episode_uuid) if output.is_inferring else ""
        entry_widget.update(entry_name)

        # Queue text from machine output
        queue_widget.update(output.queue_text)
```

**Replaced with:**
```python
# Single reactive property for machine output
processing_output: reactive[ProcessingOutput] = reactive(ProcessingOutput.idle())

def watch_processing_output(self, old: ProcessingOutput, new: ProcessingOutput) -> None:
    """Single watcher handles all processing pane updates."""
    self._update_processing_pane(new)
    self._update_entry_processing_dots(new.active_episode_uuid if new.is_inferring else None)
```

**Polling loop simplified:**
```python
async def _poll_processing_status(self) -> None:
    while True:
        try:
            status = await asyncio.to_thread(get_processing_status, DEFAULT_JOURNAL)
            output = self.processing_machine.apply_status(status)
            self.processing_output = output
            await asyncio.sleep(output.poll_interval)
        except WorkerCancelled:
            break
        except Exception as e:
            logger.debug(f"Processing poll error: {e}")
            await asyncio.sleep(2.0)  # Backoff on error
```

## Edge Cases

**Deleted active entry:** If the episode being processed is deleted while processing, `entry_name` lookup returns `None` and the pane shows empty entry name. The worker handles cleanup via `NodeNotFoundError`.

**Rapid state transitions:** If the worker transitions `idle → loading → inferring` within a single poll cycle (0.3s), the UI may skip showing the "Loading model..." state. This is acceptable - eventual consistency is sufficient for status display.

**Malformed status data:** If `get_processing_status()` returns unexpected values, `apply_status()` should default to idle state. The machine should never crash from bad input.

## File Changes

**New files:**
- `frontend/state/processing_state_machine.py` - The state machine (follows `sidebar_state_machine.py` pattern)

**Modified files:**
- `frontend/screens/home_screen.py` - Remove 4 reactive properties, 4 watchers, simplify connections pane

## State Machine Implementation Details

Following `sidebar_state_machine.py` pattern:

```python
from statemachine import StateMachine, State

class ProcessingStateMachine(StateMachine):
    """State machine for processing pane display."""

    # States
    idle = State(initial=True)
    loading = State()
    inferring = State()
    unloading = State()

    # Events with self-transitions (see Self-transitions section above)
    status_idle = idle.to(idle) | loading.to(idle) | inferring.to(idle) | unloading.to(idle)
    status_loading = idle.to(loading) | loading.to(loading) | inferring.to(loading) | unloading.to(loading)
    status_inferring = idle.to(inferring) | loading.to(inferring) | inferring.to(inferring) | unloading.to(inferring)
    status_unloading = idle.to(unloading) | loading.to(unloading) | inferring.to(unloading) | unloading.to(unloading)

    def __init__(self):
        super().__init__()
        self._active_episode_uuid: str | None = None
        self._queue_count: int = 0
        self._inference_enabled: bool = True

    @property
    def output(self) -> ProcessingOutput:
        """Compute output based on current state and internal data."""
        state_name = self.current_state.id

        # Status text with "Finishing extracting:" variant
        if state_name == "idle":
            status_text = ""
        elif state_name == "loading":
            status_text = "Loading model..."
        elif state_name == "inferring":
            status_text = "Finishing extracting:" if not self._inference_enabled else "Extracting:"
        elif state_name == "unloading":
            status_text = "Unloading model..."
        else:
            status_text = ""

        return ProcessingOutput(
            pane_visible=(state_name != "idle"),
            status_text=status_text,
            queue_text=f"Queue: {self._queue_count} remaining" if self._queue_count > 0 else "",
            show_dot=(state_name in ("loading", "inferring", "unloading")),
            poll_interval=0.3 if state_name != "idle" else 2.0,
            is_inferring=(state_name == "inferring"),
            active_episode_uuid=self._active_episode_uuid,
            inference_enabled=self._inference_enabled,
        )

    def apply_status(self, status: dict) -> ProcessingOutput:
        """Apply polling status and return output. See apply_status() section above."""
        ...
```

**Optional: `generate_diagram()` function** - Can add later if useful for debugging, following `sidebar_state_machine.py:438-466`.

## ProcessingOutput Dataclass

```python
@dataclass
class ProcessingOutput:
    pane_visible: bool
    status_text: str          # "Loading model...", "Extracting:", "Finishing extracting:", etc.
    queue_text: str           # "Queue: N remaining" or ""
    show_dot: bool            # Animated ProcessingDot
    poll_interval: float      # How fast to poll (0.3s active, 2.0s idle)

    # For connections pane / entry dots / name resolution
    is_inferring: bool
    active_episode_uuid: str | None
    inference_enabled: bool

    @classmethod
    def idle(cls) -> "ProcessingOutput":
        return cls(
            pane_visible=False,
            status_text="",
            queue_text="",
            show_dot=False,
            poll_interval=2.0,
            is_inferring=False,
            active_episode_uuid=None,
            inference_enabled=True,
        )
```

Note: `entry_name` is NOT in the output - it's resolved by HomeScreen from `active_episode_uuid` using `self.episodes`.

## Testing

**New test file:**
- `tests/frontend/state/test_processing_state_machine.py`

**State machine tests (following `test_sidebar_state_machine.py` pattern):**
- Test each state transition (idle ↔ loading ↔ inferring ↔ unloading)
- Test output properties are correct for each state
- Test `apply_status()` correctly routes raw polling data to state transitions
- Test output values: `pane_visible`, `status_text`, `show_dot`, `poll_interval`
- Test `poll_interval` is 0.3s when active, 2.0s when idle
- Test `apply_status()` with malformed/missing keys defaults to idle
- Test self-transitions don't raise `TransitionNotAllowed`

**Integration tests:**
- Test `_refresh_connections_pane()` logic with various combinations:
  - Processing this entry → spinner
  - Queued entry → "Awaiting processing..."
  - Entry with entities → list
  - No selection → "Select an entry"
  - Processed, no entities → "No connections"
- Test polling loop error handling (exception → 2.0s backoff)
- Test `watch_processing_output` triggers temporal refresh when episode completes

**Redis integration tests:**
- Test `get_processing_status()` returns correct structure
- Test machine correctly interprets all model_state values

**HomeScreen integration tests:**
- Test `_resolve_entry_name()` returns empty string when UUID not in episodes list
- Test `status_text` shows "Finishing extracting:" when `inference_enabled=False` and `is_inferring=True`

**Test isolation note:**
All new tests must use existing fixtures from `tests/conftest.py` (lines 29-39) that redirect to `tests/data/charlie-test.db` and mock Redis. Do not touch production data.

## Migration

Not needed - pre-release, will wipe existing data and start fresh.
