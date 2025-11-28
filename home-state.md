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
    # Update internal data
    self._active_episode_uuid = status["active_uuid"]
    self._queue_count = status["pending_count"]
    self._inference_enabled = status["inference_enabled"]

    # Route to appropriate state based on model_state
    model_state = status["model_state"]
    if model_state == "idle":
        self.send("status_idle")
    elif model_state == "loading":
        self.send("status_loading")
    elif model_state == "inferring":
        self.send("status_inferring")
    elif model_state == "unloading":
        self.send("status_unloading")

    return self.output
```

**Output properties** (computed from state + data):
- `pane_visible: bool` - Whether processing pane should display
- `status_text: str` - "Loading model...", "Extracting:", "Unloading model...", etc.
- `entry_name: str | None` - Name of entry being processed (when inferring)
- `queue_text: str` - "Queue: N remaining" or empty
- `show_dot: bool` - Whether to show animated ProcessingDot
- `poll_interval: float` - How fast to poll (0.3s active, 2.0s idle)
- `is_inferring: bool` - Whether currently in inferring state
- `active_episode_uuid: str | None` - For connections pane / entry dots
- `inference_enabled: bool` - For downstream consumers

**Output computation logic:**

| Property | Computation |
|----------|-------------|
| `pane_visible` | `state != idle` |
| `status_text` | idle: "", loading: "Loading model...", inferring: "Extracting:", unloading: "Unloading model..." |
| `entry_name` | Only set when inferring, looked up from episodes list by `active_episode_uuid`. Returns `None` if UUID not found (entry was deleted). |
| `queue_text` | `f"Queue: {queue_count} remaining"` if `queue_count > 0`, else `""` |
| `show_dot` | `state in (loading, inferring, unloading)` |
| `poll_interval` | See table below |
| `is_inferring` | `state == inferring` |

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
    finished_uuid = None
    if old.active_episode_uuid and old.active_episode_uuid != new.active_episode_uuid:
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

## ProcessingOutput Dataclass

```python
@dataclass
class ProcessingOutput:
    pane_visible: bool
    status_text: str          # "Loading model...", "Extracting:", etc.
    entry_name: str | None    # Name of entry being processed
    queue_text: str           # "Queue: N remaining" or ""
    show_dot: bool            # Animated ProcessingDot
    poll_interval: float      # How fast to poll (0.3s active, 2.0s idle)

    # For connections pane / entry dots
    is_inferring: bool
    active_episode_uuid: str | None
    inference_enabled: bool

    @classmethod
    def idle(cls) -> "ProcessingOutput":
        return cls(
            pane_visible=False,
            status_text="",
            entry_name=None,
            queue_text="",
            show_dot=False,
            poll_interval=2.0,
            is_inferring=False,
            active_episode_uuid=None,
            inference_enabled=True,
        )
```

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
- Test `entry_name` is `None` when `active_episode_uuid` not in episodes list

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

## Migration

Not needed - pre-release, will wipe existing data and start fresh.
