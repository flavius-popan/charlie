# The Halting Problem: Graceful Shutdown Strategy

## User Requirements

- UI responsiveness is priority #1: quitting must not freeze the Textual event loop; the user should see "Just a sec! (closing up shop)" immediately.
- No modal, no force quit option.
- Current inference task MUST finish (can't interrupt llama-cpp mid-generation).
- After shutdown is requested, no NEW task should start.
- Exit as fast as possible without killing in-flight inference; keep any finished work if possible.
- Properly resolve Huey `_tasks_in_flight` race condition (not just patch/ignore).
- Graceful shutdown must also trigger on SIGINT/SIGTERM (terminal Ctrl+C, kill, window close).

> **Note:** Textual 6.6.0 captures keyboard Ctrl+C as the `help_quit` binding (shows the help notification) and does **not** raise SIGINT. External signals (e.g., `kill -INT`, terminal window close) still reach the Python signal handler.

---

## Investigation Findings (Pre-Implementation)

### Current State Analysis

1. **Shutdown flag reset by test helper:** `shutdown_database()` in `backend/database/__init__.py:36-45` calls `_close_db()` (sets `_shutdown_requested=True`) then immediately calls `reset_lifecycle_state()` (resets to `False`). This is handy for test re-init but prevents the flag from persisting during production shutdown; it should be treated as a test-only helper, not a runtime shutdown path.

2. **Flag set too late:** Current flow sets the flag AFTER `stop_huey_consumer()` returns. Workers never see it.

3. **No task cancellation infrastructure:** `tasks.py` has no `TaskCancelled` exception or `check_cancellation()`. Generic `except Exception` marks episodes as "dead".

4. **Huey `_tasks_in_flight` scope:** In Huey 2.5.4 the set is an instance attribute initialized in `Huey.__init__` (`api.py:131`). The pop/remove race appears only on non-graceful shutdown when workers aren't joined before `notify_interrupted_tasks()` runs.

5. **`InProcessConsumer` signal safety:** Our consumer subclass skips `_set_signal_handlers()`, so consumer threads never receive SIGTERM directly. External signals to the main process still need handling, but in-app Ctrl+C is just a key binding (see note above).

6. **Orchestrator always reschedules:** `orchestrate_inference_work` reschedules itself in `finally` block with no shutdown check.

---

## Solution Architecture

### Layer 1: Single shutdown flag, never reset early

**Current bug:** `backend/database/__init__.py:shutdown_database()` calls `reset_lifecycle_state()` immediately after `_close_db()`, defeating the flag.

**Fix:**
- Add `request_shutdown()` in `backend/database/lifecycle.py` to set `_shutdown_requested` from any thread (no teardown, just flag).
- Remove `reset_lifecycle_state()` call from `shutdown_database()`. The flag should remain `True` after shutdown.
- `reset_lifecycle_state()` should only be called in test fixtures, never in production shutdown path.
- Treat `shutdown_database()` as a test helper for re-initialization; production code should use the async shutdown path (`request_shutdown()` + Huey stop + DB teardown) rather than calling `shutdown_database()` directly.
- Add `is_shutdown_requested()` checks in tasks before starting new work.

### Layer 2: Non-blocking, UI-first quit pipeline + signal handling

**UI quit flow (q key):**
- Convert `action_quit()` to async, non-blocking:
  1. Immediately call `request_shutdown()` and show `notify("Just a sec! (closing up shop)", timeout=10)` on UI thread.
  2. Use `run_worker()` or `asyncio.to_thread()` to run blocking teardown off the event loop:
     - `stop_huey_consumer()` (waits for current task to finish)
     - `shutdown_database()` (stops Redis)
  3. Call `app.exit()` after teardown completes.
- If teardown exceeds ~3s, update the notification to show progress ("Still waiting for inference to finish...").
- Double-quit should be idempotent (check `is_shutdown_requested()` and return early if already shutting down).

**Signal handling (SIGINT/SIGTERM):**
- Register signal handlers in `CharlieApp.on_mount()` using asyncio's `loop.add_signal_handler()`:
  ```python
  loop = asyncio.get_running_loop()
  for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, lambda: asyncio.create_task(self._async_shutdown()))
  ```
- This intercepts external signals (`kill -INT`, terminal close). Keyboard Ctrl+C inside the Textual app remains a key binding and does **not** raise SIGINT.
- Textual's built-in Ctrl+C key binding (`action_help_quit`) remains unchanged (shows help notification).

**`on_unmount` handler:**
- Already exists but is synchronous and blocking. Keep it as a fallback safety net.
- Primary shutdown should complete before `on_unmount` is called; if not, `on_unmount` ensures cleanup still happens.

### Layer 3: Task cancellation without losing finished inference

**New infrastructure in `backend/services/tasks.py`:**
```python
class TaskCancelled(Exception):
    """Raised when a task detects shutdown and should exit cleanly."""
    pass

def check_cancellation():
    """Raise TaskCancelled if shutdown has been requested."""
    from backend.database.lifecycle import is_shutdown_requested
    if is_shutdown_requested():
        raise TaskCancelled("Shutdown requested")
```

**Checkpoint placement in `extract_nodes_task`:**
1. At function entry (before any work)
2. Before `get_model("llm")` call (before expensive model load)
3. After `set_episode_status()` persistence (work is saved, safe to exit)

**Do NOT checkpoint:**
- Between inference completion and persistence (would lose finished work)

**Exception handling update:**
```python
try:
    check_cancellation()  # Entry checkpoint
    # ... task work ...
except TaskCancelled:
    logger.info("Task cancelled due to shutdown")
    return {"cancelled": True}  # Don't mark as dead
except Exception:
    set_episode_status(episode_uuid, "dead", journal)
    raise
```

**Orchestrator changes:**
- Check `is_shutdown_requested()` at entry; return immediately if shutting down.
- Skip `orchestrate_inference_work.schedule(delay=3)` in `finally` block if shutdown requested:
  ```python
  finally:
      if reschedule and not is_shutdown_requested():
          orchestrate_inference_work.schedule(delay=3)
  ```

### Layer 4: Huey shutdown race fix (retain cleanup without KeyError)

**Investigation findings:**
- `_tasks_in_flight` is an instance set initialized in `Huey.__init__` (`huey/api.py:131`).
- `execute()` does `.add(task)` before execution, `.remove(task)` in `finally`; `notify_interrupted_tasks()` pops remaining tasks after the consumer loop ends.
- Race to KeyError only happens on **non-graceful** shutdown when workers are not joined before `notify_interrupted_tasks()` iterates.
- Our `stop_huey_consumer()` already calls `consumer.stop(graceful=True)`, so workers are joined before `notify_interrupted_tasks()` runs; the race is currently unreachable.
- `InProcessConsumer._set_signal_handlers()` is a no-op, so the consumer itself won't be flipped to non-graceful by SIGTERM; only the main process handles signals.

**Mitigation strategy:**
1. **Primary defense (sufficient now):** Keep all shutdown paths graceful so workers drain before `notify_interrupted_tasks()` runs.
2. **Optional future guard:** If we ever introduce non-graceful exits, override `notify_interrupted_tasks()` to snapshot + `discard` to avoid KeyError without losing interrupted signals.

**Consumer stop behavior:**
- `stop_huey_consumer()` already uses `graceful=True` with 3s timeout.
- If timeout exceeded, log warning but continue shutdown. Task may be orphaned but app exits cleanly.
- Since this runs in a worker thread (Layer 2), UI remains responsive during the wait.

### Layer 5: Consistent shutdown paths

**Single async shutdown helper in `CharlieApp`:**
```python
async def _async_shutdown(self):
    """Unified shutdown path for all exit triggers."""
    from backend.database.lifecycle import request_shutdown, is_shutdown_requested

    if is_shutdown_requested():
        return  # Already shutting down (double-quit)

    request_shutdown()  # Set flag FIRST
    self.notify("Just a sec! (closing up shop)", timeout=10)

    # Run blocking teardown off the event loop
    await asyncio.to_thread(self._blocking_shutdown)
    self.exit()

def _blocking_shutdown(self):
    """Blocking teardown (runs in thread)."""
    self.stop_huey_worker()  # Waits for current task
    shutdown_database()       # Stops Redis
```

**Entry points all delegate to `_async_shutdown()`:**
| Entry Point | Implementation |
|-------------|----------------|
| `HomeScreen.action_quit()` | `asyncio.create_task(self.app._async_shutdown())` |
| `CharlieApp.on_mount()` signal handlers | `asyncio.create_task(self._async_shutdown())` |
| `CharlieApp.on_unmount()` | Synchronous fallback, calls `_blocking_shutdown()` directly |

**Ordering guarantee:**
1. `request_shutdown()` sets flag (tasks see it immediately)
2. `stop_huey_consumer()` waits for workers (current task finishes)
3. `shutdown_database()` closes Redis (flag stays True)
4. `app.exit()` triggers Textual teardown

### Layer 6: Testing & verification (headless)

**Textual headless tests (`tests/test_frontend/test_charlie.py`):**
- `test_quit_shows_notification`: Press `q`, assert notification visible within 100ms.
- `test_quit_ui_responsive_during_shutdown`: Press `q`, then verify other widgets can still be focused/interacted with.
- `test_double_quit_idempotent`: Press `q` twice, assert no errors, single shutdown sequence.
- `test_shutdown_flag_set_before_teardown`: Mock `request_shutdown()`, verify called before `stop_huey_consumer()`.

**Task tests (`tests/test_backend/test_services_tasks.py`):**
- `test_task_cancelled_not_marked_dead`: Trigger `TaskCancelled`, verify episode status is NOT "dead".
- `test_orchestrator_stops_rescheduling_on_shutdown`: Set shutdown flag, verify no new schedule call.
- `test_check_cancellation_raises_when_shutdown`: Set flag, call `check_cancellation()`, expect `TaskCancelled`.

**Huey tests (optional, only if implementing backup defense):**
- `test_notify_interrupted_tasks_no_keyerror`: Simulate race condition, verify no KeyError.

**DB isolation:** After all tests, verify `data/charlie.db` unchanged; work targets `tests/data/charlie-test.db`.

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/database/lifecycle.py` | Add `request_shutdown()` function; export it in `__all__` |
| `backend/database/__init__.py` | Remove `reset_lifecycle_state()` call from `shutdown_database()` |
| `backend/services/tasks.py` | Add `TaskCancelled` class, `check_cancellation()` function; update exception handling in `extract_nodes_task`; add shutdown check to `orchestrate_inference_work` |
| `frontend/screens/home_screen.py` | Change `action_quit()` to async, delegate to `app._async_shutdown()` |
| `charlie.py` | Add `_async_shutdown()`, `_blocking_shutdown()` methods; register signal handlers in `on_mount()`; update `_graceful_shutdown()` to use new flag |
| `tests/test_frontend/test_charlie.py` | Add headless shutdown tests |
| `tests/test_backend/test_services_tasks.py` | Add task cancellation tests (aligns with existing suite location) |

---

## What This Achieves

1. **UI stays responsive**: quit notification renders immediately; shutdown runs in background.  
2. **Current task completes & persists**: inference finishes and writes results before exit.  
3. **No new work starts**: flag checked before scheduling/starting tasks.  
4. **Race-free cleanup**: Huey interruption handling avoids `_tasks_in_flight` KeyError without losing cleanup.  
5. **Single shutdown path**: keyboard quit, ctrl+c, and window close share the same logic.  
6. **Documented, test-backed behavior**: headless tests guard against regressions.

---

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| No tasks running | Immediate notify; fast exit (<1s). |
| Task mid-inference | UI stays responsive; task completes & persists; exit after worker drains. |
| Multiple queued tasks | Current finishes; orchestrator halts; remaining tasks stay pending in Redis (resume on next app start). |
| Quit during model load | `check_cancellation()` before model load raises `TaskCancelled`; fast exit. |
| Double-quit (q pressed twice) | Second press detected via `is_shutdown_requested()`, returns early. No duplicate teardown. |
| External SIGINT (e.g., `kill -INT`) | Signal handler calls `_async_shutdown()`, same flow as q key. |
| SIGTERM (kill command) | Signal handler calls `_async_shutdown()`, graceful shutdown. |
| Consumer timeout exceeded | Warning logged, shutdown continues. Task may be orphaned but app exits cleanly. |
| Model in memory at exit | Process termination frees all memory. No temp file leaks (llama-cpp uses in-memory buffers). |

---

## Implementation Checklist

- [ ] **Layer 1:** Add `request_shutdown()` to lifecycle.py
- [ ] **Layer 1:** Remove `reset_lifecycle_state()` from `shutdown_database()`
- [ ] **Layer 2:** Create `_async_shutdown()` and `_blocking_shutdown()` in CharlieApp
- [ ] **Layer 2:** Register SIGINT/SIGTERM handlers in `on_mount()`
- [ ] **Layer 3:** Add `TaskCancelled` and `check_cancellation()` to tasks.py
- [ ] **Layer 3:** Update `extract_nodes_task` exception handling
- [ ] **Layer 3:** Add shutdown check to `orchestrate_inference_work`
- [ ] **Layer 5:** Update `HomeScreen.action_quit()` to call `_async_shutdown()`
- [ ] **Layer 6:** Write headless shutdown tests
- [ ] **Layer 6:** Write task cancellation tests (in `tests/test_backend/test_services_tasks.py`)
- [ ] **Verification:** Manual test: quit during inference, verify task completes

---

## Post-Implementation Bug: Premature Shutdown Check (2025-11-24)

### Symptom

After implementing the halting problem solution, quitting during inference caused a cascade of errors:
```
RuntimeError: Database shutdown in progress
```
followed by:
```
KeyError: backend.services.tasks.extract_nodes_task: ...
```

The log showed inference completing successfully ("Extracted 25 provisional entities") but the task couldn't persist its results.

### Root Cause

`backend/database/redis_ops.py:84-85` contained a premature shutdown check:
```python
if lifecycle.is_shutdown_requested():
    raise RuntimeError("Database shutdown in progress")
```

This check conflated two distinct concerns:
1. **"Don't start new tasks"** - what `check_cancellation()` should enforce at task entry points
2. **"Database is unavailable"** - only true when `_db` is None (after actual teardown)

The shutdown flag was intended to signal tasks to exit early at checkpoints, NOT to block all database operations globally.

### The Race Condition

```
User presses 'q':
  1. request_shutdown()         → _shutdown_requested = True
  2. notify("Just a sec!")
  3. await to_thread(_blocking_shutdown)
     └→ stop_huey_consumer()   → waits for current task (graceful=True)

Meanwhile, in Huey worker thread:
  1. extract_nodes_task running
  2. LLM inference completes    → "Extracted 25 provisional entities"
  3. get_suppressed_entities()  → calls redis_ops()
  4. redis_ops() sees flag      → raises RuntimeError!
  5. Work is lost, episode marked "dead"
```

The paradox: We waited for the task to finish (`graceful=True`), but the shutdown flag prevented it from finishing.

### Why Tests Missed It

The existing tests mocked individual Redis functions (`get_episode_status`, `set_episode_status`), bypassing the `redis_ops()` context manager entirely. The shutdown check was never exercised because mocks don't call the real context manager.

### Fix

**Removed lines 84-85 from `redis_ops.py`:**
```python
# REMOVED:
if lifecycle.is_shutdown_requested():
    raise RuntimeError("Database shutdown in progress")
```

The existing check for `lifecycle._db is None` is sufficient - it catches actual database unavailability after `_close_db()` runs.

**Added test `test_redis_ops_allows_operations_during_shutdown`** to verify the correct behavior: `redis_ops()` should allow operations while the database is available, regardless of shutdown flag.

### Key Lesson

The shutdown flag is for **cooperative cancellation at checkpoints** (via `check_cancellation()`), not for **blocking all database access**. In-flight tasks must be able to persist their completed work during the graceful shutdown window.

### Files Changed

| File | Change |
|------|--------|
| `backend/database/redis_ops.py` | Removed premature shutdown check (lines 84-85) |
| `tests/test_backend/test_redis_ops.py` | Updated `test_shutdown_behavior` → `test_shutdown_flag_allows_operations_while_db_available` |
| `tests/test_backend/test_services_tasks.py` | Added `test_redis_ops_allows_operations_during_shutdown` |

---

## Post-Implementation Bug: UI Freeze on Quit (2025-11-24)

### Symptom

After implementing the halting problem solution, pressing `q` to quit caused a complete UI freeze:
- No notification appeared ("Just a sec! (closing up shop)" never rendered)
- UI was completely unresponsive
- User had to spam quit commands to exit

### Root Cause

`action_quit()` in `frontend/screens/home_screen.py` used `await` instead of `asyncio.create_task()`:

```python
# WRONG (blocking):
async def action_quit(self):
    await self.app._async_shutdown()

# CORRECT (fire-and-forget):
def action_quit(self):
    asyncio.create_task(self.app._async_shutdown())
```

The design doc (line 164) specified `asyncio.create_task()`, but the implementation used `await`.

### Why `await` Freezes the UI

1. Textual's `_dispatch_action()` awaits the action handler to complete before returning
2. With `await self.app._async_shutdown()`, the action handler doesn't return until shutdown finishes
3. Even though `asyncio.to_thread()` runs blocking work in a thread, Textual doesn't process renders until the action dispatch completes
4. The notification is queued via `post_message()` but never rendered

### The Paradox

The implementation *looks* correct at first glance:
- `asyncio.to_thread()` runs blocking work off the event loop
- The event loop is technically "free" during the thread wait
- But Textual batches message processing and won't render until the action completes

### Fix

1. **Changed `action_quit()` to fire-and-forget** (`frontend/screens/home_screen.py:176-178`):
   ```python
   def action_quit(self):
       asyncio.create_task(self.app._async_shutdown())
   ```

2. **Added yield point after notify** (`charlie.py:191-192`):
   ```python
   self.notify("Just a sec! (closing up shop)", timeout=10)
   await asyncio.sleep(0)  # Yield to allow notification to render
   ```

3. **Updated test to verify fire-and-forget** (`tests/test_frontend/test_charlie.py:1398-1435`):
   - Test now asserts shutdown is NOT finished immediately after press (proving non-blocking)
   - Then waits and asserts shutdown completes

### Key Lesson

In Textual, async action handlers that `await` long operations will freeze the UI even if the work runs in a thread. Use `asyncio.create_task()` for fire-and-forget patterns where UI responsiveness is critical.

### Files Changed

| File | Change |
|------|--------|
| `frontend/screens/home_screen.py` | `action_quit()`: `await` → `asyncio.create_task()` |
| `charlie.py` | `_async_shutdown()`: added `await asyncio.sleep(0)` after notify |
| `tests/test_frontend/test_charlie.py` | Updated `test_quit_ui_responsive_during_shutdown` with proper assertions |
