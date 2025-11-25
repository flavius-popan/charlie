# The Halting Problem: Graceful Shutdown Strategy

## User Requirements

- UI responsiveness is priority #1: quitting must not freeze the Textual event loop; the user should see "Just a sec! (closing up shop)" immediately.
- No modal, no force quit option.
- Current inference task MUST finish (can't interrupt llama-cpp mid-generation).
- After shutdown is requested, no NEW task should start.
- Exit as fast as possible without killing in-flight inference; keep any finished work if possible.
- Properly resolve Huey `_tasks_in_flight` race condition (not just patch/ignore).
- Graceful shutdown must also trigger when the user presses Textual’s built-in Ctrl+C.

---

## Solution Architecture

### Layer 1: Single shutdown flag, never reset early

- Add `request_shutdown()` in `backend/database/lifecycle.py` to set `_shutdown_requested` from any thread.  
- Guard against premature flag reset: defer `reset_lifecycle_state()` until **after** worker/consumer teardown has fully completed (no active worker threads). Avoid clearing the flag inside `shutdown_database()` while background tasks may still be running.
- Add a helper `shutdown_in_progress()` if needed for clarity, but keep one source of truth: `_shutdown_requested`.

### Layer 2: Non-blocking, UI-first quit pipeline (covers Ctrl+C)

- Convert quit handling to an async, non-blocking flow so Textual’s event loop stays responsive:
  - Immediately call `request_shutdown()` and show `notify("Just a sec! (closing up shop)", timeout=10)`; do this on the UI thread so the message renders.
  - Schedule a background coroutine (`asyncio.create_task` or `call_later`) that performs the heavy parts off the UI thread:
    1) stop Huey consumer (see Layer 3) in a thread or `to_thread`,  
    2) then shut down the database,  
    3) finally call `app.exit()` once teardown completes.  
  - Provide a short progress pulse/spinner message if shutdown exceeds a few seconds; keep the UI alive while waiting.
- Wire Textual’s Ctrl+C path to the same pipeline (e.g., override `on_shutdown_request` / `on_key` for ctrl+c, or ensure `on_unmount` delegates to the shared async shutdown function so signal-based exits also set the shutdown flag before teardown).

### Layer 3: Task cancellation without losing finished inference

- Add `TaskCancelled` + `check_cancellation()` in `backend/services/tasks.py`.
- Checkpoints:
  1) before any work,  
  2) before model load,  
  3) before enqueueing/rescheduling,  
  4) **after persistence**, not between inference and persistence—so completed generations are saved.  
- Catch `TaskCancelled` before generic `Exception` to avoid marking tasks dead.
- Orchestrator should check the flag at entry and skip rescheduling when shutdown is requested.

### Layer 4: Huey shutdown race fix (retain cleanup without KeyError)

- Keep consumer logic but make `_tasks_in_flight` cleanup race-free instead of removing it entirely:
  - Override `notify_interrupted_tasks()` to iterate over a snapshot and use `discard` (idempotent) instead of `pop` to avoid KeyError when worker finally blocks also remove tasks.
  - Call `notify_interrupted_tasks()` only when shutdown is non-graceful; graceful path should allow worker threads to drain naturally after `stop(graceful=True)`.
- Ensure `stop_huey_consumer` blocks only the background shutdown coroutine, not the UI thread (see Layer 2). Use a generous but finite wait and surface a warning in the UI if the worker exceeds it while still keeping the interface responsive.

### Layer 5: Consistent shutdown paths

- Make `CharlieApp._graceful_shutdown`, `HomeScreen.action_quit`, `on_unmount`, and ctrl+c all delegate to the same async shutdown helper so the flag is set and respected everywhere.
- Ensure Redis/DB teardown only runs after the consumer/worker confirms exit or times out; do not tear down the DB with the flag cleared while tasks may still be alive.

### Layer 6: Testing & verification (headless)

- Textual headless tests:
  - Quit while a fake long-running task is active; assert notify is visible, UI remains responsive (e.g., another widget can be focused), and shutdown coroutine completes without blocking the pilot.
  - Ctrl+C path triggers the same shutdown helper and sets the flag.
- Huey tests:
  - Simulate non-graceful stop to ensure overridden `notify_interrupted_tasks` does not raise and leaves no leaked locks/keys.
- DB isolation check: after tests, verify `data/charlie.db` unchanged; work should target `tests/data/charlie-test.db`.

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/database/lifecycle.py` | Add `request_shutdown()`, delay lifecycle reset until worker teardown completes |
| `backend/services/tasks.py` | Add `TaskCancelled`/`check_cancellation()`, move checkpoints to avoid dropping post-inference results |
| `backend/services/queue.py` | Make `notify_interrupted_tasks` idempotent/safer; ensure stop is invoked off the UI thread |
| `frontend/screens/home_screen.py` | Convert quit to non-blocking async flow with immediate notify |
| `charlie.py` (app) | Route ctrl+c/on_unmount through the shared async shutdown helper |
| `tests/test_frontend/test_charlie.py` + new Huey test | Add headless shutdown responsiveness + race regression coverage |

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
| Multiple queued tasks | Current finishes; orchestrator halts; remaining tasks stay pending. |
| Quit during model load | Cancellation before load; fast exit. |
| Non-graceful stop (e.g., abrupt signal) | Safe `notify_interrupted_tasks` prevents KeyError/lock leaks; flag remains set. |
