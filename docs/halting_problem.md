# The Halting Problem: Graceful Shutdown Strategy

## User Requirements

- No modal, no force quit option
- Quick message: "Just a sec! (closing up shop)" then exit
- Current inference task MUST finish (can't interrupt llama-cpp mid-generation)
- Before NEXT task starts, check shutdown flag and abort
- Exit as fast as possible without killing inference
- Properly resolve huey `_tasks_in_flight` race condition (not just patch/ignore)

---

## Solution Architecture

### Layer 1: Reuse Existing Shutdown Signal

**IMPORTANT**: There's already a `_shutdown_requested` flag in `backend/database/lifecycle.py`. We add a `request_shutdown()` function to set it from anywhere, avoiding duplicate flags.

**File**: `backend/database/lifecycle.py` (add to existing file)

```python
# Add after line 36 (after _shutdown_requested = False)
def request_shutdown():
    """Request graceful shutdown. Safe to call from any thread."""
    global _shutdown_requested
    _shutdown_requested = True
```

Update `__all__` (at line 358) to include it:
```python
__all__ = [
    "_ensure_graph",
    "get_falkordb_graph",
    "enable_tcp_server",
    "disable_tcp_server",
    "get_tcp_server_endpoint",
    "get_tcp_server_password",
    "is_shutdown_requested",
    "request_shutdown",  # ADD THIS
    "reset_lifecycle_state",
]
```

### Layer 2: Task Cancellation at Safe Points

Tasks check shutdown flag at multiple checkpoints. **Critical**: Catch `TaskCancelled` BEFORE generic `Exception` to prevent cancelled tasks from being marked "dead".

**File**: `backend/services/tasks.py` (modified)

```python
from backend.database.lifecycle import is_shutdown_requested

class TaskCancelled(Exception):
    """Raised when task detects shutdown request."""
    pass

def check_cancellation():
    """Call at safe checkpoints (NOT during inference)."""
    if is_shutdown_requested():
        raise TaskCancelled("Shutdown requested")

@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    try:
        # Checkpoint 1: Before starting any work
        check_cancellation()

        current_status = get_episode_status(episode_uuid, journal)
        if current_status != "pending_nodes":
            return {"already_processed": True}

        if not get_inference_enabled():
            return {"inference_disabled": True}

        # Checkpoint 2: Before model load (avoid wait if shutting down)
        check_cancellation()

        lm = get_model("llm")

        # NO checkpoint here - let inference run to completion
        with dspy.context(lm=lm):
            result = extract_nodes(episode_uuid, journal)

        # Checkpoint 3: AFTER inference completes, before persisting
        check_cancellation()

        set_episode_status(episode_uuid, "pending_edges", journal, uuid_map=result.uuid_map)
        return {...}

    except TaskCancelled:
        # MUST be caught BEFORE generic Exception handler
        logger.info("Task cancelled for episode %s (shutdown)", episode_uuid)
        return {"cancelled": True}
    except Exception:
        # Existing error handling - only for real errors, not cancellation
        set_episode_status(episode_uuid, "dead", journal)
        raise
    finally:
        cleanup_if_no_work()


@huey.task()
def orchestrate_inference_work(reschedule: bool = True):
    """Maintenance loop - must also respect shutdown."""
    try:
        # Checkpoint: Don't work or reschedule if shutdown requested
        check_cancellation()

        # ... existing work ...

    except TaskCancelled:
        logger.info("Orchestrator stopped (shutdown requested)")
        return {"cancelled": True}
    except Exception:
        logger.exception("orchestrate_inference_work failed")
    finally:
        if reschedule:
            try:
                # Only reschedule if NOT shutting down
                if not is_shutdown_requested():
                    orchestrate_inference_work.schedule(delay=3)
            except Exception:
                logger.exception("Failed to reschedule orchestrate_inference_work")
```

### Layer 3: Fix Huey Race Condition

Override `run()` in `InProcessConsumer` to skip `notify_interrupted_tasks()`. With graceful shutdown, tasks complete and clean themselves up in their finally blocks - calling `notify_interrupted_tasks()` races with those finally blocks.

**File**: `backend/services/queue.py` (modified InProcessConsumer)

```python
# Add required imports at top
import os
import sys
import time
from huey.consumer import Consumer, ConsumerStopped

class InProcessConsumer(Consumer):
    """Consumer variant for in-process use with TUI."""

    def _set_signal_handlers(self):
        # Signals cannot be installed from non-main threads
        return

    def run(self):
        """Run consumer without notify_interrupted_tasks race condition.

        With graceful shutdown, tasks complete and clean themselves up in
        their finally blocks. Calling notify_interrupted_tasks() after the
        loop exits races with worker threads still in their finally blocks,
        causing KeyError when workers try to remove tasks that were already
        popped by notify_interrupted_tasks().
        """
        self.start()
        health_check_ts = time.time()

        while True:
            try:
                health_check_ts = self.loop(health_check_ts)
            except ConsumerStopped:
                break

        # REMOVED: self.huey.notify_interrupted_tasks()
        # Tasks clean up via worker finally blocks - no race condition

        if self._restart:
            self._logger.info('Consumer will restart.')
            python = sys.executable
            os.execl(python, python, *sys.argv)
        else:
            self._logger.info('Consumer exiting.')
```

### Layer 4: Quick Exit Flow with Notification

**File**: `frontend/screens/home_screen.py` (modified)

```python
def action_quit(self):
    """Quick graceful shutdown with user feedback."""
    from backend.database.lifecycle import request_shutdown

    # Signal shutdown immediately
    request_shutdown()

    # Show message with longer timeout (inference can take 30-90s)
    self.notify("Just a sec! (closing up shop)", timeout=10)

    # Graceful shutdown (current task will complete, next task will abort)
    self._graceful_shutdown()
    self.app.exit()
```

### Layer 5: Consumer Timeout Clarification

The timeout in `stop_huey_consumer` is for the **thread join**, not inference. Inference runs to completion regardless.

**File**: `backend/services/queue.py` (update docstring only)

```python
def stop_huey_consumer(timeout: float = 5.0) -> None:
    """Stop the in-process Huey consumer (best-effort, bounded wait).

    Note: This timeout bounds the wait for the consumer thread to join after
    stop() is called. It does NOT bound LLM inference time. If a task is
    mid-inference when shutdown is requested, inference completes first,
    then the task checks the cancellation flag and exits.
    """
    ...
```

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/database/lifecycle.py` | Add `request_shutdown()` function, update `__all__` |
| `backend/services/tasks.py` | Add `TaskCancelled`, `check_cancellation()`, checkpoints in both tasks |
| `backend/services/queue.py` | Add imports, override `run()` in InProcessConsumer, update docstring |
| `frontend/screens/home_screen.py` | Call `request_shutdown()`, increase notify timeout to 10s |

---

## What This Achieves

1. **Current task completes** - LLM inference runs to finish (unavoidable)
2. **Next task aborts** - Checks flag at start, never begins
3. **Orchestrator stops** - Won't reschedule itself during shutdown
4. **No race condition** - `notify_interrupted_tasks()` removed; tasks clean up in finally blocks
5. **No duplicate flags** - Reuses existing `_shutdown_requested` in lifecycle.py
6. **Proper exception order** - `TaskCancelled` caught before `Exception`, so cancelled tasks don't get marked "dead"
7. **User feedback** - 10s notification covers most inference durations

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| No tasks running | Exits in <1 second |
| Task in LLM inference | Waits for inference, then exits |
| Multiple queued tasks | Current finishes, orchestrator stops, others stay pending |
| Task between phases | Aborts immediately at next checkpoint |
| User quits during model load | Checkpoint before load, exits quickly |
| Orchestrator mid-work | Catches TaskCancelled, doesn't reschedule |
