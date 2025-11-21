# Huey Orchestrator Open Issues (21 Nov 2025)

These are blocking gaps observed after reviewing `plans/huey_implementation_plan.md` against the architecture goals in `plans/huey_orchestrator.md` and `V2_PLAN.md`. Each item includes code‑level evidence for follow‑up fixes.

Issues are ordered by priority and dependency: foundational contract changes come first, then dependent fixes, then testing infrastructure.

---

## 1) Worker/producer Redis split & port collision risk

- **Problem**: The Huey consumer is launched in a separate process that re-initializes FalkorDB/Redis, even though the TUI already started an embedded Redis with a TCP listener enabled. This can spawn a second redis-server (port 6379 by default) or point Huey at a different DB file, so enqueued jobs may never be seen by the worker or the worker may crash on startup.
- **Evidence**:
  - TCP listener enabled by default: `backend/settings.py:11-17`.
  - `_init_db` spins up a redis-server using that port each time a process calls `_ensure_graph`: `backend/database/lifecycle.py:204-227`.
  - The worker process imports `backend.services.queue`, which calls `_ensure_graph` when the module loads: `backend/services/queue.py:15-48`.
  - The TUI starts the Huey consumer as a *separate process* (`subprocess.Popen`), not a thread: `charlie.py:478-534`. That new process will repeat the DB init path above, racing or conflicting with the already running redis-server.
- **Impact**: Tasks can be orphaned (producer writes to one Redis, worker listens to another) or the worker fails to start due to "address already in use", leaving background inference non-functional.
- **Follow-up note**: Turning off the TCP listener does **not** remove the split-brain risk. Even in default redislite/unix-socket mode, each process (TUI vs. Huey worker subprocess) initializes its *own* FalkorDB instance when importing `_ensure_graph()`. The TUI's Huey producer and the worker's Huey consumer can therefore point at different embedded redis servers, so jobs are enqueued to one broker and consumed from another. Shutdown in either process can also kill the server the other relies on. The TCP-enabled case adds port-collision failures; the underlying hazard is per-process FalkorDB init, not just TCP exposure.

---

## 2) Redis episode state tracking contract is incomplete (FOUNDATIONAL)

**Priority**: HIGH - This is a foundational contract change that issues #3 and #4 depend on.

- **Problem**: The current Redis episode lifecycle deletes hashes on completion and leaves them in place on failure, creating ambiguity between "work complete" (hash absent) vs "work failed" (hash stuck in `pending_nodes`) vs "work pending" (hash in `pending_nodes`). Deletion also breaks Redis/graph sync when episodes are deleted from the graph but Redis keys remain, causing worker crash loops.
- **Evidence**:
  - Deletion path doesn't clean up Redis: UI delete calls `delete_episode` only; no Redis cleanup: `charlie.py:286-295`. `delete_episode` deletes just the Episodic node: `backend/database/persistence.py:354-387`.
  - Redis keys persist after graph deletion: `set_episode_status` + `enqueue_pending_episodes` will re-enqueue `pending_nodes` even if the episode is gone: `backend/database/redis_ops.py:103-200,232-240`.
  - Success path deletes hash implicitly: When extraction succeeds with no entities, `remove_episode_from_queue()` deletes the hash: `backend/services/tasks.py:88`.
  - Success path with entities leaves hash in `pending_edges`: Extraction with entities sets status to `pending_edges` then stops (extract_edges_task is TODO), leaving the hash indefinitely: `backend/services/tasks.py:73-86`.
  - Failure paths leave hash in `pending_nodes`: Exception handler logs and re-raises without status update: `backend/services/tasks.py:100-102`. Inference-disabled path returns early without status update: `backend/services/tasks.py:53-55`.
  - Runtime proof: `logs/charlie.log` on 2025-11-21 11:22:41–11:22:51 shows repeated `graphiti_core.errors.NodeNotFoundError` for episode UUIDs that had been deleted, indicating Redis still held those queue keys.
- **Impact**:
  - Worker crash loops when pulling deleted episodes from Redis queue.
  - No distinction between "failed" and "pending" episodes - both stuck in `pending_nodes`.
  - No way to query "which episodes are complete?" - must rely on hash absence.
  - Graph drift: MENTIONS edges are removed with the episode, but extracted entity nodes remain orphaned; future dedupe/edges can anchor to stale nodes.
  - Blocks model unload (see issue #3) because stuck hashes mis-signal active work.

---

### Required Fixes for Issue #2 (State Tracking Contract)

**Core principle**: Redis hash lifetime must match graph episode lifetime. Terminal states must be explicit, not implicit (hash deletion).

#### 2.1) Add terminal state support

**Change**: Stop using hash deletion to signal completion. Add explicit terminal statuses.

**New states**:
- `pending_nodes` - Active: waiting for entity extraction
- `pending_edges` - Active: waiting for edge extraction (future)
- `done` - Terminal: processing complete successfully
- `dead` - Terminal: processing failed with unrecoverable error

**Task updates**:
- No-entity path: `remove_episode_from_queue()` → `set_episode_status(uuid, "done")` (`backend/services/tasks.py:88`)
- Entity extraction success: Keep current `set_episode_status(uuid, "pending_edges", uuid_map=...)` (`backend/services/tasks.py:76`)
- Edge extraction complete (future): `set_episode_status(uuid, "done")`
- Unrecoverable errors (e.g., `NodeNotFoundError`): `set_episode_status(uuid, "dead")` then return/raise
- Exception handler: Catch exceptions, set `status=dead`, then re-raise (`backend/services/tasks.py:100-102`)

#### 2.2) Sync Redis with graph deletion

**Change**: Delete Redis hash whenever graph episode is deleted.

**Implementation**:
- `delete_episode()`: Add `remove_episode_from_queue(episode_uuid)` call after graph delete: `backend/database/persistence.py:354-387`.
- UI delete path: Already calls `delete_episode()`, will inherit Redis cleanup: `charlie.py:286-295`.
- Make idempotent: `remove_episode_from_queue()` should silently succeed if hash already absent.

#### 2.3) Update cleanup logic to ignore terminal states

**Change**: `cleanup_if_no_work()` should only block on active work states.

**Implementation** (`backend/inference/manager.py:46-63`):
```python
def cleanup_if_no_work() -> None:
    """Unload models if no active work remains (event-driven cleanup)."""
    from backend.database.redis_ops import get_episodes_by_status

    pending_nodes = get_episodes_by_status("pending_nodes")
    pending_edges = get_episodes_by_status("pending_edges")

    # Ignore terminal states (done, dead) - those are historical records
    if len(pending_nodes) == 0 and len(pending_edges) == 0:
        logger.info("No active work in queue, unloading models")
        unload_all_models()
    else:
        logger.debug(
            "Active work remains (%d pending_nodes, %d pending_edges), keeping models loaded",
            len(pending_nodes),
            len(pending_edges),
        )
```

#### 2.4) Graph orphan pruning

**Change**: When deleting an episode, only delete entities that are no longer referenced.

**Implementation** (`backend/database/persistence.py:354-387`):
- Remove MENTIONS edges from episode to entities
- For each entity: count remaining incoming MENTIONS edges (journal-scoped)
- Delete entity only if MENTIONS count = 0
- Prevents collateral damage when multiple episodes reference the same entity

#### 2.5) Status contract tests

**New tests required** (`tests/test_backend/test_services_tasks.py`):
- No-entity extraction → `status=done`, hash retained
- Entity extraction → `status=pending_edges` with `uuid_map`
- Edge completion (future) → `status=done`
- Episode deletion → hash removed from Redis
- Exception during extraction → `status=dead`
- Unrecoverable error (NodeNotFoundError) → `status=dead`

---

## 3) Cleanup deadlock and inference-disabled handling (DEPENDS ON ISSUE #2)

**Priority**: MEDIUM - Blocked by issue #2's state tracking refactor.

**Prerequisite**: Issue #2 must be implemented first (terminal states + cleanup logic changes).

- **Remaining problems after issue #2 is fixed**:
  1. **Inference disabled**: Episodes stay in `pending_nodes` when inference is disabled, but `cleanup_if_no_work()` doesn't distinguish between "blocked by disabled inference" vs "active work". Models stay loaded even though no work will run.
  2. **No re-enqueue on inference enable**: When inference is toggled back on, pending episodes are not automatically re-enqueued.

- **Evidence**:
  - Task early-return when inference is off leaves status unchanged: `backend/services/tasks.py:53-55`.
  - `cleanup_if_no_work` checks for any `pending_nodes` but doesn't check if inference is enabled: `backend/inference/manager.py:46-63`.
  - No listener for inference toggle changes to trigger `enqueue_pending_episodes()`.

- **Impact**: Models stay loaded indefinitely when inference is disabled, wasting memory for work that will never run.

### Required Fixes for Issue #3 (After Issue #2)

#### 3.1) Update cleanup logic to check inference state

**Change**: `cleanup_if_no_work()` should treat `pending_nodes` as non-blocking when inference is disabled.

**Implementation** (`backend/inference/manager.py:46-63`):
```python
def cleanup_if_no_work() -> None:
    """Unload models if no active work remains (event-driven cleanup)."""
    from backend.database.redis_ops import get_episodes_by_status, get_inference_enabled

    # If inference is disabled, pending episodes are blocked, not active
    if not get_inference_enabled():
        logger.info("Inference disabled, treating pending episodes as blocked work")
        logger.info("Unloading models - they will reload when inference is re-enabled")
        unload_all_models()
        return

    pending_nodes = get_episodes_by_status("pending_nodes")
    pending_edges = get_episodes_by_status("pending_edges")

    if len(pending_nodes) == 0 and len(pending_edges) == 0:
        logger.info("No active work in queue, unloading models")
        unload_all_models()
    else:
        logger.debug(
            "Active work remains (%d pending_nodes, %d pending_edges), keeping models loaded",
            len(pending_nodes),
            len(pending_edges),
        )
```

#### 3.2) Auto-enqueue on inference enable

**Change**: When inference is toggled ON in settings UI, call `enqueue_pending_episodes()` to resume work.

**Implementation** (`charlie.py` SettingsScreen):
- After `set_inference_enabled(True)`, call `enqueue_pending_episodes()` to resume blocked work
- This triggers workers to re-process episodes that were waiting during the disabled period

#### 3.3) Tests

**New tests required**:
- Inference disabled + pending episodes → models unload immediately
- Inference re-enabled → `enqueue_pending_episodes()` called, work resumes
- Models reload on first task after re-enable

---

## 4) Queue/integration behavior untested (only call_local)

**Priority**: LOW - Testing infrastructure to validate fixes for issues #1, #2, and #3.

- **Problem**: Tests never run a real Huey consumer or cross-process queue; they invoke `.call_local()` directly and patch Redis/model layers, so producer/consumer wiring, connection sharing, and startup collisions are unverified.
- **Evidence**:
  - Task tests use `extract_nodes_task.call_local(...)` throughout: `tests/test_backend/test_services_tasks.py:14-258`.
  - Queue tests assert pool extraction but never start a consumer or enqueue jobs: `tests/test_backend/test_services_queue.py:10-55`.
- **Impact**: The critical inter-process bugs (Redis split, port contention, lost jobs, state sync) are not covered by the test suite; failures will surface only at runtime.

### Required Fixes for Issue #4 (Integration Testing)

**New integration tests required**:
- Start real Huey consumer in subprocess
- Enqueue tasks from main process
- Verify tasks execute in worker process
- Verify Redis connection sharing (no split-brain)
- Verify port collision handling (issue #1)
- Verify episode deletion cleans up Redis (issue #2.2)
- Verify orphan pruning retains shared entities (issue #2.4)
- Verify `enqueue_pending_episodes` skips deleted episodes (issue #2.2)
- Verify model unloads when queues empty (issue #2.3)
- Verify cleanup respects inference disabled state (issue #3.1)

---

These issues should be addressed in order (1 → 2 → 3 → 4) before integrating additional graph modules to ensure the orchestrator reliably drives background inference.
