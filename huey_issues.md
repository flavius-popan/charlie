# Huey Orchestrator Open Issues (21 Nov 2025)

These are blocking gaps observed after reviewing `plans/huey_implementation_plan.md` against the architecture goals in `plans/huey_orchestrator.md` and `V2_PLAN.md`. Each item includes code‑level evidence for follow‑up fixes.

---

## 1) Worker/producer Redis split & port collision risk
- **Problem**: The Huey consumer is launched in a separate process that re-initializes FalkorDB/Redis, even though the TUI already started an embedded Redis with a TCP listener enabled. This can spawn a second redis-server (port 6379 by default) or point Huey at a different DB file, so enqueued jobs may never be seen by the worker or the worker may crash on startup.
- **Evidence**:
  - TCP listener enabled by default: `backend/settings.py:11-17`.
  - `_init_db` spins up a redis-server using that port each time a process calls `_ensure_graph`: `backend/database/lifecycle.py:204-227`.
  - The worker process imports `backend.services.queue`, which calls `_ensure_graph` when the module loads: `backend/services/queue.py:15-48`.
  - The TUI starts the Huey consumer as a *separate process* (`subprocess.Popen`), not a thread: `charlie.py:478-534`. That new process will repeat the DB init path above, racing or conflicting with the already running redis-server.
- **Impact**: Tasks can be orphaned (producer writes to one Redis, worker listens to another) or the worker fails to start due to "address already in use", leaving background inference non-functional.

## 2) Pending queue can deadlock after failure or while inference is disabled
- **Problem**: If inference is disabled or extraction raises, episodes stay in `pending_nodes` with no retry/enqueue path, keeping the queue non-empty and blocking model unload logic.
- **Evidence**:
  - Task early-return when inference is off leaves status unchanged: `backend/services/tasks.py:53-55`.
  - Exceptions do not clear the status; episode remains pending: `backend/services/tasks.py:86-88` (and test confirms no removal on exception at `tests/test_backend/test_services_tasks.py:200-213`).
  - `cleanup_if_no_work` refuses to unload models while any `pending_nodes` exist, so a single stuck episode keeps the model resident: `backend/inference/manager.py:46-55`.
- **Impact**: Backlog never drains; warm model memory stays allocated indefinitely; pending episodes require manual cleanup or ad-hoc retries.

## 3) Queue/integration behavior untested (only call_local)
- **Problem**: Tests never run a real Huey consumer or cross-process queue; they invoke `.call_local()` directly and patch Redis/model layers, so producer/consumer wiring, connection sharing, and startup collisions are unverified.
- **Evidence**:
  - Task tests use `extract_nodes_task.call_local(...)` throughout: `tests/test_backend/test_services_tasks.py:14-258`.
  - Queue tests assert pool extraction but never start a consumer or enqueue jobs: `tests/test_backend/test_services_queue.py:10-55`.
- **Impact**: The critical inter-process bugs above (Redis split, port contention, lost jobs) are not covered by the test suite; failures will surface only at runtime.

---

## 4) Deletion lifecycle leaves orphan Redis keys and dangling graph nodes
- **Problem**: Deleting a journal entry removes only the graph episode; Redis queue state is left behind, so Huey workers keep pulling nonexistent episodes and crash with `NodeNotFoundError`. Graph cleanup omits orphan pruning for extracted entities/edges.
- **Evidence**:
  - UI delete path calls `delete_episode` only; no Redis/Huey cleanup: `charlie.py:286-295`.
  - `delete_episode` deletes just the Episodic node and warns about missing orphan cleanup: `backend/database/persistence.py:354-387`.
  - Redis queue entries never cleared on delete; `set_episode_status` + `enqueue_pending_episodes` will re-enqueue `pending_nodes` even if the episode is gone: `backend/database/redis_ops.py:103-200,232-240`.
- Extraction success sets status to `pending_edges` then stops (extract_edges_task is TODO), leaving the per-episode Redis hash `episode:<uuid>` (fields: status/journal/uuid_map) in place indefinitely: `backend/services/tasks.py:73-86`.
  - `cleanup_if_no_work` only inspects `pending_nodes`; stuck hashes there block model unload: `backend/inference/manager.py:46-63`.
  - Runtime proof: `logs/charlie.log` on 2025-11-21 11:22:41–11:22:51 shows repeated `graphiti_core.errors.NodeNotFoundError` for episode UUIDs that had been deleted, indicating Redis still held those queue keys.
- **Impact**:
  - Worker crash loops and noisy logs; extraction never progresses for the stuck items.
  - `pending_nodes` hashes keep models resident and mis-signal work.
  - Graph drift: MENTIONS edges are removed with the episode, but extracted entity nodes remain orphaned; future dedupe/edges can anchor to stale nodes.
- **Required fixes**:
  - Delete path: remove `episode:<uuid>` Redis hash (and any task/uuid_map metadata) whenever an episode is deleted (user delete or batch). Idempotent if already gone.
  - Task path: on unrecoverable errors (e.g., `NodeNotFoundError`), drop or mark the Redis key to avoid retry storms; optionally mark `status=dead` for diagnostics.
  - Status completeness: if edge extraction isn’t implemented, `pending_edges` must not be a resting state—either promote to `done` and delete the hash, or add a no-op edge step that clears the queue so the lifecycle finishes cleanly.
  - Redis/graph coupling: the per-episode Redis hash is the authoritative state tracker only while the episode exists; the hash must be deleted whenever the episode is deleted so Redis and FalkorDB stay in sync.
  - Graph pruning: when deleting an episode, remove MENTIONS edges and delete only entities with no remaining incoming MENTIONS (journal-scoped) to avoid collateral damage.
  - Tests: add integration tests that start a real consumer/simulator to prove (a) delete wipes Redis keys, (b) orphan pruning retains entities still referenced elsewhere, (c) `enqueue_pending_episodes` skips absent episodes, and (d) model unloads once queues are empty.

---

These issues should be addressed before integrating additional graph modules to ensure the orchestrator reliably drives background inference.

---

## Follow-up: Redis splitting when TCP is off
- Turning off the TCP listener does **not** remove the split-brain risk. Even in default redislite/unix-socket mode, each process (TUI vs. Huey worker subprocess) initializes its *own* FalkorDB instance when importing `_ensure_graph()` in `backend/services/queue.py:15-48`. The TUI’s Huey producer and the worker’s Huey consumer can therefore point at different embedded redis servers, so jobs are enqueued to one broker and consumed from another. Shutdown in either process can also kill the server the other relies on. The TCP-enabled case adds port-collision failures; the underlying hazard is per-process FalkorDB init, not just TCP exposure. 상세 code refs unchanged (see Issues #1 above).
