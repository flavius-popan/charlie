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

## 2) Relationship/edges stage never runs (state machine stalls at pending_nodes)
- **Problem**: V2 requires `extract_nodes` *then* `extract_edges` with Redis state advancing `pending_nodes → pending_edges` → done (see V2 plan), but the implementation only permits `pending_nodes` and removes episodes immediately after node extraction. Edge extraction and uuid_map handoff are impossible.
- **Evidence**:
  - V2 requirement for two-stage progression: `V2_PLAN.md:23-34`.
  - Redis status enforcement only allows `pending_nodes`: `backend/database/redis_ops.py:104-131`.
  - `extract_nodes_task` removes the episode from Redis right after node extraction, with no edge task enqueued: `backend/services/tasks.py:73-78`.
  - `extract_nodes` returns a `uuid_map` needed for edge resolution but it is dropped when the episode is removed: `backend/graph/extract_nodes.py:340-349`.
- **Impact**: No relationships are ever built, so the personal knowledge graph cannot satisfy V2 timelines/relationship queries; uuid_map data is discarded before any edge resolver could run.

## 3) Pending queue can deadlock after failure or while inference is disabled
- **Problem**: If inference is disabled or extraction raises, episodes stay in `pending_nodes` with no retry/enqueue path, keeping the queue non-empty and blocking model unload logic.
- **Evidence**:
  - Task early-return when inference is off leaves status unchanged: `backend/services/tasks.py:53-55`.
  - Exceptions do not clear the status; episode remains pending: `backend/services/tasks.py:86-88` (and test confirms no removal on exception at `tests/test_backend/test_services_tasks.py:200-213`).
  - `cleanup_if_no_work` refuses to unload models while any `pending_nodes` exist, so a single stuck episode keeps the model resident: `backend/inference/manager.py:46-55`.
- **Impact**: Backlog never drains; warm model memory stays allocated indefinitely; pending episodes require manual cleanup or ad-hoc retries.

## 4) Queue/integration behavior untested (only call_local)
- **Problem**: Tests never run a real Huey consumer or cross-process queue; they invoke `.call_local()` directly and patch Redis/model layers, so producer/consumer wiring, connection sharing, and startup collisions are unverified.
- **Evidence**:
  - Task tests use `extract_nodes_task.call_local(...)` throughout: `tests/test_backend/test_services_tasks.py:14-258`.
  - Queue tests assert pool extraction but never start a consumer or enqueue jobs: `tests/test_backend/test_services_queue.py:10-55`.
- **Impact**: The critical inter-process bugs above (Redis split, port contention, lost jobs) are not covered by the test suite; failures will surface only at runtime.

---

These issues should be addressed before integrating additional graph modules to ensure the orchestrator reliably drives background inference.

---

## Follow-up: Redis splitting when TCP is off
- Turning off the TCP listener does **not** remove the split-brain risk. Even in default redislite/unix-socket mode, each process (TUI vs. Huey worker subprocess) initializes its *own* FalkorDB instance when importing `_ensure_graph()` in `backend/services/queue.py:15-48`. The TUI’s Huey producer and the worker’s Huey consumer can therefore point at different embedded redis servers, so jobs are enqueued to one broker and consumed from another. Shutdown in either process can also kill the server the other relies on. The TCP-enabled case adds port-collision failures; the underlying hazard is per-process FalkorDB init, not just TCP exposure. 상세 code refs unchanged (see Issues #1 above).
