# Huey Orchestrator – Developer README

This document explains how Charlie’s background inference pipeline works today, where to look when something breaks, and which files own each responsibility. No code blocks, just the mental model.

## Architecture at a Glance
- Single process: the Textual TUI starts the embedded FalkorDB/Redis and an in-process Huey consumer thread. There is no external huey_consumer subprocess.
- Two layers:
  - Redis state tracker (inside the FalkorDB lite instance) owns episode lifecycle data and the inference toggle.
  - Huey executor (thread worker, 1 worker, thread type) runs tasks sequentially for safety with llama-cpp.
- Model persistence: `backend/inference/manager.py` keeps models warm in memory while work exists; unloads when no active work or inference is disabled.
- Cache hygiene: `backend/dspy_cache.py` pins DSPy cache to `backend/prompts/.dspy_cache`.

## Key Files and What They Do
- Queue wiring: `backend/services/queue.py` creates the shared `PriorityRedisHuey` instance using the embedded Redis connection and manages the in-process consumer thread (start/stop/status).
- Tasks: `backend/services/tasks.py` defines `extract_nodes_task` (entity extraction). It is idempotent, respects the inference toggle, and supports priority levels (user-triggered=1, background=0).
- State storage: `backend/database/redis_ops.py` stores unified journal hashes (status, journal, uuid_map, nodes), inference toggle, and provides helpers to scan/enqueue pending work.
- Model management: `backend/inference/manager.py` loads/unloads the llama.cpp model and performs event-driven cleanup.
- Loading primitives: `backend/inference/dspy_lm.py` and `backend/inference/loader.py` wrap llama.cpp for DSPy; configuration lives in `backend/settings.py`.
- TUI integration: `charlie.py` starts the consumer thread after DB readiness, exposes the Settings modal toggle, and shuts down the consumer before database teardown.
- Entry ingestion: `backend/__init__.py:add_journal_entry` persists the episode, sets Redis status to pending_nodes, and enqueues the task if inference is enabled.

## Lifecycle of a Journal Entry
1) User saves entry → `add_journal_entry` writes to graph and sets Redis status `pending_nodes` with journal name.
2) If inference is enabled, `extract_nodes_task` is enqueued immediately; otherwise the entry waits in Redis.
3) Task runs (single worker thread):
   - Skips if status != pending_nodes (idempotent).
   - Aborts early if inference is disabled (status stays pending_nodes).
   - Runs `extract_nodes` under `dspy.context` using the warm model.
   - Transitions to `pending_edges` when entities found; to `done` when none; to `dead` on exception.
4) `cleanup_if_no_work` unloads models when inference is off or both pending_nodes and pending_edges are empty.
5) Episode hash remains in Redis (active and terminal) until the episode is deleted from the graph; deletion calls `remove_episode_from_queue`.

## Redis Keys and States
- Unified journal hash `journal:{journal}:{uuid}` holds all episode metadata: status, journal, uuid_map, nodes (extracted entities).
- Valid statuses: pending_nodes, pending_edges, done, dead. pending_edges is still treated as active work even though edge tasks are not implemented yet.
- Global flag `app:inference_enabled` controls enqueueing and task execution.
- Cache is deleted atomically with episode deletion (no orphaned keys).

## Settings and Toggles
- Settings modal switch → `set_inference_enabled` in Redis.
- Turning inference off does not stop the worker; tasks simply no-op and models unload via cleanup.
- Re-enabling inference calls `enqueue_pending_episodes` to pick up any backlog in pending_nodes.

## Worker Lifecycle and Debugging
- Start: `_ensure_huey_worker_running` in `charlie.py` after DB init; uses `start_huey_consumer` (background thread, no signals).
- Stop: `_shutdown_huey`/`stop_huey_consumer` on app exit.
- Health: `is_huey_consumer_running` tells you if the consumer thread is alive. There is no auto-restart watchdog yet.
- Logs: consumer shares the process logger; check `logs/charlie.log`.
- Common checks:
  - Is Redis running? `_db.client.ping()` inside `backend.database.lifecycle`.
  - Are there stuck locks? Huey flush_locks is disabled to avoid cross-process issues; restarting the app clears local locks.
  - Are tasks enqueued? Look for `journal:*` hashes and match counts of pending_nodes vs pending_edges.

## Model Handling
- Configuration: repo/quantization/context/gpu layers in `backend/settings.py`.
- Load path: `load_model` auto-downloads from Hugging Face on first use; cold start may be slow and occurs inside the consumer thread.
- Persistence: single worker thread keeps the model object warm across tasks; cleanup unloads immediately when queue is empty or inference is disabled.
- Cache: DSPy cache stored under `backend/prompts/.dspy_cache` via `backend/dspy_cache.py` on import.

## Tests That Cover This Layer
- Queue wiring: `tests/test_backend/test_services_queue.py`
- Task behavior and cleanup: `tests/test_backend/test_services_tasks.py`
- Consumer lifecycle: `tests/test_backend/test_huey_consumer_inprocess.py`, `tests/test_backend/test_huey_process_spawn.py`
- Manager cleanup logic: `tests/test_backend/test_inference_manager.py`
- Enqueue path from add_journal_entry: `tests/test_backend/test_add_journal_entry.py`
- Frontend toggle and worker integration: `tests/test_frontend/test_charlie.py`

## Debugging Checklist
- Inference seems stalled:
  - Confirm `is_huey_consumer_running` is true.
  - Check `app:inference_enabled` is true.
  - Count pending_nodes in Redis; if non-zero but no task logs, restart the app to recreate the consumer thread.
- Memory pressure or VRAM errors:
  - Verify only one worker is configured (HUEY_WORKERS=1, HUEY_WORKER_TYPE=thread).
  - Ensure inference is disabled when running heavy non-inference tasks; cleanup will unload models.
- Incorrect state transitions:
  - Inspect the `journal:{journal}:{uuid}` hash to see the last status and uuid_map.
  - Re-enqueue by setting status to pending_nodes and calling `enqueue_pending_episodes` (only when inference is enabled).

## Known Limitations and Next Steps
- Edge extraction tasks are not implemented; pending_edges remains a terminal-active state placeholder.
- First-time model download can block the consumer thread; consider prefetching models or surfacing progress.

## Operational Notes
- Redis TCP is currently enabled by default for local debugging; disable or password-protect before release to avoid port conflicts or unintended exposure.
