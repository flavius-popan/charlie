# Huey Orchestrator Follow‑Up Issues (open)

Context: Redis state management has been simplified. Issues #1 (episodes stuck in pending_edges) and #2 (backlog not processed after toggle) have been resolved by removing the two-layer queue architecture and implementing scan-based lookups with recovery on toggle enable. The items below remain unresolved.

## 1) Worker start error leaves file descriptor open
- **Problem**: `_ensure_huey_worker_running` opens the log file (`logs/charlie.log` via `self.huey_log_file`) before calling `subprocess.Popen`; if `Popen` raises, that handle is left open.
- **Impact**: Minor FD leak; repeated failures could exhaust descriptors.
- **Fix**: Close the file in the exception path (or open with context and detach handle only on successful spawn).

## 2) Plan alignment gaps to confirm
- **Threaded worker guarantees**: Ensure consumer is launched with `-k thread -w 1` (code sets defaults, but deployment scripts should be checked).
- **Warm model unload behavior**: Verify that `cleanup_if_no_work()` unloads models when pending_nodes queue is empty.
