# Huey Orchestrator Implementation Plan

**Goal**: Add async task queue to Charlie for non-blocking LLM inference with model persistence during active work.

**Architecture**: See `plans/huey_orchestrator.md` for detailed rationale on thread safety, model persistence, and resource management.

**Current Scope (Nov 21, 2025)**: Only `extract_nodes` is implemented. Episodes advance to `pending_edges` as a staging marker, but no edge worker exists yet. `cleanup_if_no_work` deliberately ignores `pending_edges` so models can unload; reinstate the pending_edges check once `extract_edges_task` is added.

## Prerequisite: Lock DSPy Cache Location (simple, importable helper)

- Add `backend/dspy_cache.py`:
  - sets `DSPY_CACHEDIR`/`DSPY_CACHE_DIR`/`DSPY_CACHE` to `backend/prompts/.dspy_cache`
  - creates the directory if missing
  - exposes `CACHE_DIR` for debugging
- Import `backend.dspy_cache` before any `import dspy` in entrypoints (Huey worker, CLI, tests). This keeps cache writes inside `backend/prompts` without runtime hacks or env-file coupling. A single `rm -rf backend/prompts` wipes DSPy cache, GEPA artifacts, and prompt JSONs.
- Delete any `pipeline._dspy_setup` imports when copying v1 files; the new helper replaces that hack.

**Status**: ✅ COMPLETE (prerequisite already existed)

---

## Implementation Status

### Batch 1: COMPLETE ✅

**Completed Phases**: Phase 1 (Model Factory), Phase 7 (Settings), Phase 2 (Model Manager)

**Files Created:**
- `backend/inference/__init__.py`
- `backend/inference/dspy_lm.py`
- `backend/inference/loader.py`
- `backend/inference/manager.py`
- `tests/test_backend/test_inference.py`
- `tests/test_backend/test_inference_manager.py`

**Files Modified:**
- `backend/settings.py` - Added model and Huey configuration
- `tests/test_backend/conftest.py` - Updated to use `backend.inference`, added `reset_model_manager` fixture

**Enhancements Beyond Plan:**
1. **Parameter Naming**: Renamed `model_path` parameter to `repo_id` throughout for clarity
2. **Settings Centralization**: All model settings properly centralized in `backend/settings.py`:
   - `MODEL_REPO_ID` - HuggingFace repository
   - `MODEL_QUANTIZATION` - Quantization level (separately adjustable)
   - `LLAMA_CPP_N_CTX`, `LLAMA_CPP_GPU_LAYERS` - Hardware settings (from env vars)
   - `LLAMA_CPP_VERBOSE` - Verbose logging flag
   - `MODEL_CONFIG` - Generation parameters
   - `HUEY_WORKER_TYPE`, `HUEY_WORKERS` - Task queue config
3. **Runtime Validation**: Added `ValueError` for invalid `model_type` in `get_model()`
4. **Comprehensive Test Coverage**:
   - 9 unit tests for manager (mocked, fast)
   - 7 real inference tests for manager (with actual model loading/unloading)
   - 4 unit tests for loader/dspy_lm (mocked)
   - 4 real inference tests for loader/dspy_lm
   - Total: 24 tests, all passing

**Migration Notes:**
- Old `inference_runtime` module remains for backward compatibility with pipeline optimizers
- New `backend.inference` module is the canonical implementation for Huey/app use
- Tests updated to use `backend.inference` and `backend.settings.MODEL_REPO_ID`

---

### Batch 2: COMPLETE ✅

**Completed Phases**: Phase 3 (Queue & Tasks), Phase 4 (Inference Toggle - partial)

**Files Created:**
- `backend/services/__init__.py`
- `backend/services/queue.py`
- `backend/services/tasks.py`
- `tests/test_backend/test_services_queue.py`
- `tests/test_backend/test_services_tasks.py`

**Files Modified:**
- `backend/database/redis_ops.py` - Added `get_inference_enabled()` and `set_inference_enabled()`
- `pyproject.toml` - Added `huey>=2.5.4` dependency

**Implementation Details:**
1. **Huey Queue Configuration** (`backend/services/queue.py`):
   - Extracts FalkorDB Redis connection pool via `_get_redis_connection()`
   - Creates `RedisHuey` instance reusing embedded Redis (no separate broker)
   - Comprehensive error handling for db/client/pool unavailability

2. **Task Implementation** (`backend/services/tasks.py`):
   - `extract_nodes_task()` - Entity extraction with state transition to pending_edges
   - Checks episode status before processing (safe to enqueue multiple times)
   - Respects inference toggle via `get_inference_enabled()`
   - Uses model persistence via `get_model("llm")` from manager
   - Calls `cleanup_if_no_work()` for event-driven model unload
   - Transitions episode to pending_edges with uuid_map persistence (ready for future edge extraction)

3. **Inference Toggle & Recovery** (`backend/database/redis_ops.py`):
   - `get_inference_enabled()` - Returns True if enabled (default: True)
   - `set_inference_enabled(enabled: bool)` - Persists to Redis at `app:inference_enabled`
   - `enqueue_pending_episodes()` - Scans and enqueues pending work when inference is enabled
   - Setting survives app restarts via Redis storage
   - Simplified state management: scan-based lookups; current pipeline has only the entity-extraction stage (more stages can be added later)

4. **Test Coverage**: 15 passing unit tests
   - 5 tests for queue configuration and error handling
   - 10 tests for task behavior, lifecycle, recovery, and scan consistency
   - Tests use `task.call_local()` to execute tasks synchronously
   - All mocking uses correct module paths for local function imports

**Code Review Notes:**
- Initial review suggested `AsyncRedisHuey` which doesn't exist in Huey
- `RedisHuey` correctly supports async tasks via standard asyncio integration
- All other review suggestions confirmed implementation quality

---

### Batch 3: COMPLETE ✅

**Completed Phases**: Phase 4 (Settings UI Integration - complete), Phase 6 (Application Lifecycle), Phase 5 (Optimizer Documentation)

**Files Created:**
- None (documentation phase only for Phase 5)

**Files Modified:**
- `charlie.py` - Added worker lifecycle management and settings UI integration:
  - `_ensure_huey_worker_running()` - Auto-start Huey worker with proper error handling
  - `_shutdown_huey()` - Graceful worker shutdown
  - `SettingsScreen` - Inference toggle with Redis persistence
  - File descriptor leak fix in worker startup exception handler (charlie.py:531-533)
- `backend/__init__.py` - Integration with inference toggle for task enqueueing

**Implementation Details:**
1. **Worker Lifecycle Management** (`charlie.py`):
   - Automatic worker startup in `_init_and_load()` after database initialization
   - Proper command-line argument building with `-k thread -w 1` (thread safety)
   - Log file redirection with buffering for real-time output
   - Graceful shutdown with SIGINT, 10-second timeout, then SIGTERM fallback
   - File descriptor cleanup on startup errors (prevents resource leaks)

2. **Settings UI Integration** (`charlie.py`):
   - `SettingsScreen` with inference toggle linked to Redis
   - Loads initial state from `get_inference_enabled()`
   - Persists changes via `set_inference_enabled()`
   - Toggle survives app restarts (Redis-backed storage)

3. **Task Enqueueing Logic** (`backend/__init__.py`):
   - Episodes always get status set in Redis (enables discovery/recovery)
   - Tasks only enqueued to Huey when inference enabled (responsive UI)
   - Background processing without blocking TUI

4. **Architecture Validation**:
   - Single-stage processing confirmed
   - Thread safety verified: `-k thread -w 1` ensures sequential execution
   - Event-driven cleanup confirmed: `cleanup_if_no_work()` in task `finally` blocks
   - No file descriptor leaks: proper cleanup in exception paths

**Test Coverage**: All existing tests pass
- Worker startup tests verify proper lifecycle management
- Settings persistence tests confirm Redis integration
- File descriptor leak prevention test added (test_charlie.py)

---

## Phase 1: Model Factory (Stateless)

**Purpose**: Shared model loading infrastructure for app and optimizer scripts.

### Files to Create

#### Copy `inference_runtime/` → `backend/inference/`

```
backend/inference/
├── __init__.py         # Copy from inference_runtime/__init__.py
├── dspy_lm.py          # Copy from inference_runtime/dspy_lm.py
└── loader.py           # Copy from inference_runtime/loader.py
```

**Changes required after copy:**
- Remove any `from pipeline import _dspy_setup` imports (cache path is handled by `backend.dspy_cache`).
- `dspy_lm.py`: Change `from settings import MODEL_CONFIG` → `from backend.settings import MODEL_CONFIG`
- `loader.py`: Change `from settings import LLAMA_*` → `from backend.settings import LLAMA_*`

**What it provides:**
- `DspyLM` class (dspy.BaseLM implementation wrapping llama.cpp)
- `load_model()` function (HuggingFace auto-download)
- Used by both manager (Phase 2) and optimizer scripts (Phase 4)

---

## Phase 2: Model Manager (Stateful)

**Purpose**: Model persistence management - keep models loaded in memory while work remains in queue.

### Files to Create

#### `backend/inference/manager.py`

**Core functionality:**
```python
# Global registry (safe with single worker thread)
MODELS = {
    'llm': {'model': None},
}

def get_model(model_type='llm') -> DspyLM:
    """Load model if not cached, return instance."""
    # Check cache → return if loaded (warm)
    # If not loaded → instantiate DspyLM() → cache → return (cold)

def unload_all_models():
    """Unload all models to free memory."""
    # Set all models to None → gc.collect()

def cleanup_if_no_work():
    """Unload models if no pending episodes remain (event-driven cleanup)."""
    # Check Redis for pending_nodes (and pending_edges for future expansion)
    # If queues empty → unload_all_models()
```

**Key points:**
- NO locking needed (single worker thread)
- NO timestamps needed (work queue determines unload)
- Models stay loaded while work remains in queue (model persistence)
- Event-driven cleanup (called at end of each task)
- Simple rule: No work in queue = unload immediately
- Current scope: only the instruct LLM is supported; adding embedding/reranker models will be a follow-on design.

**Used by:** Huey tasks (Phase 3)

**NOT used by:** Optimizer scripts (they use `backend.inference` directly)

---

## Phase 3: Queue & Tasks

**Purpose**: Async task queue integrated with existing FalkorDB Redis.

### Files to Create

#### `backend/services/__init__.py`
- Empty package marker

#### `backend/services/queue.py`

**Core functionality:**
```python
def _get_redis_connection() -> ConnectionPool:
    """Extract connection pool from FalkorDB (_db.client)."""
    # Call _ensure_graph() to trigger DB init
    # Return _db.client.connection_pool

huey = RedisHuey(
    'charlie',
    connection_pool=_get_redis_connection(),
)
```

**Key points:**
- Reuses FalkorDB's embedded Redis (no separate broker)
- RedisHuey supports async tasks via asyncio integration
- Must initialize AFTER database is ready
- Single Huey consumer with one worker; no per-model workers in this iteration.

#### `backend/services/tasks.py`

**Core functionality:**
```python
@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    """Background entity extraction (current implemented stage).

    # Single-stage processing
    Processes episode and removes from queue immediately upon completion.
    """
    from backend.database.redis_ops import (
        get_episode_status,
        get_inference_enabled,
        remove_episode_from_queue,
    )
    from backend.inference.manager import get_model, cleanup_if_no_work
    from backend.graph.extract_nodes import extract_nodes
    import dspy

    # Idempotency check
    if get_episode_status(episode_uuid) != "pending_nodes":
        return {"already_processed": True}

    # Respect inference toggle
    if not get_inference_enabled():
        return {"inference_disabled": True}

    lm = get_model('llm')
    with dspy.context(lm=lm):
        result = extract_nodes(episode_uuid, journal)

    # Single-stage processing: remove from queue when done
    remove_episode_from_queue(episode_uuid)
    cleanup_if_no_work()
    return result
```

**Key points:**
- Single-stage processing: `pending_nodes` → removed from queue
- Tasks call `get_model()` from manager (model persistence)
- Tasks call `cleanup_if_no_work()` at end (event-driven unload)
- NO periodic cleanup task needed (work queue drives unload)
- Use `backend.database.redis_ops` for episode status management

### Integration with Existing Code

**No changes needed to:**
- `backend/database/redis_ops.py` - episode status functions already exist
- `backend/graph/extract_nodes.py` - works as-is, just needs dspy.context() set

**Optional enhancement to `backend/__init__.py`:**
```python
# In add_journal_entry(), after set_episode_status():
from backend.services.tasks import extract_nodes_task
extract_nodes_task(episode_uuid, journal)  # Enqueue immediately
```

**Alternative:** Rely on worker startup recovery to discover pending episodes (lazy).

---

## Phase 4: Settings UI Integration

**Purpose**: Replace placeholder toggles with functional inference controls.

### Queue System Architecture

**Two-layer design with clear separation of concerns:**

**Layer 1: Redis (State Tracker)**
- Tracks episode lifecycle state: `pending_nodes` → [removed when complete]
- **Architecture**: Single-stage processing
- Provides discovery for UI, manual operations, crash recovery
- Source of truth for "what needs processing"

**Layer 2: Huey (Work Executor)**
- Executes tasks asynchronously (non-blocking UI)
- Provides task isolation, error handling, worker management
- Checks Redis state for idempotency (safe to enqueue multiple times)

**Coordination model:**
```
add_journal_entry()
  ↓
  set_episode_status(uuid, "pending_nodes")  # State: needs work
  ↓
  if get_inference_enabled():
      extract_nodes_task(uuid, journal)  # Execution: do work

extract_nodes_task():
  ↓
  Check Redis: is episode still "pending_nodes"?
  ↓
  If not → already processed → exit
  ↓
  If yes AND get_inference_enabled():
      Run extraction → remove from queue (current stage complete)
  ↓
  If yes BUT inference disabled:
      Exit (episode remains in Redis for later)
```

**Key insight:** Redis and Huey serve different purposes - not redundant, but complementary. The queue state is owned by Redis; Huey executes whatever stages exist today (currently entity extraction) and clears the Redis hash when that episode’s work is done.

### Files to Modify

#### `backend/database/redis_ops.py`

**New functions to add:**
```python
def set_inference_enabled(enabled: bool) -> None:
    """Enable/disable inference globally. Persisted across restarts."""

def get_inference_enabled() -> bool:
    """Get current inference enabled status. Default: True."""
```

**Key points:**
- Use Redis for persistence (survives app restarts)
- Setting stored under `app:inference_enabled`
- Default: inference enabled (True)
- No timeout setting needed (auto-unload is work-queue driven)

#### `charlie.py` - Settings Modal

**Update SettingsModal:**
```python
class SettingsModal(ModalScreen):
    def compose(self) -> ComposeResult:
        # Replace placeholder toggles with:
        # - "Enable Inference" toggle (controls task enqueueing)
        # Load initial state from redis_ops.get_inference_enabled()
        pass

    def on_switch_changed(self, event) -> None:
        # Save to Redis via set_inference_enabled()
        pass
```

**Key points:**
- Single toggle only: "Enable Inference"
- Load initial state on modal open
- Save state change to Redis immediately
- No timeout configuration needed (system auto-manages memory)

#### `backend/__init__.py`

**Update add_journal_entry:**
```python
async def add_journal_entry(...) -> str:
    # ... existing code ...

    # Set episode state in Redis (always)
    set_episode_status(episode_uuid, "pending_nodes", journal=journal)

    # Enqueue task only if inference enabled
    if get_inference_enabled():
        from backend.services.tasks import extract_nodes_task
        extract_nodes_task(episode_uuid, journal)

    return episode_uuid
```

**Key points:**
- Redis state always set (enables manual re-enrichment later)
- Huey task only enqueued if inference enabled (responsive)
- No work happens if inference disabled (episodes wait in Redis)

#### `backend/services/tasks.py`

**Single-stage task implementation:**
```python
@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    from backend.database.redis_ops import (
        get_episode_status,
        get_inference_enabled,
        remove_episode_from_queue,
    )
    from backend.inference.manager import cleanup_if_no_work

    # Check Redis state: is this episode still pending?
    current_status = get_episode_status(episode_uuid)
    if current_status != "pending_nodes":
        # Already processed by another task
        return {'already_processed': True}

    # Check if inference still enabled (may have changed since enqueue)
    if not get_inference_enabled():
        # Leave episode in pending_nodes for later
        return {'inference_disabled': True}

    # Proceed with extraction (current implemented stage)
    result = extract_nodes(episode_uuid, journal)

    # Single-stage: remove from queue when done
    remove_episode_from_queue(episode_uuid)
    cleanup_if_no_work()

    return result
```

**Key points:**
- Tasks check Redis state first (idempotent - safe to enqueue multiple times)
- Tasks check inference toggle before processing (respects user intent)
- Tasks call `cleanup_if_no_work()` at end (event-driven unload)
- No periodic cleanup task needed (work queue drives memory management)

#### `backend/inference/manager.py`

**Functions needed:**
```python
def unload_all_models():
    """Unload all models to free memory."""
    # Set all model references to None
    # Call gc.collect()

def cleanup_if_no_work():
    """Unload models if no pending episodes remain (event-driven cleanup)."""
    from backend.database.redis_ops import get_episodes_by_status

    pending_nodes = get_episodes_by_status("pending_nodes")
    pending_edges = get_episodes_by_status("pending_edges")  # For future expansion

    if len(pending_nodes) == 0 and len(pending_edges) == 0:
        unload_all_models()
```

**Key points:**
- No timeout parameter needed
- No timestamp tracking needed
- Simple rule: Empty work queue = unload immediately
- Called by tasks (event-driven, not periodic)

### Manual Re-enrichment Support (Future)

**The architecture supports manual re-triggering:**
```python
# Future UI feature: "Re-enrich this entry" button
def trigger_manual_enrichment(episode_uuid: str, journal: str):
    # Reset state in Redis
    set_episode_status(episode_uuid, "pending_nodes")

    # Enqueue for processing
    if get_inference_enabled():
        extract_nodes_task(episode_uuid, journal)
    # If inference disabled, episode waits in Redis until toggled on
```

**Key points:**
- Redis state enables discovery of what needs work
- Idempotent tasks make it safe to re-enqueue
- User can manually trigger enrichment regardless of inference toggle
- No special handling needed - existing task flow handles it

### Testing Requirements

**Persistence tests:**
- Settings survive app restart (Redis storage)
- Toggle changes reflected in task behavior

**Idempotency tests:**
- Enqueueing same episode multiple times processes only once
- Redis state prevents double-processing

**Queue coordination tests:**
- Inference OFF: episodes stay in Redis, no processing
- Inference ON: episodes process and advance through states
- Toggle ON→OFF mid-flight: running tasks finish, queued tasks exit early
- Pending work: models stay loaded (no unload during batch)
- Work completes: models unload immediately (event-driven)

**UI tests:**
- Toggle loads correct initial state
- Toggle changes save to Redis
- Single toggle only (no timeout configuration)

---

## Phase 5: Optimizer Integration (Future)

**Purpose**: Document how future DSPy optimizers will integrate.

**NOTE**: Optimizers will NOT be created during this plan. This phase documents the pattern for future implementation.

### Future Files (Not Created Now)

#### `backend/optimizers/extract_nodes_optimizer.py`

**Pattern to follow (future reference):**
```python
from backend.inference import DspyLM  # Import from factory, NOT manager
import dspy

# Load model directly (stateless, no caching)
lm = DspyLM()
dspy.settings.configure(lm=lm)

# Run optimizer (MIPROv2, BootstrapFewShot, etc.)
# Save results to pipeline/prompts/
```

**Key points:**
- Naming pattern: `{operation}_optimizer.py` (matches `pipeline/optimizers/`)
- Import from `backend.inference` (stateless factory)
- DO NOT import from `backend.inference.manager` (stateful, app-only)
- DO NOT import from `backend.services` (queue is app-only)
- Optimizers run standalone, bypass all app infrastructure

### Documentation to Create

#### `backend/optimizers/README.md`

**Contents:**
- Architecture: Optimizers isolated from app runtime
- Shared infrastructure: Model loading via `backend.inference`
- Naming conventions: `extract_nodes_optimizer.py`, etc.
- Usage examples: Import pattern, running scripts
- Configuration: Use root `settings.py` for optimizer config

**Status**: Create README now, add actual optimizers in future work

---

## Phase 6: Application Lifecycle

**Purpose**: Auto-start/stop Huey worker when launching TUI.

### Files to Modify

#### `charlie.py`

**Add to CharlieApp class:**
```python
class CharlieApp(App):
    def __init__(self):
        super().__init__()
        self.huey_process = None

    def on_mount(self):
        """Start Huey worker subprocess."""
        # Prefer console script to avoid runpy double-import warning.
        cmd = ['huey_consumer', 'backend.services.tasks.huey', '-k', 'thread', '-w', '1', '-q']
        if shutil.which('huey_consumer') is None:
            cmd = [sys.executable, '-m', 'huey.bin.huey_consumer', 'backend.services.tasks.huey', '-k', 'thread', '-w', '1', '-q']

        self.huey_process = subprocess.Popen(cmd)
        atexit.register(self._shutdown_huey)

    def _shutdown_huey(self):
        """Terminate worker gracefully."""
        if self.huey_process:
            self.huey_process.send_signal(signal.SIGTERM)
            self.huey_process.wait(timeout=5)

    def on_unmount(self):
        """Cleanup on app exit."""
        self._shutdown_huey()

    def action_quit(self):
        # Stop Huey before shutting down database to prevent redis reconnect noise
        self.stop_huey_worker()
        shutdown_database()
        self.app.exit()
```

**Key points:**
- Worker starts automatically in `on_mount()`
- Graceful shutdown with SIGTERM
- Critical flags: `-k thread -w 1` (model persistence + thread safety)

---

## Phase 7: Settings & Configuration

### Settings to Add

#### `backend/settings.py`

```python
# Model configuration
LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))
LLAMA_CPP_N_CTX = int(os.getenv("LLAMA_CTX_SIZE", "4096"))

MODEL_CONFIG = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 2048,
}

# Huey configuration
HUEY_WORKER_TYPE = 'thread'  # CRITICAL
HUEY_WORKERS = 1             # CRITICAL
```

**Key points:**
- llama.cpp settings control GPU/memory usage
- Huey settings enforce thread safety
- MODEL_CONFIG shared between app and optimizers
- No timeout setting needed (work-queue driven unload)

---

## Implementation Order

1. **Phase 1** (Foundation): Copy inference_runtime → backend/inference, update imports
2. **Phase 7** (Config): Add settings to backend/settings.py
3. **Phase 2** (Manager): Create manager.py with model persistence logic
4. **Phase 3** (Queue): Create queue.py + tasks.py, integrate with existing redis_ops
5. **Phase 4** (Settings UI): Add redis_ops functions, update charlie.py toggles
6. **Phase 6** (Lifecycle): Update charlie.py to spawn worker
7. **Phase 5** (Optimizers): Document patterns, create README (no actual optimizers yet)

**Why this order:**
- Phase 1 must be first (foundation for everything)
- Phase 7 early (settings needed by Phase 2)
- Phase 2 before 3 (tasks need manager)
- Phase 4 after 3 (UI needs task behavior functions)
- Phase 6 after 4 (worker needs settings functions)
- Phase 5 last (documentation only, independent)

---

## Testing Strategy

### Fast Tests (No Model Loading)
- Import validation
- Registry structure checks
- Connection pool extraction
- Mock-based unload logic

### Slow Tests (With Model Loading)
- Model loading and caching
- Full task execution
- Pipeline integration (add_entry → extract_nodes → extract_edges)

### Manual Verification
```bash
# Start worker manually
huey_consumer backend.services.tasks.huey -k thread -w 1 -q

# Check worker running
ps aux | grep huey_consumer

# Monitor Redis state (Layer 1)
python -c "
from backend.database.redis_ops import get_episodes_by_status
print('Pending nodes:', len(get_episodes_by_status('pending_nodes')))
# Note: Single-stage processing - episodes removed from queue when complete
"

# Test settings persistence
python -c "
from backend.database.redis_ops import set_inference_enabled, get_inference_enabled
set_inference_enabled(False)
print('Inference enabled:', get_inference_enabled())
"
```

### UI Testing Scenarios

**Scenario 1: Inference disabled**
- Toggle inference OFF in settings
- Add journal entry
- Verify: Episode in `status:pending_nodes` Redis set
- Verify: No task processing (check worker logs)
- Toggle inference ON
- Verify: Episode processes (manual re-trigger may be needed)

**Scenario 2: Responsive processing**
- Toggle inference ON
- Add journal entry
- Verify: Task starts immediately (check worker logs)
- Verify: Model loads (cold start, 1-2 seconds)
- Add second journal entry quickly
- Verify: Model stays loaded (model persists, instant)

**Scenario 3: Auto-unload (event-driven)**
- Add journal entry (trigger model load)
- Wait for task to complete
- Verify: Model unloads immediately after task (no delay)

**Scenario 4: Batch processing keeps models warm**
- Add 5 journal entries quickly
- Check Redis: 5 episodes in pending_nodes
- Watch episodes process sequentially
- Verify: Model stays loaded through all 5 (model persists)
- After last episode completes
- Verify: Model unloads immediately (event-driven)

**Scenario 5: Settings persistence**
- Toggle inference OFF
- Exit app
- Restart app
- Open settings → verify inference toggle is OFF
- Toggle ON → verify episodes start processing

---

## Critical Requirements

**Thread Safety:**
- Huey MUST use `-k thread -w 1` (single worker thread)
- llama-cpp-python is NOT thread-safe
- No locking needed in manager (sequential execution guaranteed)

**Connection Reuse:**
- Queue MUST use `_get_redis_connection()` to extract FalkorDB's pool
- Do NOT create new Redis connection (causes file locking conflicts)

**Model Persistence:**
- Thread worker type keeps process alive (global state persists)
- Process worker type spawns new process per task (cold start every time)

**Import Discipline:**
- App code: Import from `backend.inference.manager` (stateful)
- Optimizer code: Import from `backend.inference` (stateless)
- Never import manager from optimizer scripts

**Queue Coordination:**
- Redis = State tracker (what needs processing, for UI/discovery/recovery)
- Huey = Work executor (task isolation, error handling, non-blocking)
- Tasks are idempotent (check Redis state, safe to enqueue multiple times)
- Always set Redis state (enables manual re-enrichment later)
- Only enqueue to Huey if inference enabled (responsive, no lag)

---

## Success Criteria

- [x] Models stay loaded while work remains in queue
- [x] TUI remains responsive during inference
- [x] Worker starts/stops automatically with app
- [x] Episode status tracking works end-to-end
- [x] Single inference toggle functional with persistence across restarts
- [x] Inference can be disabled/enabled via UI
- [x] Models unload immediately when work queue empties (event-driven)
- [x] Batch processing keeps models warm (no unload mid-batch)
- [x] Optimizer pattern documented (no implementations created yet)
- [x] No database file locking conflicts
- [x] No periodic cleanup task (simplified architecture)

**ALL SUCCESS CRITERIA MET ✅** - Huey orchestrator implementation complete!
