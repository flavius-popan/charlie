# Local LLM Orchestration & Inference Infrastructure

## Purpose
This infrastructure is designed to enable high-performance, local AI features within a Textual TUI application while strictly managing limited system resources (RAM/VRAM). It solves the critical challenge of running heavy models (LLMs, Embeddings, Rerankers) alongside a user interface without freezing the application or crashing the system due to Out-Of-Memory (OOM) errors.

The design keeps a single worker process alive so models can stay resident during a backlog of tasks and are unloaded as soon as no work remains (no timers, no periodic cleanup).

## Huey's Role
**Huey** serves as the asynchronous orchestration layer that bridges the user interface and the heavy inference engine. Its specific responsibilities are:

*   **Non-Blocking Execution:** Offloads heavy inference tasks (generation, embedding) to a background process, ensuring the TUI remains responsive to user input at all times.
*   **Resource Serialization:** configured with a **single-threaded worker**, Huey acts as a strict FIFO (First-In-First-Out) queue. This guarantees that two heavy models are never triggered simultaneously, preventing resource contention.
*   **State Persistence:** By utilizing the `thread` worker type (rather than `process`), Huey keeps the worker process alive between tasks. This allows the **Model Manager** to keep a model resident while work is queued, then unload immediately when queues are empty.
*   **Infrastructure Efficiency:** Huey integrates directly with the application's existing embedded `redislite`/`falkordblite` database, requiring no external broker services or additional system overhead.

### Phase 1: The Inference Foundation (Stateless)

**Target Files:** `backend/inference/__init__.py`, `backend/inference/dspy_lm.py`, `backend/inference/loader.py`

**Goal:** Create a "Model Factory" layer that knows *how* to instantiate models but holds no opinion on *when* or *how long* they are kept.

**Why:**
*   **Decoupling:** Your DSPy optimizer scripts need raw access to model objects without the overhead of job queues, database connections, or memory management logic.
*   **Reusability:** The exact same loading logic (using paths from `backend/settings.py`) must be shared between the App and the Dev scripts to ensure optimization results are valid in production.

**Structure:**
*   Stateless factory layer: `DspyLM` (DSPy BaseLM) and `load_model()` helpers.
*   Reads configuration from `backend/settings.py`; stores no global state.
*   Optimizers and app share the same loaders to guarantee parity.

### Phase 2: The Orchestrator (Stateful)

**Target File:** `backend/inference/manager.py`

**Goal:** Create a "Model Manager" layer that manages the lifecycle of models within the application's runtime.

**Why:**
*   **Memory Management:** A local machine cannot hold an Instruct Model, an Embedding Model, and a Reranker in VRAM simultaneously. The system must enforce "Exclusive Access"—automatically unloading one model to make room for another.
*   **Latency Reduction:** Loading a model from disk is slow. The Manager keeps the model in memory while tasks remain and unloads when queues are empty to free resources.
*   **Abstraction:** The consumer tasks shouldn't care about memory constraints. They should simply request "Instruct," and the Manager handles loading and immediate cleanup when no work remains.

**Current scope:** Only the instruct model is implemented. Embedding and reranker support will be added in a later iteration.

**Structure:**
*   Imports the factory utilities from `backend/inference`.
*   Maintains a global registry of currently loaded models (currently only `llm`).
*   Loads on demand; tasks trigger `cleanup_if_no_work()` to unload everything when Redis queues are empty.
*   **Crucially:** This module is *never* imported by Dev/Optimizer scripts. It is exclusive to the Application/Huey worker.

### Phase 3: The Service Layer (Asynchronous Queue)

**Target Files:** `backend/services/queue.py` and `backend/services/tasks.py`

**Goal:** Integrate a lightweight task queue that shares the existing embedded infrastructure.

**Why:**
*   **Non-Blocking UI:** Inference takes time. The TUI must remain responsive while the LLM generates text or embeddings.
*   **Resource Safety:** By using a single-threaded worker, you strictly serialize inference operations. This prevents the application from accidentally trying to run two heavy models at once, which would crash the application (OOM).
*   **Infrastructure Reuse:** You are already using `redislite`. The queue must connect to this *existing* socket rather than spinning up a new Redis instance, which would cause file contention and locking issues.

**Structure:**
*   `backend/services/queue.py` bridges the `huey` library with your existing `backend/database/lifecycle.py` module. It must access the existing connection pool to ensure it talks to the same embedded database file defined in `backend/settings.py`.
*   `backend/services/tasks.py` defines the specific jobs that wrap `backend/graph/*` operations (currently `extract_nodes_task` - single-stage processing).
*   These tasks call graph operation functions (like `backend/graph/extract_nodes.py::extract_nodes()`) which internally use `backend/inference/manager.py` to retrieve models.
*   Tasks do NOT import the Factory directly - all model access flows through the Manager.

### Phase 3.5: Episode Status Management (Worker Orchestration)

**Target File:** `backend/database/redis_ops.py`

**Goal:** Enable workers to discover episodes that need processing without knowing UUIDs ahead of time, supporting batch operations, startup recovery, and status monitoring.

**Why:**
*   **Worker Discovery:** Workers need to find episodes in specific states (pending_nodes) for batch processing and retry logic.
*   **State Tracking:** Episodes move through processing states then are removed from Redis when complete.
*   **Simplified Architecture:** Single-stage processing (original two-stage design was simplified).
*   **Operational Visibility:** Status monitoring for dashboards, recovery after crashes.

#### Redis Key Structure

```
# Per-episode metadata (Redis Hash) - ONLY for episodes needing processing
episode:{uuid} → Hash {
    "status": "pending_nodes",
    "journal": DEFAULT_JOURNAL,
    "uuid_map": "{...json...}"  # Optional, stores entity resolution mapping
}
```

**Design Rationale:**
- O(1) status lookups by UUID via Hash
- O(n) scanning by status via SCAN (n = episodes in queue, bounded by backlog)
- Single source of truth: episode hash contains all state
- **Single-stage processing**: Only `pending_nodes` status used (episodes removed when processing completes)
- **Episodes removed from Redis entirely once enrichment completes**
- Completed episodes live in FalkorDB, not Redis (prevents unbounded growth)

#### Episode Lifecycle Workflow

```
1. User adds journal entry
   ↓
   add_journal_entry() creates episode in FalkorDB
   ↓
   set_episode_status(uuid, "pending_nodes", journal=journal)
   ↓
   [Episode now in Redis, discoverable via get_episodes_by_status("pending_nodes")]

2. Worker processes extract_nodes
   ↓
   extract_nodes(uuid, journal) extracts entities
   ↓
   remove_episode_from_queue(uuid)
   [Episode REMOVED from Redis entirely - it's done, lives in FalkorDB]
```

**Critical Design Principle:**
Redis contains ONLY episodes that need processing. Once enrichment is complete, episodes are removed from Redis. Historical data lives in FalkorDB. This keeps Redis scan bounded by queue backlog size, not total episodes.

#### Worker API Functions

**All functions are in `backend/database/redis_ops.py`:**

```python
from backend.database.redis_ops import (
    set_episode_status,
    get_episode_status,
    get_episode_data,
    get_episode_uuid_map,
    get_episodes_by_status,
    remove_episode_from_queue,
)

# Set episode status and update indexes
set_episode_status(
    episode_uuid: str,
    status: str,
    journal: str | None = None,  # Required for initial status
    uuid_map: dict[str, str] | None = None  # Store UUID mapping
)

# Get current status
status = get_episode_status(episode_uuid)  # Returns str | None

# Get all episode metadata (raw strings, uuid_map is JSON)
data = get_episode_data(episode_uuid)  # Returns dict[str, str]

# Get parsed uuid_map (recommended for workers)
uuid_map = get_episode_uuid_map(episode_uuid)  # Returns dict[str, str] | None

# Scan episodes by status (for batch operations)
episodes = get_episodes_by_status("pending_nodes")  # Returns list[str]

# Remove episode from all Redis structures
remove_episode_from_queue(episode_uuid)
```

#### Worker Usage Patterns

**Pattern 1: Startup Recovery**
```python
# On worker startup, re-enqueue any pending episodes
from backend.database.redis_ops import get_episodes_by_status
from backend.services.tasks import extract_nodes_task

def recover_pending_episodes():
    # Re-enqueue pending node extractions (single-stage processing)
    for uuid in get_episodes_by_status("pending_nodes"):
        data = get_episode_data(uuid)
        extract_nodes_task(uuid, data["journal"])
```

**Pattern 2: Batch Retry**
```python
# Find episodes stuck in pending_nodes for >1 hour, retry
import time
from backend.database.redis_ops import get_episodes_by_status, get_episode_data

def retry_stale_episodes():
    for uuid in get_episodes_by_status("pending_nodes"):
        data = get_episode_data(uuid)
        created_at = float(data.get("created_at", 0))
        if time.time() - created_at > 3600:  # 1 hour
            extract_nodes_task(uuid, data["journal"])
```

**Pattern 3: Task Implementation**
```python
@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    """Background task for entity extraction."""
    from backend.database.redis_ops import remove_episode_from_queue, get_episode_status
    from backend.graph.extract_nodes import extract_nodes
    import dspy

    # Idempotency check
    if get_episode_status(episode_uuid) != "pending_nodes":
        return {"already_processed": True}

    # Use model manager for model persistence
    lm = get_model("llm")
    with dspy.context(lm=lm):
        result = extract_nodes(episode_uuid, journal)

    # Always remove from queue when done
    remove_episode_from_queue(episode_uuid)

    # Cleanup models if no work remains
    cleanup_if_no_work()

    return result
```

#### Important Notes

**Concurrency:**
- `set_episode_status()` is NOT atomic across concurrent calls
- Use Huey's task deduplication to ensure only one worker processes a given episode
- Single-threaded worker configuration (-w 1) provides sequential execution guarantee

**State Transitions:**
Valid transitions in the state machine (single-stage processing):
- `pending_nodes` → [removed from Redis] (processing complete)

**Cleanup Strategy:**
- `remove_episode_from_queue()` removes episode from Redis
- Call immediately when processing completes
- Redis should ONLY contain episodes actively being processed or waiting in queue
- Completed episodes live in FalkorDB only - query FalkorDB for historical data
- This keeps Redis scan bounded by queue backlog, preventing unbounded growth

**Testing:**
All functions have comprehensive test coverage in `tests/test_backend/test_redis_ops.py`:
- 27 tests covering all functions, edge cases, and lifecycle
- Tests verify scan-based lookups, recovery mechanisms, and idempotency
- Integration tests verify full episode lifecycle without stuck states

### Phase 4: The Developer Workflow (Native Optimization)

**Target Files:** `backend/optimizers/optimize_*.py` (Your external scripts)

**Goal:** Enable DSPy optimizers (MIPROv2, BootstrapFewShot) to run natively as standalone scripts.

**Why:**
*   **Complexity Reduction:** Optimizers are iterative and long-running. Wrapping them in background jobs makes debugging impossible and adds unnecessary abstraction.
*   **Isolation:** Optimization runs should be completely isolated from the application state. If an optimizer crashes, it shouldn't affect the database or the queue.

**Structure:**
*   These scripts import directly from `backend/inference` (DspyLM + loader).
*   They bypass the Manager and the Queue entirely.
*   This ensures that when you run an optimizer, you are getting a "clean room" environment with the exact model configuration used in production, but without the production machinery attached.

### Phase 5: Application Lifecycle

**Target File:** `charlie.py` (Main TUI Application)

**Goal:** Manage the background worker process automatically.

**Why:**
*   **User Experience:** The user should run one command to start the app. They shouldn't need to manually start a Redis server or a worker process in a separate terminal.
*   **Process Ownership:** Since `redislite` is embedded, the main TUI process owns the database file. The worker process must be spawned *after* `backend/database/lifecycle.py` has initialized the DB, and must be terminated cleanly when the TUI closes.

**Structure:**
*   Modify the application entry point.
*   Use Python's subprocess management to spawn the Huey consumer pointing at `backend/services/tasks.py`.
*   Configure the consumer to use the `thread` execution model (not `process`). This is vital because `backend/inference/manager.py` relies on global variables to keep models "warm." If the consumer forked a new process for every task, you would reload the model from disk every time, destroying performance.

---

## Critical Architecture Validations

### Thread Safety Analysis

**Verified:** Huey with `-k thread -w 1` creates THREE threads:
1. **Main Consumer Thread:** Supervisor for health checks and signals
2. **Scheduler Thread:** Enqueues periodic tasks (does NOT execute user code)
3. **Single Worker Thread:** Executes ALL tasks sequentially (regular + periodic)

**Critical Finding:** User code ONLY executes in the worker thread. The scheduler thread never calls user-defined functions - it only enqueues task objects. This means:

- NO `threading.Lock` required for model access
- Global `MODELS` dictionary is safe - all access is sequential
- Periodic cleanup tasks run in the same worker thread as inference tasks
- No concurrent access to llama-cpp-python (which is NOT thread-safe)

**WARNING:** llama-cpp-python is NOT thread-safe. Attempting to use multiple worker threads (`-w 2`) or accessing models from outside the worker thread will cause assertion failures and crashes. Single-worker configuration is mandatory.

### Execution Flow

```
[Scheduler Thread]
  ↓ (every 60s)
  read_periodic() → enqueue(task)
  ↓
[Queue]
  ↓
[Worker Thread] ← ALL user code runs here
  ↓
  dequeue() → execute(task)
  ↓
  Your graph operations (extract_nodes - single-stage processing)
  ↓
  Model Manager → get_model() → llama-cpp-python
```

**Result:** Sequential execution guarantees no race conditions.

---

## Implementation Details

### Phase 3 Detailed: Queue Setup

**File:** `backend/services/queue.py`

```python
"""Huey queue configuration using existing FalkorDB connection."""

from huey import RedisHuey
from backend.database.lifecycle import _db, _ensure_graph

def _get_redis_connection():
    """Access the existing redislite connection from FalkorDB.

    The FalkorDB instance from backend.database.lifecycle owns the
    embedded redis-server process. We must reuse its connection pool
    to avoid file locking conflicts.

    Returns:
        redis.ConnectionPool: The connection pool from the embedded redis server

    Raises:
        RuntimeError: If FalkorDB is not initialized
    """
    # Ensure database is initialized (triggers _init_db if needed)
    from backend.settings import DEFAULT_JOURNAL
    _ensure_graph(DEFAULT_JOURNAL)  # Any journal works to trigger init

    # Extract redis connection pool from FalkorDB's internal client
    if _db is None:
        raise RuntimeError("FalkorDB not initialized")

    # FalkorDB wraps a redis client - access its connection pool
    redis_client = _db.client
    return redis_client.connection_pool

# Initialize Huey with existing connection
# RedisHuey supports async tasks via asyncio integration
huey = RedisHuey(
    'charlie',
    connection_pool=_get_redis_connection(),
    # Single worker in thread mode; no per-model workers in this iteration
)
```

**File:** `backend/services/tasks.py`

```python
"""Huey tasks for graph operations (single-stage processing)."""

from backend.services.queue import huey
from backend.graph.extract_nodes import extract_nodes

@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    """Background task for entity extraction (single-stage processing).

    Simplified from original two-stage design (nodes→edges) to single-stage.
    Processes episode and removes from queue immediately upon completion.

    Calls backend/graph/extract_nodes.py which uses the Model Manager
    to access LLM. Returns metadata for TUI updates.

    Note: extract_nodes() internally calls get_model('llm') from the
    Manager and uses dspy.context() to configure the EntityExtractor.
    """
    result = extract_nodes(episode_uuid, journal)
    return {
        'episode_uuid': result.episode_uuid,
        'extracted_count': result.extracted_count,
        'new_entities': result.new_entities,
        'resolved_count': result.resolved_count,
    }
```

### Phase 2 Detailed: Model Manager

**File:** `backend/inference/manager.py`

```python
"""Model lifecycle management for model persistence."""

import gc
import time
from typing import Literal

from backend.inference import DspyLM

# Global state - safe because single worker thread
MODELS = {
    'llm': {'model': None},
    # Future: add embedding/reranker entries when implemented
}

ModelType = Literal['llm']  # Expand as more models are added

def get_model(model_type: ModelType = 'llm') -> DspyLM:
    """Load model if needed, update timestamp, return instance.

    NO LOCKING REQUIRED: Single worker thread guarantees sequential access.

    Args:
        model_type: Currently only 'llm' supported

    Returns:
        DspyLM instance (implements dspy.BaseLM interface)

    Notes:
        - DspyLM wraps llama.cpp with DSPy-compatible interface
        - Model stays loaded in memory for subsequent calls (model persistence)
        - First call loads from disk (~4GB), subsequent calls are instant
    """
    entry = MODELS[model_type]

    # Load the model (only LLM for now)
    if entry['model'] is None:
        if model_type == 'llm':
            entry['model'] = DspyLM()  # Uses settings from backend.settings
        else:
            raise ValueError(f"Model type '{model_type}' not yet implemented")

    return entry['model']

def unload_all_models():
    """Unload all models to free memory."""
    for entry in MODELS.values():
        if entry['model'] is not None:
            entry['model'] = None
    gc.collect()
```

**Cleanup (event-driven, no timers):**

Tasks call `cleanup_if_no_work()` after completion, which checks Redis backlog and calls `unload_all_models()` if `pending_nodes` is empty.

### Phase 1 Detailed: Model Factory

**NOTE:** Copy `inference_runtime/` to `backend/inference/` as the foundation.
This provides the stateless model loading layer.

**Files to Copy:**
- `inference_runtime/__init__.py` → `backend/inference/__init__.py`
- `inference_runtime/dspy_lm.py` → `backend/inference/dspy_lm.py`
- `inference_runtime/loader.py` → `backend/inference/loader.py`

**Modifications to `backend/inference/dspy_lm.py`:**

```python
"""DSPy BaseLM implementation backed directly by llama.cpp."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import dspy

from backend.settings import MODEL_CONFIG  # Updated import
from .loader import load_model

# ... rest of DspyLM class unchanged ...
```

**Modifications to `backend/inference/loader.py`:**

```python
"""Helpers for loading llama.cpp models."""

from __future__ import annotations

import contextlib
import logging
import os
import sys

from llama_cpp import Llama

from backend.settings import (  # Updated import
    LLAMA_CPP_GPU_LAYERS,
    LLAMA_CPP_N_CTX,
    LLM_MODEL_PATH,  # Optional: override default model
)

# ... rest unchanged, but optionally use LLM_MODEL_PATH if provided ...
```

**Key Points:**
- `DspyLM` (dspy.BaseLM) and `load_model()` stay unchanged aside from import paths (inference_runtime → backend.inference).
- Optimizer scripts import from `backend.inference` (same as app).
- No additional factory wrapper needed - `DspyLM` is the factory.

### Phase 5 Detailed: Subprocess Management

**File:** `charlie.py` (modifications)

```python
"""Main TUI application with embedded Huey worker."""

import subprocess
import atexit
import signal
from textual.app import App

class CharlieApp(App):
    def __init__(self):
        super().__init__()
        self.huey_process = None

    def on_mount(self):
        """Start Huey worker after database initialization."""
        # Database is already initialized by backend.database.lifecycle
        # Now spawn the worker
        self.huey_process = subprocess.Popen([
            'huey_consumer',
            'backend.services.tasks.huey',
            '-k', 'thread',  # CRITICAL: single-thread execution
            '-w', '1',       # CRITICAL: one worker
            '-v',            # Verbose logging
        ])

        # Register cleanup
        atexit.register(self._shutdown_huey)

    def _shutdown_huey(self):
        """Terminate Huey worker gracefully."""
        if self.huey_process is not None:
            self.huey_process.send_signal(signal.SIGTERM)
            try:
                self.huey_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.huey_process.kill()
```

---

## Configuration Requirements

**Add to `backend/settings.py`:**

```python
from pathlib import Path

# Existing settings
# DB_PATH = Path("data/charlie.db")
# DEFAULT_JOURNAL = "default"

# Model paths (optional; defaults to HF cache locations if not set)
MODELS_DIR = Path("models")
LLM_MODEL_PATH = MODELS_DIR / "llama-3-8b-instruct.gguf"
EMBEDDING_MODEL_PATH = MODELS_DIR / "nomic-embed-text-v1.5.gguf"
RERANKER_MODEL_PATH = MODELS_DIR / "bge-reranker-v2-m3.gguf"

# Huey configuration
HUEY_WORKER_TYPE = 'thread'
HUEY_WORKERS = 1
```

---

## Testing Strategy

### Unit Tests (No Model Loading)

```python
# tests/test_inference/test_manager.py
def test_model_registry_structure():
    """Test MODELS dictionary structure."""
    from backend.inference.manager import MODELS
    assert 'llm' in MODELS
    assert 'model' in MODELS['llm']

def test_unload_logic():
    """Test unload_all_models without real models."""
    # Mock model objects and verify unload behavior
    pass
```

### Integration Tests (With Models)

```python
# tests/test_services/test_tasks.py
@pytest.mark.asyncio
@pytest.mark.slow
async def test_extract_nodes_task(isolated_graph):
    """Test full task execution with model loading."""
    from backend import add_journal_entry
    from backend.services.tasks import extract_nodes_task

    episode_uuid = await add_journal_entry("Today I met Sarah.")
    from backend.settings import DEFAULT_JOURNAL
    result = await extract_nodes_task(episode_uuid, DEFAULT_JOURNAL)

    assert result['extracted_count'] > 0
```

### Manual Verification

```bash
# 1. Start Huey consumer manually
huey_consumer backend.services.tasks.huey -k thread -w 1 -v

# 2. In another terminal, enqueue a task
python -c "
from backend.services.tasks import extract_nodes_task
from backend.settings import DEFAULT_JOURNAL
from backend import add_journal_entry
import asyncio

async def test():
    uuid = await add_journal_entry('Test content')
    task = extract_nodes_task(uuid, DEFAULT_JOURNAL)
    print(f'Task enqueued: {task.id}')

asyncio.run(test())
"

# 3. Watch worker logs for execution
```
