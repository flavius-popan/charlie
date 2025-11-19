# Local LLM Orchestration & Inference Infrastructure

## Purpose
This infrastructure is designed to enable high-performance, local AI features within a Textual TUI application while strictly managing limited system resources (RAM/VRAM). It solves the critical challenge of running heavy models (LLMs, Embeddings, Rerankers) alongside a user interface without freezing the application or crashing the system due to Out-Of-Memory (OOM) errors.

The design implements a **"Warm Session" architecture**, where models remain loaded in memory for rapid sequential inference (e.g., a chat session) but are automatically swapped or unloaded when the context changes (e.g., switching from Chat to Journal Ingestion).

## Huey's Role
**Huey** serves as the asynchronous orchestration layer that bridges the user interface and the heavy inference engine. Its specific responsibilities are:

*   **Non-Blocking Execution:** Offloads heavy inference tasks (generation, embedding) to a background process, ensuring the TUI remains responsive to user input at all times.
*   **Resource Serialization:** configured with a **single-threaded worker**, Huey acts as a strict FIFO (First-In-First-Out) queue. This guarantees that two heavy models are never triggered simultaneously, preventing resource contention.
*   **State Persistence (Warm Sessions):** By utilizing the `thread` worker type (rather than `process`), Huey keeps the worker process alive between tasks. This allows the **Model Manager** to maintain global state (loaded weights) in RAM, making subsequent inference calls near-instant.
*   **Infrastructure Efficiency:** Huey integrates directly with the application's existing embedded `redislite`/`falkordblite` database, requiring no external broker services or additional system overhead.

### Phase 1: The Inference Foundation (Stateless)

**Target File:** `backend/inference/engine.py`

**Goal:** Create a "Model Factory" layer that knows *how* to instantiate models but holds no opinion on *when* or *how long* they are kept.

**Why:**
*   **Decoupling:** Your DSPy optimizer scripts need raw access to model objects without the overhead of job queues, database connections, or memory management logic.
*   **Reusability:** The exact same loading logic (using paths from `backend/settings.py`) must be shared between the App and the Dev scripts to ensure optimization results are valid in production.

**Structure:**
*   This module contains pure factory functions.
*   It reads configuration from `backend/settings.py` but stores no global state.
*   It provides a custom `DSPy` adapter class that wraps the raw `llama_cpp` object, ensuring the interface matches what DSPy expects for signatures and predictors.

### Phase 2: The Orchestrator (Stateful)

**Target File:** `backend/inference/manager.py`

**Goal:** Create a "Model Manager" layer that manages the lifecycle of models within the application's runtime.

**Why:**
*   **Memory Management:** A local machine cannot hold an Instruct Model, an Embedding Model, and a Reranker in VRAM simultaneously. The system must enforce "Exclusive Access"—automatically unloading one model to make room for another.
*   **Latency Reduction:** Loading a model from disk is slow. The Manager must maintain a "Warm Session" (global state) so that sequential requests (like a chat conversation) are instant.
*   **Abstraction:** The consumer tasks shouldn't care about memory constraints. They should simply request "Instruct," and the Manager handles the complex logic of checking what is currently loaded, swapping if necessary, and returning the object.

**Structure:**
*   This module imports the *Factory* from `backend/inference/engine.py`.
*   It maintains a global registry (dictionary) of currently loaded models.
*   It implements the "Swap Logic": Before loading a requested model via the Factory, it checks the registry and explicitly unloads/garbage-collects conflicting models.
*   **Crucially:** This module is *never* imported by your Dev/Optimizer scripts. It is exclusive to the Application/Huey worker.

### Phase 3: The Service Layer (Asynchronous Queue)

**Target Files:** `backend/services/queue.py` and `backend/services/tasks.py`

**Goal:** Integrate a lightweight task queue that shares the existing embedded infrastructure.

**Why:**
*   **Non-Blocking UI:** Inference takes time. The TUI must remain responsive while the LLM generates text or embeddings.
*   **Resource Safety:** By using a single-threaded worker, you strictly serialize inference operations. This prevents the application from accidentally trying to run two heavy models at once, which would crash the application (OOM).
*   **Infrastructure Reuse:** You are already using `redislite`. The queue must connect to this *existing* socket rather than spinning up a new Redis instance, which would cause file contention and locking issues.

**Structure:**
*   `backend/services/queue.py` bridges the `huey` library with your existing `backend/database/lifecycle.py` module. It must access the existing connection pool to ensure it talks to the same embedded database file defined in `backend/settings.py`.
*   `backend/services/tasks.py` defines the specific jobs that wrap `backend/graph/*` operations (e.g., `extract_nodes_task`, `extract_edges_task`).
*   These tasks call graph operation functions (like `backend/graph/extract_nodes.py::extract_nodes()`) which internally use `backend/inference/manager.py` to retrieve models.
*   Tasks do NOT import the Factory directly - all model access flows through the Manager.

### Phase 4: The Developer Workflow (Native Optimization)

**Target Files:** `backend/optimizers/optimize_*.py` (Your external scripts)

**Goal:** Enable DSPy optimizers (MIPROv2, BootstrapFewShot) to run natively as standalone scripts.

**Why:**
*   **Complexity Reduction:** Optimizers are iterative and long-running. Wrapping them in background jobs makes debugging impossible and adds unnecessary abstraction.
*   **Isolation:** Optimization runs should be completely isolated from the application state. If an optimizer crashes, it shouldn't affect the database or the queue.

**Structure:**
*   These scripts import **Directly from `backend/inference/engine.py`**.
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
  Your graph operations (extract_nodes, extract_edges)
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

from huey import AsyncRedisHuey  # AsyncRedisHuey for async task support
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
    _ensure_graph("default")  # Any journal works to trigger init

    # Extract redis connection pool from FalkorDB's internal client
    if _db is None:
        raise RuntimeError("FalkorDB not initialized")

    # FalkorDB wraps a redis client - access its connection pool
    redis_client = _db.client
    return redis_client.connection_pool

# Initialize Huey with existing connection
# CRITICAL: Use AsyncRedisHuey (not RedisHuey) for async task support
huey = AsyncRedisHuey(
    'charlie',
    connection_pool=_get_redis_connection(),
)
```

**File:** `backend/services/tasks.py`

```python
"""Huey tasks for graph operations."""

from backend.services.queue import huey
from backend.graph.extract_nodes import extract_nodes
from backend.graph.extract_edges import extract_edges

@huey.task()
async def extract_nodes_task(episode_uuid: str, journal: str):
    """Background task for entity extraction.

    Calls backend/graph/extract_nodes.py which uses the Model Manager
    to access LLM. Returns metadata for TUI updates.

    Note: extract_nodes() internally calls get_model('llm') from the
    Manager and uses dspy.context() to configure the EntityExtractor.
    """
    result = await extract_nodes(episode_uuid, journal)
    return {
        'episode_uuid': result.episode_uuid,
        'extracted_count': result.extracted_count,
        'resolved_count': result.resolved_count,
    }

@huey.task()
async def extract_edges_task(episode_uuid: str, journal: str):
    """Background task for relationship extraction."""
    result = await extract_edges(episode_uuid, journal)
    return {
        'episode_uuid': result.episode_uuid,
        'edges_created': result.edges_created,
    }
```

### Phase 2 Detailed: Model Manager

**File:** `backend/inference/manager.py`

```python
"""Model lifecycle management for warm sessions."""

import gc
import time
from typing import Literal

from backend.inference import DspyLM

# Global state - safe because single worker thread
MODELS = {
    'llm': {'model': None, 'last_used': 0},
    # Future: 'embedding', 'reranker' when needed
}

IDLE_TIMEOUT = 300  # Unload after 5 minutes

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
        - Model stays loaded in memory for subsequent calls (warm session)
        - First call loads from disk (~4GB), subsequent calls are instant
    """
    entry = MODELS[model_type]
    entry['last_used'] = time.time()

    if entry['model'] is not None:
        return entry['model']

    # Load the model (only LLM for now)
    if model_type == 'llm':
        entry['model'] = DspyLM()  # Uses settings from backend.settings
    else:
        raise ValueError(f"Model type '{model_type}' not yet implemented")

    return entry['model']

def _unload_all_except(keep: ModelType | None = None):
    """Unload models to free memory."""
    for name, entry in MODELS.items():
        if name != keep and entry['model'] is not None:
            entry['model'] = None
            gc.collect()

def unload_idle_models():
    """Called by periodic task to free memory.

    Runs in worker thread - safe to access globals.
    """
    now = time.time()
    for name, entry in MODELS.items():
        if entry['model'] is not None:
            idle_time = now - entry['last_used']
            if idle_time > IDLE_TIMEOUT:
                entry['model'] = None
                gc.collect()
```

**Periodic Task (in `backend/services/tasks.py`):**

```python
from huey import crontab

@huey.periodic_task(crontab(minute='*'))
def cleanup_idle_models():
    """Runs every minute to unload unused models.

    Executes in worker thread - no concurrency issues.
    """
    from backend.inference.manager import unload_idle_models
    unload_idle_models()
```

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
- Copy preserves existing `DspyLM` (dspy.BaseLM implementation)
- Copy preserves `load_model()` with Llama.from_pretrained()
- Only import paths need updating (inference_runtime → backend.inference)
- Optimizer scripts import from `backend.inference` (same as app)
- No factory wrapper needed - `DspyLM` is the factory

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
            '-k', 'thread',  # CRITICAL: thread mode for warm sessions
            '-w', '1',       # CRITICAL: single worker for thread safety
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

# Model paths
MODELS_DIR = Path("models")
LLM_MODEL_PATH = MODELS_DIR / "llama-3-8b-instruct.gguf"
EMBEDDING_MODEL_PATH = MODELS_DIR / "nomic-embed-text-v1.5.gguf"
RERANKER_MODEL_PATH = MODELS_DIR / "bge-reranker-v2-m3.gguf"

# Huey configuration
HUEY_WORKER_TYPE = 'thread'
HUEY_WORKERS = 1
HUEY_IDLE_TIMEOUT = 300  # seconds
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
    assert 'last_used' in MODELS['llm']

def test_unload_logic():
    """Test _unload_all_except without real models."""
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
    result = await extract_nodes_task(episode_uuid, "default")

    assert result['extracted_count'] > 0
```

### Manual Verification

```bash
# 1. Start Huey consumer manually
huey_consumer backend.services.tasks.huey -k thread -w 1 -v

# 2. In another terminal, enqueue a task
python -c "
from backend.services.tasks import extract_nodes_task
from backend import add_journal_entry
import asyncio

async def test():
    uuid = await add_journal_entry('Test content')
    task = extract_nodes_task(uuid, 'default')
    print(f'Task enqueued: {task.id}')

asyncio.run(test())
"

# 3. Watch worker logs for execution
```

