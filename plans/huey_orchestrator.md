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
*   `backend/services/tasks.py` defines the specific jobs (e.g., `chat_response`, `ingest_entry`).
*   These tasks import `backend/inference/manager.py` to retrieve models. They do *not* import the Factory directly.

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
