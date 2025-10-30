Phase 01 – Foundations & Guardrails
===================================

**Outcome**: Agree on the custom pipeline boundaries, confirm model reuse expectations, and deliver a lightweight Gradio harness (inspired by `gradio_app.py`) that exercises today’s extraction module as a baseline for later phases.

Key Activities
--------------
1. Inventory the existing Graphiti assets (models, FalkorDB driver, bulk utilities) and capture any version pinning or MLX constraints.
   - Include a manual sweep of the local `.venv` installation (e.g., `graphiti_core`, FalkorDB drivers) to align the plan with shipped code.
   - Reconcile findings with the technical briefs in `research/` (FalkorDB backend, embeddings, reranking) so investigations stay grounded in prior work.
2. Review the embedded `falkordblite` runtime (see `falkordblite-evaluation.md`) and document how its lifecycle maps onto our adapters.
3. Document the end-to-end episode → graph flow, focusing on where we will hand off to DSPy modules instead of Graphiti prompts.
4. Produce a small set of canonical test episodes plus expected entities/edges for sanity checks across phases.
5. Evaluate the existing `distilbert_ner.py` + ONNX runtime (see repository root) to catalogue capabilities, latency, and how NER outputs can drive multiple extraction modes (NER-only, hybrid, hints).
6. Establish a shared configuration surface (env vars or `.env` overrides) for FalkorDBLite database files, model paths (Qwen + DistilBERT ONNX), and DSPy adapters.
7. Validate thread-safety assumptions for any shared adapters or embeddings, annotating locking expectations inline (per repository guidelines).

Graphiti Integration Notes
--------------------------
- **Core Nodes**: `EpisodicNode` (raw episode), `EntityNode` (entity with embeddings & attributes), `CommunityNode` (optional grouping).
- **Core Edges**: `EpisodicEdge` for MENTIONS, `EntityEdge` for relationships with `fact`, temporal bounds, and link-back episodes.
- **Bulk Save**: continue to rely on `add_nodes_and_edges_bulk` for persistence; confirm FalkorDB-specific merge semantics.
- **UUID Discipline**: decide on a deterministic UUID scheme for extracted entities/edges prior to deduplication steps.

Embedded Falkor Runtime
-----------------------
- Adopt `falkordblite` to host FalkorDB locally; capture database path conventions and how to share the embedded server across processes.
- Note that FalkorDBLite exposes Redis-compatible sockets (default Unix socket plus optional TCP); plan adapter hooks accordingly.
- Record startup/shutdown expectations so we can manage lifecycle explicitly during tests, CLI scripts, and Gradio sessions.

NER Acceleration Baseline
-------------------------
- Audit `distilbert_ner.py` and the `distilbert-ner-uncased-onnx/` assets to document model inputs, batching limits, and ONNX runtime requirements.
- Record how the existing Gradio demo fuses DistilBERT NER outputs with LLM extraction (entity hints, toggle behaviors).
- Outline hypotheses for using DistilBERT as: (a) the primary entity source, (b) a blended source, or (c) a steering hint provider; ensure later phases can toggle among them.
- Decide on baseline caching, stride settings, and confidence thresholds so later phases can reuse consistent heuristics.

Data Flow Checkpoints
---------------------
1. Gather prior episodes (`EPISODE_WINDOW_LEN`) for context retrieval via FalkorDB.
2. Create an `EpisodicNode` from the current episode but hold persistence until extraction completes.
3. Route the content through DSPy modules for entities, relationships, and optional attributes.
4. Attach embeddings (Qwen3) and reranker scores only after deduplication to avoid churn.
5. Save the resulting nodes/edges in a single bulk operation with transactional guarantees.

Testing & Tooling
-----------------
- Begin crafting fixtures for `pytest tests/test_constrained_generation.py` so later phases can extend them.
- Add lightweight smoke scripts (YAML or shell) that invoke FalkorDB health checks and DSPy model loading.
- Track open questions or deferred decisions in `plans/TODO.md` if scope arises (keep breadcrumbs per repo guidance).

Gradio Checkpoint
-----------------
- Fork `gradio_app.py` into `gradio_foundation.py` (or similar) with minimal adjustments: ensure it loads the current DAG, allows users to paste episode text, and renders the extraction result using the existing Digraph helper.
- Add explicit annotations in the UI for which parts are baseline vs. to-be-replaced DSPy components.
- Expose toggles for loading canned episodes and exporting JSON so we can snapshot outputs before deeper changes.

Dependencies & Deliverables
---------------------------
- Confirm MLX builds and weights for Qwen models are locally available before the next phase.
- Verify the DistilBERT ONNX bundle is downloaded/accessible (or document download automation) and record expected checksum/size.
- Produce a short README section detailing how to run the foundation Gradio app and any required environment setup.
- Capture a decision log for FalkorDB indexes and transaction strategy to unblock Phase 02 work.

Validation Gate
---------------
- Present FalkorDBLite lifecycle documentation, baseline Gradio behavior, and configuration surfaces for review.
- Pause further implementation until maintainers sign off on the recorded constraints and foundation harness.
