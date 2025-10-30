Phase 02 – Core Extraction Loop
===============================

**Outcome**: Replace Graphiti prompt calls with DSPy modules for entity/edge extraction, generate embeddings & reranker scores via Qwen3 models, and persist results to FalkorDB through the embedded `falkordblite` runtime. Ship a Gradio UI that surfaces these steps for interactive validation.

Implementation Steps
--------------------
Consult the FalkorDB and embedding notes under `research/` (particularly `research/06-falkordb-backend.md`, `research/09-embedding-integration.md`, and the NER guidance in `research/10-distilbert-ner-integration.md`) and review the active `.venv` packages to align API usage with shipped Graphiti code before coding each step.
1. **FalkorDBLite Orchestration & Indexes**
   - Spin up a managed `falkordblite` instance, capturing database file placement, socket paths, and lifecycle hooks.
   - Ensure Graphiti adapters can reuse the embedded server or spawn isolated instances for tests.
   - Create required indexes upfront: UUID uniqueness, `EntityNode.name_embedding` vector index, and factual text search indexes.
   - Wrap database calls in context managers to guarantee transactions for bulk saves and controlled teardown.

2. **DSPy LLMClient Wrapper**
   - Implement the custom DSPy `LLMClient`/adapter bridging MLX-backed Outlines to Graphiti expectations.
   - Support temperature, max-tokens, and schema-aware generation compatible with `EntityNode`/`EntityEdge` definitions.
   - Log prompt/response metadata for later MIPRO optimization.

3. **Extraction Modules**
   - Finalize `ExtractEntitiesModule` (name, labels, candidate UUID) and `ExtractRelationshipsModule` (source, target, fact).
   - Ensure outputs align with Graphiti Pydantic models—no ad-hoc dicts.
   - Handle reference-time propagation so temporal defaults exist before Phase 04.

4. **DistilBERT NER Integration Modes**
   - Evaluate three modes: (a) DistilBERT-only entity extraction, (b) hybrid (NER + DSPy reconciliation), (c) DSPy-only with NER hints/steering.
   - Build configuration flags to toggle among modes at runtime for benchmarking.
   - Define caching and batching strategy for ONNX runtime sessions to minimize warm-start costs.
   - Document trade-offs (latency, recall, alignment with Graphiti schemas) uncovered during experimentation.

5. **Embeddings & Reranker**
   - Implement `Qwen3EmbedderClient` (MLX quantized) and integrate into the pipeline post-deduplication placeholder.
   - Implement `Qwen3RerankerClient` and wire into search/retrieval utilities; ensure batching to avoid latency spikes.
   - Cache model handles to avoid repeated initialization; respect thread-safety conventions set in Phase 01.

6. **Custom Pipeline Orchestration**
   - Orchestrate steps: context fetch → extraction → provisional UUID assignment → embedding generation → reranker metadata → bulk save.
   - Persist intermediary outputs (JSON) for replay in later phases.
   - Instrument with structured logging and timing for each stage.

- Add unit tests for entity/edge generation using fixtures derived from the canonical episodes.
- Create targeted tests for the DistilBERT integration covering all three modes (NER-only, blended, hint-only) to guarantee consistent behavior under load.
- Extend `tests/test_constrained_generation.py` to cover the DSPy signatures with mocked FalkorDB interactions.
- Create an integration test that runs the full Phase 02 pipeline end-to-end against an ephemeral `falkordblite` instance (or stub).
- Capture embedding dimensionality and reranker scoring smoke tests to catch mismatch issues early.

Gradio Checkpoint
-----------------
- Clone the foundation UI into `gradio_core_extraction.py`.
- Add controls to toggle DSPy extraction on/off, inspect raw module outputs, and visualize Falkor persistence status.
- Surface latency metrics (per step) and embed vector previews (shape, sample values) for manual verification.
- Expose DistilBERT NER outputs for each mode (NER-only, hybrid, hints) so analysts can compare quality/latency trade-offs.
- Allow exporting the structured outputs (entities, edges, embeddings) to JSON for regression testing.

Deliverables
------------
- Functional DSPy-powered extraction loop producing `EntityNode` & `EntityEdge` objects with embeddings.
- FalkorDBLite populated with base entities/relationships as verified via the Gradio app and automated tests.
- Updated documentation describing configuration, logging, and replay procedures for this phase.

Validation Gate
---------------
- Demo the embedded FalkorDBLite lifecycle, DSPy extraction outputs, and Falkor persistence via the Gradio UI.
- Pause further implementation until reviewers confirm data integrity, indexing strategy, and adapter behavior.
