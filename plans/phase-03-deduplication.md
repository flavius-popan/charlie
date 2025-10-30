Phase 03 – Deduplication & Resolution
=====================================

**Outcome**: Merge duplicate entities/edges across episodes, maintain UUID mappings, and surface contradiction handling while keeping the embedded FalkorDBLite datastore consistent. Provide an interactive Gradio view to explore dedupe decisions.

Implementation Steps
--------------------
Cross-reference the deduplication and graph driver notes in `research/06-falkordb-backend.md` (and companion docs) and inspect the active `.venv` Graphiti modules before implementing each step so heuristics stay aligned with existing behavior.
1. **Embedding Search Integration**
   - Finalize similarity queries against `name_embedding` to fetch candidate matches using FalkorDBLite’s embedded vector indexes.
   - Normalize embeddings before comparison and cache lookups to limit round-trips.
   - Surface confidence thresholds (configurable) for merge decisions.

2. **Entity Deduplication**
   - Implement `DedupeNodesModule` to decide match/merge/new entity outcomes across all extraction modes (NER-only, hybrid, DSPy-only).
   - Maintain a `uuid_map` for provisional → resolved entity IDs so downstream steps remain stable.
   - Record provenance (source episode IDs, reasoning text, extraction mode) for auditing conflict decisions.
   - Incorporate DistilBERT NER spans/confidences as features when available, while preserving graceful degradation when only DSPy output exists.

3. **Relationship Resolution**
   - Implement `DedupeEdgesModule` and `InvalidateEdgesModule` to merge facts, detect contradictions, and set `invalid_at`.
   - Ensure relationship merges update `episodes` back-references and maintain referential integrity.
   - Capture invalidation events for later temporal processing.

4. **Persistence & Transactions**
   - Extend the bulk save path to handle merges vs. inserts cleanly (upserts by UUID) against the FalkorDBLite-managed server.
   - Guard against race conditions with optimistic locking or retry loops where necessary.
   - Update logging to highlight merge outcomes and any discarded data.

Testing Strategy
----------------
- Build table-driven tests covering match, merge, and new-entity outcomes with deterministic embeddings.
- Add multi-episode regression cases verifying UUID stability and edge reconciliation.
- Validate error handling when FalkorDBLite returns conflicting records or concurrent modifications.
- Ensure pytest fixtures cover both high-confidence and ambiguous dedupe scenarios.
- Add unit tests that confirm dedupe logic behaves correctly for each extraction mode and when the ONNX runtime is unavailable.

Gradio Checkpoint
-----------------
- Extend the Phase 02 UI into `gradio_deduplication.py`.
- Include panels that show incoming entities/edges, candidate matches, DSPy rationale, and the final merged result.
- Allow analysts to adjust thresholds live and replay dedupe decisions against stored transcripts.
- Provide “before vs after” graphs (Graphviz) and downloadable audit trails for manual QA.

Deliverables
------------
- Production-ready dedupe modules with confidence tuning and audit metadata.
- Verified FalkorDBLite datasets demonstrating low duplicate rates and clean relationship merges.
- Updated documentation capturing operational runbooks for dedupe-related incidents.

Validation Gate
---------------
- Walk reviewers through merge heuristics, FalkorDBLite transaction traces, and the Gradio audit tooling.
- Pause before advancing until dedupe behavior, thresholds, and rollback plans are approved.
