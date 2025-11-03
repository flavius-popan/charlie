# Phase Parity – Custom Pipeline vs. `graphiti.add_episode()`

Objective: deliver a near 1:1 synchronous replica of Graphiti’s `add_episode()` workflow (excluding embeddings/reranking execution for now), implemented via DSPy + Outlines, instrumented in `graphiti_pipeline.py`, and fully explorable through the Gradio UI (`graphiti-poc.py`). This plan captures the remaining gaps versus Graphiti’s stock behavior and sequences the work so each stage can be exercised and optimized in isolation.

---

## 1. Current Snapshot
- **Implemented** (validated in `graphiti_pipeline.py`, `graphiti-poc.py`):
  - DistilBERT NER (all entities or persons-only toggle).
  - DSPy fact & relationship extraction via `FactExtractionSignature` / `RelationshipSignature`.
  - Basic Episodic/Entity/Relationship object creation and Falkor write.
  - Gradio stage-by-stage visualization up through FalkorDB persistence.
- **Missing vs. `add_episode()`**:
  - Retrieval of prior episodes (context window) and deterministic group/UUID handling.
  - Entity resolution/deduping against existing graph data (UUID remapping, provenance, audit trail).
  - Relationship resolution & invalidation handling; temporal fields on edges.
  - Entity attribute extraction + persistence; episode + entity summaries.
  - Temporal defaults/propagation (valid/invalid/expired at) and reference-time plumbing.
  - Episode-level summary generation & storage.
  - Embedding/reranker integration points (currently absent, need stubs).
  - Gradio surfacing for the above artifacts (context episodes, dedupe decisions, summaries, temporals, attributes, audit metadata).
  - Regression tests covering new stages and FalkorDB interactions.

---

## 2. Parity Goals (Mirroring `add_episode()` Steps)
1. **Initialization & Context**  
   - Align inputs (episode name, reference time, group) and fetch prior episodes (`EPISODE_WINDOW_LEN`).  
   - Ensure synchronous API mirrors async flow semantics for now.
2. **Node Extraction**  
   - Keep DistilBERT-only mode initially; structure for hybrid/DSPy-only extension.  
   - Enrich with DSPy-based entity summaries and episode summary generation.
3. **Node Resolution (Deduplication)**  
   - Implement dedupe against existing nodes (embedding search stubbed, rely on exact/heuristic match).  
   - Maintain UUID mapping & provenance notes.
4. **Edge Extraction & Resolution**  
   - Preserve DSPy relationship extraction; add dedupe/merge/invalidation logic.  
   - Attach temporal metadata placeholders.
5. **Attribute Extraction**  
   - Generate entity attributes (labels, properties) consistent with Graphiti models.
6. **Episode Data Processing**  
   - Build episodic edges, attach entity edge UUIDs, update summaries, handle optional content stripping.  
   - Prepare data structures for embeddings/reranker (stubbed).
7. **Persistence & Instrumentation**  
   - Bulk-save parity with `add_nodes_and_edges_bulk` semantics.  
   - Capture audit logs, structured telemetry, replay artifacts.

---

## 3. Implementation Roadmap

### A. Pipeline Foundations
- Introduce a pipeline configuration object (group id, window size, toggle flags for dedupe, attributes, temporals) to keep stages configurable.
- Add context retrieval helper that queries FalkorDBLite for the last `EPISODE_WINDOW_LEN` episodes by group/reference time. Surface in Gradio (read-only table).
- Normalize naming/UUID strategy to match Graphiti (`group_id`, deterministic `uuid` for new episodes/nodes when possible).

### B. Entity & Episode Summaries
- Add DSPy signatures/modules for:
  - `SummarizeEpisode` – produce episode summary text stored on `EpisodicNode.summary` (or analogous field).  
  - `SummarizeEntity` – generate per-entity summary leveraging extracted relationships + prior edges.  
- Update pipeline to invoke summaries post-extraction but pre-persistence.  
- Extend Gradio to show both summaries (text area per entity + episode section) and allow rerun tweaks.

### C. Deduplication Layer
- Build synchronous dedupe helpers:
  - `fetch_candidate_entities` querying FalkorDB by normalized name and (stubbed) embedding similarity.
  - `resolve_entities` to map provisional UUIDs to persisted ones, merge attributes, carry provenance.  
- Maintain `uuid_map` (extracted → resolved) and expose decisions in Gradio (accordion/table with rationale).
- Implement deterministic heuristics for match vs. new entity (normalize, previous episodes list).  
- Stub embedding similarity call with placeholder scoring; insert TODO for future Qwen embedder.

### D. Relationship Resolution & Temporal Metadata
- Create relationship dedupe functions mirroring Graphiti’s `_extract_and_resolve_edges` semantics:
  - Merge duplicates (update `episodes`, maintain fact text).  
  - Flag contradictions, set `invalid_at` when needed.  
  - Propagate `reference_time` to `valid_at` (default) and accept overrides.  
- Add Gradio panels for resolved vs. invalidated edges and temporal fields preview.

### E. Attribute Extraction
- Introduce DSPy signature (`ExtractAttributes`) returning attribute dict per entity (respect Graphiti schema).  
- Merge attributes onto resolved nodes without overwriting existing non-empty values.  
- Support label augmentation (e.g., `["Person"]`) derived from DistilBERT tags.  
- Visualize attributes in Gradio (JSON panel with diff vs. existing).

### F. Episode Processing & Persistence
- Update Stage 4 builder to:
  - Accept resolved nodes/edges, apply summaries/attributes/temporals.  
  - Update episode summary field and `entity_edges` list.  
- Extend Stage 5 write to leverage `add_nodes_and_edges_bulk` (or mimic structure exactly) while stubbing embedding/reranker invocation points:
  - Create `prepare_embeddings()` and `prepare_reranker()` placeholders returning metadata + TODO comment.  
  - Ensure hooks in pipeline + UI toggles show current stub status.
- Enhance write result logging with dedupe/temporal info.

### G. Testing & Validation
- Add pytest coverage:
  - Unit tests for dedupe heuristics, attribute merging, temporal propagation (`tests/test_constrained_generation.py` or new modules).  
  - Integration test simulating multi-episode ingestion verifying UUID stability, summaries, temporals.  
  - Snapshot/assertions for Gradio pipeline states (use fast API harness or module-level tests).  
- Document manual verification steps (Gradio flows, Falkor queries).

### H. Documentation & TODO Tracking
- Update `README.md` / `docs/` with instructions for new stages & configuration toggles.  
- Record deferred items (embeddings, reranker, NER mode expansion) in `plans/TODO.md` or inline TODOs as appropriate.

---

## 4. Gradio (`graphiti-poc.py`) Enhancements
- Restructure UI into collapsible stages aligned with new pipeline steps (context, dedupe, summaries, attributes, temporals).
- Surface intermediate artifacts:
  - Prior episodes & retrieved context.  
  - Dedupe decisions (`match`, `merge`, `new`).  
  - Entity/episode summaries; allow re-run buttons per section.  
  - Attribute dictionaries + diff view.  
  - Temporal metadata fields.  
- Add configuration widgets (toggles/sliders) for thresholds, dedupe enable/disable, reference time override.  
- Display stub status for embeddings/reranker (disabled badge + TODO link).  
- Ensure database stats/graph render remain current; optionally include audit log download (JSON).

---

## 5. Milestones & Checkpoints
1. **Parity Foundations** – context retrieval, config object, Gradio context panel.  
2. **Summaries & Attributes** – DSPy modules + UI panels, persistence wiring.  
3. **Dedup & Temporal** – entity/edge resolution, UUID mapping, temporal fields, Gradio inspection.  
4. **Persistence Parity** – bulk save alignment, embedding/reranker stubs, structured logging.  
5. **Validation Suite** – pytest coverage, manual playbook, documentation refresh.

Each milestone should end with a Gradio walkthrough showing the new artifacts and a FalkorDB fixture representing the expected graph state.

---

## 6. Deferred Items (tracked for Phase 02+)
- Embedding generation (Qwen3) & reranker integration (ensure stubs document expected inputs/outputs).
- Expanded NER modes (hybrid and DSPy-only) with runtime toggles and benchmarking.
- Community updates / higher-order grouping (explicitly out of scope for parity pass).
- Async refactor once synchronous parity stabilizes.

---

Delivering this plan keeps the custom pipeline aligned with Graphiti’s production flow while preserving room for future optimization and model experimentation.
