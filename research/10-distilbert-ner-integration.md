DistilBERT NER Integration Notes
================================

Overview
--------
The repository ships a lightweight ONNX Runtime wrapper (`distilbert_ner.py`) and local assets under `distilbert-ner-uncased-onnx/` to provide fast, CPU-friendly named-entity recognition. The model is a DistilBERT base encoder fine-tuned on CoNLL-2003 with BIO tagging, exported to ONNX (`onnx/model.onnx`, ~255 MB). The helper exposes two top-level entrypoints:

- `predict_entities(text: str, *, max_length=512, stride=64)` → List of word-level entities with label, confidence, offsets, and chunk metadata. The function tiles long inputs across multiple windows, deduplicates overlaps, and preserves original casing.
- `format_entities(entities, include_labels=False, include_confidence=False)` → human-readable display strings for UI/debugging.

Key Implementation Details
--------------------------
- **Model Loading**: `ModelLoader.ensure_downloaded` lazily syncs the ONNX file via `huggingface_hub` if missing; otherwise, it reuses the local copy. `load_session` builds an ORT session with `GraphOptimizationLevel.ORT_ENABLE_ALL` and CPU execution.
- **Tokenizer**: Uses HuggingFace `AutoTokenizer` from the on-disk DistilBERT vocab. Encoding returns NumPy arrays compatible with ONNX Runtime, capped at 512 tokens (with configurable stride to handle longer documents).
- **Chunking & Deduplication**: `predict_entities` slices text into overlapping windows; `_deduplicate_chunk_entities` merges duplicates within a chunk, while `_deduplicate_entities` resolves duplicates/noise across chunks. Confidence values come from a softmax over logits; BIO tags are collapsed into entity spans.
- **EntityExtractor**: Aggregates subword tokens, merges BIO sequences, handles punctuation spacing, and preserves original casing.
- **Thread Safety**: The module caches the ORT session and tokenizer at the module level. Although ORT allows concurrent inference, we should document safe reuse patterns (e.g., a single shared session guarded by read locks or per-thread sessions for high throughput).

Performance Characteristics
---------------------------
- In `gradio_app.py`, average end-to-end inference for short inputs (~300 tokens) is <100 ms on Apple Silicon CPU. Latency grows linearly with text length; stride reduction (e.g., 32) increases accuracy at the cost of 1.5–2× runtime.
- Memory footprint is modest (model + tokenizer + ORT overhead ~350 MB). This aligns with our goal of running entire pipelines locally without GPU access.
- Throughput can be increased via batching (multiple texts stacked along batch dimension). Current code processes one text at a time; pipeline integrations should consider vectorizing batched inference when ingesting multiple episodes.

Integration Opportunities
-------------------------
1. **Primary Extraction Mode**: Use `predict_entities` as the authoritative entity generator, bypassing DSPy for node discovery while retaining DSPy for relationship/attribute work. Requires normalization to Graphiti schemas and reconciliation with downstream dedupe.
2. **Hybrid Mode**: Merge DistilBERT detections with DSPy outputs—treat NER entities as candidates that DSPy validates or augments. This balances latency gains with LLM reasoning for complex spans.
3. **Hint/Steering Mode**: Feed high-confidence hints (surface forms, labels, offsets) into DSPy prompts to shrink search space and improve recall for high-frequency entity types.
4. **Type Priming**: Map DistilBERT labels (`PER`, `ORG`, `LOC`, `MISC`) to Graphiti label sets (`EntityNode.labels`) in any mode to keep typing consistent.
5. **Confidence-Based Filtering**: Set threshold defaults (e.g., ≥0.60) to filter noise. Flag sub-threshold detections for manual investigation in the Gradio checkpoints.
6. **Temporal/Attribute Hints**: For Phase 04, DistilBERT output can help differentiate entity types when selecting attribute schemas or applying community detection heuristics.
7. **Fallback Strategy**: When DistilBERT is unavailable (missing model, ORT failure), the pipeline should gracefully degrade to DSPy-only extraction and log warnings—critical for automation resilience.

Recommended Enhancements
------------------------
- **Session Manager**: Wrap the existing global session in a simple manager that supports warm-up, telemetry (latency, cache hits), and graceful shutdown. This dovetails with Phase 01’s configuration surface.
- **Batch API**: Add a `predict_batch(texts: list[str])` helper to amortize tokenizer and session overhead for multi-episode processing.
- **Caching**: Implement optional caching keyed by text hash (or episode UUID) to avoid recomputing NER results during iterative runs and in the Gradio validation loops.
- **Metrics Hook**: Emit counters/timers (duration, entity count, confidence histogram) for inclusion in Phase 05 optimization dashboards.
- **Mode Telemetry**: Instrument runtime switches between NER-only, hybrid, and DSPy-only extraction to support comparative analysis later in the plan.

Test Coverage
-------------
- `tests/test_distilbert_ner.py` provides extensive unit tests (deduplication, formatting, stride behavior, chunk overlap). Reuse these fixtures when integrating with DSPy to ensure changes preserve expected semantics.
- Add integration tests where DistilBERT hints flow into DSPy modules and FalkorDB persistence to catch serialization mismatches early.

Operational Considerations
--------------------------
- Verify that ONNX Runtime is available in deployment environments (or vendor the wheel). Document the ~255 MB download so CI images include it or skip tests with a clear marker.
- Track model provenance (HF repo, revision `refs/pr/4`) in release notes. If we upgrade the model, update checksums and regression baselines.
- When running in restricted environments, pre-download the model and commit instructions for offline usage.

Linkages to Plan Phases
-----------------------
- **Phase 01**: Baseline evaluation, configuration surface, caching policy, and articulation of the three extraction modes.
- **Phase 02**: Implement the mode toggles, expose UI comparisons, and collect latency/quality metrics for each option.
- **Phase 03**: Ensure dedupe logic accepts inputs from any mode and records provenance for audit trails.
- **Phase 04**: Seed attribute defaults/labels and provide QA overlays in Gradio regardless of the active extraction mode.
- **Phase 05**: Track NER contribution relative to DSPy-only baselines, including ONNX inference telemetry and mode-specific benchmarks.
