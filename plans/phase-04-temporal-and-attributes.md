Phase 04 – Temporal & Contextual Enrichment
===========================================

**Outcome**: Augment the graph hosted by FalkorDBLite with temporal metadata, typed attributes, and optional community detection while preserving auditability. Offer a Gradio surface to validate enriched outputs and time-aware queries.

Implementation Steps
--------------------
Review temporal handling and attribute experiments in `research/` (e.g., graph backend and reranking notes) and audit the `.venv` Graphiti models/utilities to mirror the current API surface before making changes.
1. **Temporal Extraction**
   - Implement `ExtractTemporalModule` to infer `valid_at`, `invalid_at`, and `expired_at` for edges.
   - Propagate episode reference times as defaults and capture ambiguity flags for downstream review.
   - Update pipeline to version historical edges instead of mutating in place.

2. **Attribute Population**
   - Implement `ExtractAttributesModule` with schema-aware rules per entity type.
   - Enforce type hints and validation before persisting attribute dictionaries.
   - Support partial updates so new attributes can merge without clobbering prior values.
   - Reuse DistilBERT NER outputs when running in NER-only or hybrid modes to seed attribute defaults (e.g., `labels=["Person"]`) and flag low-confidence entities for manual review, while preserving DSPy-derived attributes when NER is disabled.

3. **Community & Higher-Order Signals**
   - Add optional community detection or grouping heuristics (e.g., label propagation).
   - Store community memberships as `CommunityNode`/`CommunityEdge` records with lineage.
   - Record scoring metadata so adjustments can be made without losing traceability.

4. **Query & API Extensions**
   - Extend FalkorDBLite queries to support time-travel lookups and attribute filters (handling embedded connection contexts).
   - Ensure dedupe audit trails are compatible with temporal snapshots.
   - Document expected SLAs and fallback behavior when temporal data is missing.

Testing Strategy
----------------
- Create fixtures that cover overlapping time ranges, contradictory facts, and missing timestamps.
- Add regression tests for attribute validation (happy path, type mismatch, merge semantics).
- Verify DistilBERT-derived labels feed into attribute/type assignments across all extraction modes while preserving override pathways.
- Measure community detection performance on synthetic dense graphs to avoid regressions.
- Verify API/query helpers return correct results for time-bounded and attribute-filtered requests against FalkorDBLite.

Gradio Checkpoint
-----------------
- Extend the UI into `gradio_temporal_attributes.py`.
- Provide controls for selecting a timestamp, viewing the corresponding graph slice, and inspecting attribute evolution.
- Visualize community groupings with color-coding and expose attribute diffs per entity.
- Include downloadable CSV/JSON exports summarizing temporal spans and attribute provenance.

Deliverables
------------
- Temporal-aware pipeline capable of maintaining historical facts without data loss in FalkorDBLite.
- Typed attributes for priority entity classes with validation tooling.
- Optional community insights documented with clear entry/exit criteria.

Validation Gate
---------------
- Review temporal semantics, attribute merge behavior, and FalkorDBLite query extensions with stakeholders.
- Pause further development until time-aware outputs and Gradio diagnostics are approved.
