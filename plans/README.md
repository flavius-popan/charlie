Graphiti DSPy Pipeline Plan
===========================

This directory now houses a staged implementation plan for the custom Graphiti knowledge-graph pipeline. Each phase is isolated so the work can be validated independently, and every phase includes a dedicated Gradio checkpoint that mirrors the exploratory UI patterns in `gradio_app.py`. Start at Phase 01 and work downward. When refining any phase, cross-reference the background material under `research/` (e.g., FalkorDB driver notes, embedding integration) and inspect the installed Graphiti packages inside `.venv` to validate assumptions against live code.

- `phase-01-foundations.md` – establish architecture guardrails, tooling, and a baseline Gradio harness.
- `phase-02-core-extraction.md` – stand up the DSPy-driven extraction loop with FalkorDB persistence via the embedded `falkordblite` runtime.
- `phase-03-deduplication.md` – add entity/edge resolution and cross-episode reconciliation.
- `phase-04-temporal-and-attributes.md` – layer temporal reasoning, attributes, and communities.
- `phase-05-optimization.md` – optimize, benchmark, and harden the end-to-end system.

- `falkordblite-evaluation.md` – deep dive on the embedded FalkorDBLite library, constraints, and how it affects our adapters.
- `graphiti-models-archive.md` – archived, unvetted legacy plan; reference specific sections only to avoid context bloat.

All unverified prototype code that previously lived in `graphiti-models.md` was moved to `snippets/unverified_graphiti_pipeline_examples.py`. Treat those snippets as exploratory only—they intentionally do **not** prescribe the production implementation.
