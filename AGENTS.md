# Repository Guidelines

  ## Project Overview
  Local-first knowledge graph ingestion pipeline targeting functional parity with graphiti-core's `add_episode()` while replacing API-based LLM calls with MLX inference. Uses DistilBERT NER for entities, DSPy signatures for facts/relationships, and reuses graphiti-core utilities and data models where possible.

  ## Project Layout
  - `graphiti_pipeline.py` – main ingestion pipeline orchestrating extraction, resolution, and persistence.
  - `entity_utils.py` & `falkordb_utils.py` – pipeline support for graph construction and database operations.
  - `dspy_outlines/` – core integration between DSPy, Outlines, and MLX; start with `README.md` for threading and constraint details.
  - `gradio_app.py` – quick UI demo for knowledge-graph extraction.
  - `dspy-poc.py` – CLI playground for experimenting with signatures.
  - `tests/` – pytest suites validating constrained generation and MLX behavior.
  - `distilbert_ner.py` & `distilbert-ner-uncased-onnx/` – distilled NER helper and local assets.
  - `pipeline-parity-report.md` – detailed analysis of pipeline vs graphiti-core.

  - Use Ruff (configured via `uv.lock`) and Python defaults: 4-space indents, type hints for new functions, docstrings only when clarity
  suffers.
  - Favor small helper functions inside `dspy_outlines/` when adding adapter logic; keep thread-safety comments near locking code.
  - Name DSPy signatures with intent (`ExtractKnowledgeGraph`, `StrictCount`) and Pydantic models with singular nouns.

  ## Testing
  - Default to `pytest tests/test_constrained_generation.py` when touching adapters or signature schemas.
  - For MLX changes, extend or run `tests/test_mlx.py`; skip-heavy tests mark known gaps.
  - Add regression tests beside the module you touch; mirror existing `test_*` naming.

  ## Commits & PRs
  - Follow short, present-tense commit subjects (see `git log --oneline`).
  - PRs should link related TODO items, summarize model/config changes, and attach before/after snippets or screenshots for UI work.
  - Leave experiments and artifacts in `plans/` or issue comments; keep main history tidy.

  ## Agent Notes
  Keep changes incremental. This codebase is still volatile, so leave breadcrumbs in TODOs or inline comments when you defer work, and
  note any MLX locking or Outlines constraint assumptions you rely on.
