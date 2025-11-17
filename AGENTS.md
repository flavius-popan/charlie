# AGENTS.md - Navigation Guide for AI Agents

## What is Charlie?

Charlie is an end-user focused journaling aide that transforms journal entries into a deeply connected knowledge graph. It processes natural language text into entities (Person, Place, Organization, Activity), relationships, attributes, and summaries - all stored in a queryable graph database.

## Tech Stack

**Unique fusion of three technologies:**
- **MLX**: Local LLM inference (Apple Silicon optimized, no API calls)
- **DSPy**: Signature-based LLM programming with automated prompt optimization
- **graphiti-core**: Deterministic graph tooling, schema validation, and FalkorDB drivers

MLX inference lives in `mlx_runtime/` which provides a thin DSPy `BaseLM` wrapper and shared locking so we can rely on native DSPy adapters (Chat/JSON) without extra fallback code.

## Core Architecture

```
Journal Entry → add_journal() → 4-Stage Pipeline → FalkorDB Graph
                                  ↓
                    [Entities, Edges, Attributes, Summaries]
```

All processing is synchronous DSPy modules wrapped by async orchestrators for database I/O.

## Task-Based Navigation

### Understanding the Pipeline
- **Full architecture**: `pipeline/README.md` (comprehensive, read this first)
- **Entry point**: `pipeline/__init__.py::add_journal()` - orchestrates all 4 stages
- **Stage 1 (Entities)**: `pipeline/extract_nodes.py` - entity extraction + MinHash deduplication
- **Stage 2 (Relationships)**: `pipeline/extract_edges.py` - edge extraction + exact match resolution
- **Stage 3 (Attributes)**: `pipeline/extract_attributes.py` - type-specific attribute extraction
- **Stage 4 (Summaries)**: `pipeline/generate_summaries.py` - entity summary generation
- **Graph schema**: `pipeline/entity_edge_models.py` - Pydantic models for entities/edges

### Modifying LLM Behavior
- **MLX runtime**: `mlx_runtime/dspy_lm.py` - MLXDspyLM class, generation params
- **Lock + loader**: `mlx_runtime/__init__.py`, `mlx_runtime/loader.py` - shared lock + MLX loader
- **DSPy adapters**: use stock `dspy.ChatAdapter` / `dspy.JSONAdapter`; no custom fallback layer
- **Optimizer entrypoints**: `pipeline/optimizers/<stage>_optimizer.py::configure_dspy()` mirrors runtime config

### DSPy Optimization
- **Optimizers**: `pipeline/optimizers/<stage_name>_optimizer.py` - MIPROv2 prompt optimization
- **Saved prompts**: `pipeline/prompts/<stage>.json` - auto-loaded by stage modules
- **Pattern**: Each stage has pure `dspy.Module` (optimizable) + orchestrator (DB I/O)
- **Running**: `python -m pipeline.optimizers.extract_nodes_optimizer`

### Database Layer
- **Driver**: `pipeline/falkordblite_driver.py` - FalkorDB-Lite embedded graph database
- **Persistence**: `persist_episode_and_nodes()` wraps graphiti-core's bulk operations
- **Graph queries**: Uses FalkorDB's Cypher-like query language (not Neo4j)
- **Storage**: Local `.fdb` files, no server required

### Testing and Debugging
- **Interactive UI**: `python -m pipeline.gradio_ui` - test all 4 stages, view statistics, reset DB
- **End-to-end tests**: `tests/test_add_journal.py`
- **Per-stage tests**: `tests/test_extract_nodes.py`, `tests/test_extract_edges.py`, etc.
- **Failed generations**: Logged to `debug/failed_generation_YYYYMMDD_HHMMSS.json`

## Key Conventions

**Two-Layer Pattern**: Every pipeline stage separates LLM logic (pure `dspy.Module`, sync, no DB, optimizable) from orchestration (async DB I/O, episode creation, post-processing). This makes DSPy optimization 10-100x faster.

**Episode-First**: Full `EpisodicNode` created BEFORE extraction begins (graphiti-core pattern). All entities link back via episodic edges.

**Code Reuse**: Maximize graphiti-core imports for deterministic algorithms (deduplication, validation, datetime utilities). Only write custom code for LLM operations (DSPy signatures) and database queries (FalkorDB-specific).

**Thread Safety**: MLX is NOT thread-safe. `mlx_runtime.MLXDspyLM` acquires `MLX_LOCK` while calling `mlx_lm.generate`. DSPy modules are stateless and safe for concurrent use.

## Deprecated / Pending Removal

**distilbert-NER**: All NER integration code is pending removal. The project uses LLM-based extraction only (Stage 1's `EntityExtractor`). Do not integrate or reference distilbert/NER components.

## Common Workflows

**Add new entity type**: Edit `pipeline/entity_edge_models.py` → update Stage 3 attribute extraction → re-run optimizer if needed.

**Improve extraction quality**: Run stage optimizer (`pipeline/optimizers/<stage>_optimizer.py`) → prompts saved to `pipeline/prompts/` → auto-loaded on next run.

**Debug bad extractions**: Check `debug/failed_generation_*.json` → adjust `max_tokens` in settings → verify prompt quality → consider re-optimization.

**Change LLM model**: Edit `settings.DEFAULT_MODEL_PATH` (or call `MLXDspyLM(model_path=...)`). `mlx_runtime/loader.py::load_mlx_model()` handles download/caching.

**Query the graph**: Use `pipeline/falkordblite_driver.py::execute_query()` with Cypher-like syntax.

## Quick Reference

- **Main entry**: `pipeline/__init__.py::add_journal(content, group_id, ...)`
- **Pipeline docs**: `pipeline/README.md` (deep overview)
- **MLX runtime tips**: `docs/benchmark_pool.md`
- **Schema**: `pipeline/entity_edge_models.py`
- **Settings**: `pipeline/settings.py` (max_tokens, defaults, model path)
- **DSPy config**: Configured once at import, shared across all stages
