# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Charlie is a knowledge graph extraction system that uses DSPy signatures with Outlines for constrained generation and MLX for local inference. The core innovation is the `dspy_outlines` integration layer that bridges DSPy's signature-based programming with Outlines' guaranteed structured output.

## Development Setup

### Environment

This project uses direnv for environment management. The `.envrc` file automatically activates the virtual environment located at `.venv/`.

### Dependencies

Install dependencies using uv:
```bash
uv sync
```

The project requires Python 3.13 and includes:
- DSPy for signature-based LLM programming
- Outlines for constrained generation
- MLX for Apple Silicon inference
- Graphiti-core for reference implementation as a case study
- Gradio for the UI

### Testing

Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/test_constrained_generation.py
```

Run a single test:
```bash
pytest tests/test_constrained_generation.py::test_knowledge_graph_extraction -v
```

## Code Architecture

### Core Integration Layer: `dspy_outlines/`

The integration bridges three frameworks:

**OutlinesLM** (`lm.py`) - Main LM implementation
- Inherits from `dspy.BaseLM` (not `dspy.LM` - see note below)
- Receives Pydantic schemas from OutlinesAdapter via `_outlines_schema` kwarg
- Routes to Outlines for constrained generation
- Wraps output with field name for DSPy adapter compatibility
- Returns validated JSON to DSPy

**OutlinesAdapter** (`adapter.py`) - Custom DSPy adapter
- Extends DSPy's ChatAdapter
- Extracts Pydantic schemas from signature output fields
- Passes schema and field name to OutlinesLM via lm_kwargs
- Enables constrained generation by bridging DSPy's adapter system with Outlines

**Schema Extraction** (`schema_extractor.py`)
- Inspects DSPy Signature classes at runtime
- Extracts Pydantic BaseModel from output field annotations
- Used by OutlinesAdapter to identify schemas

**MLX Model Loading** (`mlx_loader.py`)
- Loads quantized MLX models from `.models/` directory
- Creates Outlines wrapper via `outlines.from_mlxlm()`
- Default model: `mlx-community--Qwen3-4B-Instruct-2507-8bit`

### Application Entry Points

**dspy-poc.py** - CLI interface
- Interactive REPL for knowledge graph extraction
- Multi-line input with double-Enter submission
- JSON output of extracted graphs

**gradio_app.py** - Web UI
- Graph visualization using Graphviz
- Real-time extraction with visual + JSON output
- Example prompts provided

Both applications use the same DSPy signature pattern:
1. Define Pydantic models (Node, Edge, KnowledgeGraph)
2. Create DSPy Signature with Pydantic output field
3. Configure DSPy with both LM and adapter: `dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())`
4. Use `dspy.Predict(Signature)` - schema extraction and constrained generation happen automatically

### Important Implementation Notes

**BaseLM vs LM**: This project uses `dspy.BaseLM` instead of `dspy.LM` because:
- BaseLM is simpler and doesn't require LiteLLM
- Better for custom backends like Outlines+MLX
- Requires returning SimpleNamespace objects (not dicts) from `forward()`
- Usage tracking must be dict-convertible (use AttrDict for the usage field)

**Outlines-core Override**: The project temporarily overrides `outlines-core` to version 0.2.13 via git (see `pyproject.toml` `[tool.uv]` section). This includes MLX type-safety fixes from PR #230. Remove this override when outlines updates to outlines-core >= 0.2.12.

**Model Storage**: MLX models are stored in `.models/` directory (not the default location). This is noted as a future improvement in TODO.md.

## Common Workflows

### Running the CLI Demo
```bash
python dspy-poc.py
```

### Running the Gradio UI
```bash
python gradio_app.py
```

### Adding New Extraction Types

To add new structured extraction beyond knowledge graphs:

1. Define Pydantic models for your output structure
2. Create a DSPy Signature with your models as output fields
3. Initialize with `dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())`
4. Use `dspy.Predict(YourSignature)` - the adapter and LM handle the rest
5. No changes needed to `dspy_outlines/` layer

The schema extraction and constrained generation happen automatically through the adapter.

### Testing Changes to the Integration Layer

When modifying `dspy_outlines/`:
- Run `pytest tests/test_schema_extractor.py` for schema extraction logic
- Run `pytest tests/test_constrained_generation.py` for end-to-end integration
- The test suite includes real model inference (not mocked)
- `test_outlines_actually_constrains()` validates that constraints are enforced
