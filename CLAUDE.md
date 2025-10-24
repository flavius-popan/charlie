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

The `dspy_outlines/` module bridges DSPy, Outlines, and MLX to enable guaranteed structured output from local models.

**Architecture**: `DSPy Signature → OutlinesAdapter → OutlinesLM → Outlines → MLX`

**For detailed documentation**, see [`dspy_outlines/README.md`](dspy_outlines/README.md), which covers:
- Component responsibilities and thread safety
- MLX locking requirements (MLX is NOT thread-safe)
- Implementation details and design decisions
- Usage examples and troubleshooting

**Quick overview**:
- **OutlinesAdapter**: Extracts Pydantic schemas from DSPy signatures
- **OutlinesLM**: Executes constrained generation (requires thread locking for MLX safety)
- **Schema Extractor**: Runtime introspection of signature output fields
- **MLX Loader**: Loads quantized models from `.models/` directory

### Application Entry Points

**dspy-poc.py** - CLI interface
- Interactive REPL for knowledge graph extraction
- Multi-line input with double-Enter submission
- JSON output of extracted graphs

**gradio_app.py** - Web UI
- Graph visualization using Graphviz
- Real-time extraction with visual + JSON output
- Example prompts provided

Both applications use the same pattern:
1. Define Pydantic models for output structure
2. Create DSPy Signature with Pydantic-typed output field
3. Configure: `dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())`
4. Use: `dspy.Predict(Signature)` - constrained generation happens automatically

### Important Notes

**Outlines-core Override**: Temporarily using outlines-core 0.2.13 via git (see `pyproject.toml`) for MLX type-safety fixes. Remove when outlines updates to >= 0.2.12.

**Model Storage**: MLX models are in `.models/` directory (not default location).

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

See [`dspy_outlines/README.md`](dspy_outlines/README.md#usage-examples) for examples.

Quick pattern:
1. Define Pydantic models
2. Create DSPy Signature with Pydantic-typed output field
3. `dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())`
4. Use `dspy.Predict(Signature)`

### Testing Changes

When modifying `dspy_outlines/`, see [`dspy_outlines/README.md`](dspy_outlines/README.md#testing) for test guidance.
