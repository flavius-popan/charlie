# Graphiti Pipeline Modules

Modular DSPy-based implementation of the Graphiti knowledge graph ingestion pipeline for local MLX inference.

## Overview

Each module in this directory represents a discrete stage in the pipeline, designed to:
- Accept structured inputs (follows graphiti-core's episode-first convention)
- Process data through DSPy signatures with Outlines structured output
- Return typed outputs compatible with downstream stages
- Maximize code reuse from graphiti-core utilities (deduplication, validation, etc.)
- Use async for database I/O while keeping LLM inference synchronous

## Modules

### `extract_nodes.py` - Entity Extraction & Resolution

Accepts journal text, creates episode, extracts entities, resolves against existing graph.

**Input**: Journal text, optional reference time and entity types
**Output**: `EpisodicNode`, resolved `EntityNode` objects, UUID mappings

**Usage:**
```python
import asyncio
from pipeline import ExtractNodes

async def main():
    extractor = ExtractNodes(group_id="user_123")
    result = await extractor(content="Today I met with Sarah at Stanford...")
    print(f"Episode: {result.episode.uuid}")
    print(f"Extracted {len(result.nodes)} entities")

asyncio.run(main())
```

## Architecture

- **Async modules**: Database I/O is async, DSPy signatures stay sync
- **Episode-first**: Create full `EpisodicNode` before processing (follows graphiti-core)
- **Code reuse**: Import graphiti-core helpers (dedup, validators), customize LLM logic only
- **dspy_outlines**: Uses existing adapter, no modifications needed
- **One event loop**: All modules share single async event loop (no conflicts)

## Pipeline Stages

```
Episode → ExtractNodes → ExtractEdges → ExtractAttributes → GenerateSummaries → Database
```

Each stage is a `dspy.Module` with async `forward()` method. Call with `await module(input)`, not `module.forward(input)`.
