# Graphiti Pipeline Modules

Modular DSPy-based implementation of the Graphiti knowledge graph ingestion pipeline for local MLX inference.

## Overview

Each module in this directory represents a discrete stage in the pipeline, designed to:
- Accept structured inputs (follows graphiti-core's episode-first convention)
- Process data through DSPy signatures with Outlines structured output
- Return typed outputs compatible with downstream stages
- Maximize code reuse from graphiti-core utilities (deduplication, validation, etc.)
- Use async for database I/O while keeping LLM inference synchronous

## Quick Start

```python
import asyncio
from pipeline import add_journal

async def main():
    result = await add_journal(
        content="Today I met with Sarah at Stanford to discuss AI ethics...",
        group_id="user_123"
    )
    print(f"Episode: {result.episode.uuid}")
    print(f"Extracted {len(result.nodes)} entities")

asyncio.run(main())
```

## Pipeline Entry Point

### `add_journal()` - Main Pipeline Orchestrator

Entry point analogous to graphiti-core's `add_episode()`. Accepts plain text journal content and orchestrates all pipeline stages.

**Input**: Journal text, optional group_id, reference time, entity types
**Output**: `AddJournalResults` with episode, nodes, uuid_map, metadata

**Usage:**
```python
from pipeline import add_journal

result = await add_journal(
    content="Journal entry text...",
    group_id="user_123",  # Optional, defaults to FalkorDB default
    entity_types=None,     # Optional custom entity schemas
    excluded_entity_types=None  # Optional types to exclude
)
```

## Pipeline Stages

```
add_journal() → ExtractNodes → ExtractEdges → ExtractAttributes → GenerateSummaries → Database
```

### Stage 1: `extract_nodes.py` - Entity Extraction & Resolution

Accepts journal text, creates episode, extracts entities, resolves against existing graph.

**Input**: Journal text, optional reference time and entity types
**Output**: `ExtractNodesOutput` with episode, resolved nodes, UUID mappings

Internal stage called by `add_journal()`. Can also be used standalone:
```python
from pipeline import ExtractNodes

extractor = ExtractNodes(group_id="user_123")
result = await extractor(content="Today I met with Sarah...")
```

## Architecture

- **Async modules**: Database I/O is async, DSPy signatures stay sync
- **Episode-first**: Create full `EpisodicNode` before processing (follows graphiti-core)
- **Code reuse**: Import graphiti-core validators, dedup helpers, minimize custom code
- **dspy_outlines**: Uses existing adapter, no modifications needed
- **One event loop**: All modules share single async event loop (no conflicts)
