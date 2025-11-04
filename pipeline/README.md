# Graphiti Pipeline Modules

**Goal**: Completely local MLX inference pipeline for knowledge graph ingestion, replicating graphiti-core's functionality without remote LLM APIs.

## Architecture

This pipeline reimplements graphiti-core's ingestion stages using:
- **DSPy modules** for each pipeline stage
- **dspy_outlines adapter** for structured output via Outlines constrained generation
- **MLX** for local LLM inference (no API calls)
- **graphiti-core utilities** for validators, deduplication, and graph operations (maximize code reuse)

### Design Pattern

Each stage is a `dspy.Module` that:
1. **Accepts structured input** from previous stage (or plain text for entry point)
2. **Performs ONE major LLM operation** via DSPy signature (if needed)
3. **Returns typed output** compatible with next stage
4. **Reuses graphiti-core code** wherever possible (only customize LLM extraction logic)
5. **Uses async for I/O**, sync for LLM inference (MLX_LOCK handled by dspy_outlines)

**Episode-First Convention**: Following graphiti-core's pattern, create the full `EpisodicNode` object BEFORE any extraction begins. All downstream operations reference this episode.

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

### `add_journal()` - Orchestrator

Analogous to graphiti-core's `add_episode()`. Entry point that accepts plain text and orchestrates all stages.

**Usage:**
```python
from pipeline import add_journal

result = await add_journal(
    content="Journal entry text...",
    group_id="user_123",  # Optional, defaults to FalkorDB default '\\_'
    entity_types=None,     # Optional custom entity schemas
    excluded_entity_types=None  # Optional types to exclude
)
# Returns: AddJournalResults(episode, nodes, uuid_map, metadata)
```

## Pipeline Stages

```
Journal Text → add_journal() → ExtractNodes → ExtractEdges → ExtractAttributes → GenerateSummaries → Database
```

### Stage 1: Extract Nodes (✓ Implemented)

**Module**: `extract_nodes.py` / `ExtractNodes`
**Analogous to**: graphiti-core's `extract_nodes()` + `resolve_extracted_nodes()`

Creates episode, extracts entities via DSPy signature, resolves duplicates using graphiti-core's MinHash LSH.

**LLM Call**: Entity extraction with type classification
**Reused Code**: `_build_candidate_indexes()`, `_resolve_with_similarity()`, validators
**Custom Code**: `EntityExtractionSignature`, episode creation, async DB wrappers

```python
# Can be used standalone for testing
extractor = ExtractNodes(group_id="user_123")
result = await extractor(content="Today I met with Sarah...")
```

### Stage 2: Extract Edges (TODO)

**Analogous to**: graphiti-core's `extract_edges()` + `resolve_extracted_edges()`

Extract relationships between resolved entities.

### Stage 3: Extract Attributes (TODO)

**Analogous to**: graphiti-core's `extract_attributes_from_nodes()`

Enrich entity nodes with attributes.

### Stage 4: Generate Summaries (TODO)

**Analogous to**: graphiti-core's summary generation in node operations

Generate entity summaries.

### Stage 5: Database Persistence (TODO)

Save all graph elements to FalkorDB.

## Implementation Notes

- **One DSPy config**: Configure once at module import, shared across all stages
- **Async/sync hybrid**: Database queries async, DSPy signatures sync (MLX_LOCK in adapter)
- **Single event loop**: All async stages share one event loop (no conflicts)
- **Code reuse first**: Import graphiti-core utilities before writing custom code
- **Test coverage**: Both end-to-end (`test_add_journal.py`) and per-stage (`test_extract_nodes.py`) tests
