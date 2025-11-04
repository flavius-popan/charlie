# Graphiti Pipeline Modules

**Goal**: Completely local MLX inference pipeline for knowledge graph ingestion, replicating graphiti-core's functionality without remote LLM APIs.

## Architecture

This pipeline reimplements graphiti-core's ingestion stages using:
- **DSPy modules** for each pipeline stage
- **dspy_outlines adapter** for structured output via Outlines constrained generation
- **MLX** for local LLM inference (no API calls)
- **graphiti-core utilities** for validators, deduplication, and graph operations (maximize code reuse)

### Design Pattern

Each stage follows a **two-layer architecture** for DSPy optimization:

1. **Pure `dspy.Module`**: LLM extraction only (sync, no DB/async). Optimizable with teleprompters.
2. **Orchestrator class**: Handles DB I/O, episode creation, post-processing. Accepts compiled modules.

This separation makes optimization 10-100x faster (no DB overhead) and metrics clearer (no post-processing dilution).

**Episode-First Convention**: Create full `EpisodicNode` BEFORE extraction begins (graphiti-core pattern).

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

**Module**: `extract_nodes.py`
**Analogous to**: graphiti-core's `extract_nodes()` + `resolve_extracted_nodes()`

**Architecture** (two-layer pattern):
1. **`EntityExtractor`** (dspy.Module): Pure LLM extraction. Sync, no DB. Optimizable.
2. **`ExtractNodes`** (orchestrator): Full pipeline with DB, episode creation, MinHash LSH resolution.

```python
# Standard usage
extractor = ExtractNodes(group_id="user_123")
result = await extractor(content="Today I met with Sarah...")

# With optimized EntityExtractor
entity_extractor = EntityExtractor()
compiled = optimizer.compile(entity_extractor, trainset=examples)
extractor = ExtractNodes(group_id="user_123", entity_extractor=compiled)
result = await extractor(content="...")
```

**Important**: Stage 1 extracts entity names and types ONLY. Custom attributes (e.g., Person.relationship_type, Emotion.specific_emotion) are extracted in Stage 3, following graphiti-core's separation of concerns.

**Pattern for Future Stages**: Repeat this two-layer design. Create a pure `dspy.Module` for each LLM operation, inject into orchestrator.

### Stage 2: Extract Edges (TODO)

**Analogous to**: graphiti-core's `extract_edges()` + `resolve_extracted_edges()`

Extract relationships between resolved entities.

### Stage 3: Extract Attributes (TODO)

**Module**: `extract_attributes.py` (not yet implemented)
**Analogous to**: graphiti-core's `extract_attributes_from_nodes()`

**Purpose**: Extract custom entity attributes based on entity type and episode context.

**Design**:
- For each resolved entity from Stage 1, extract type-specific attributes:
  - **Person**: `relationship_type` (e.g., "friend", "colleague", "family")
  - **Emotion**: `specific_emotion` (e.g., "anxiety", "joy"), `category` (e.g., "positive", "negative")
- Pass entity's Pydantic model (from `entity_edge_models.py`) as `response_model` to LLM
- LLM extracts attributes based on episode context
- Results merged into `EntityNode.attributes` dict

**Architecture** (follows two-layer pattern):
1. **`AttributeExtractor`** (dspy.Module): Pure LLM extraction. Takes entity name, type, and episode context. Returns Pydantic model with attributes.
2. **`ExtractAttributes`** (orchestrator): Iterates over resolved entities, calls AttributeExtractor for each, merges results.

**Example**:
```python
# For a Person entity "Sarah" in episode "Today I met with my friend Sarah"
# LLM extracts: {"relationship_type": "friend"}
# Merged into: EntityNode(name="Sarah", labels=["Entity", "Person"], attributes={"relationship_type": "friend"})
```

**Critical**: This stage follows graphiti-core's separation of concerns. Entity names/types are extracted in Stage 1, attributes in Stage 3. This enables optimizing each stage independently and matches graphiti-core's architecture.

### Stage 4: Generate Summaries (TODO)

**Analogous to**: graphiti-core's summary generation in node operations

Generate entity summaries.

### Stage 5: Database Persistence (TODO)

Save all graph elements to FalkorDB.

## Optimization

Each pipeline stage can be optimized using DSPy's BootstrapFewShot optimizer to improve prompt quality.

### Pattern

All optimizers follow this structure:

1. **Location**: `pipeline/optimizers/<stage_name>_optimizer.py`
2. **Prompts**: Saved to `pipeline/prompts/<stage_name>.json`
3. **Auto-loading**: Stage modules automatically load optimized prompts if present
4. **Template**: 8-section functional template (no shared utilities)

### Running Optimizers

```bash
# Optimize Stage 1: Entity Extraction
python -m pipeline.optimizers.extract_nodes_optimizer

# Future stages will follow same pattern
# python -m pipeline.optimizers.extract_edges_optimizer
```

**What happens:**
1. Baseline evaluation on training set
2. BootstrapFewShot optimization (5 demos)
3. Optimized evaluation
4. Prompts saved to `pipeline/prompts/<stage>.json`
5. Stage auto-loads optimized prompts on next run

### Optimizer Structure

Each optimizer script contains:

1. **configure_dspy()** - Setup LM and adapter
2. **build_trainset()** - 15-25 training examples
3. **<stage>_metric()** - Stage-specific evaluation metric (e.g., F1 for entities)
4. **optimize()** - Run BootstrapFewShot
5. **evaluate()** - Measure quality
6. **main()** - Orchestrate: baseline → optimize → evaluate → save

### Adding New Stage Optimizers

1. Copy `pipeline/optimizers/extract_nodes_optimizer.py` as template
2. Customize `build_trainset()` with stage-specific examples
3. Implement stage-specific metric function
4. Update imports to use your stage's dspy.Module
5. Set correct `PROMPT_OUTPUT` path
6. Add auto-load logic to your stage's Module.__init__()

## Testing UI

### Gradio Interactive UI

Test individual pipeline stages interactively with `pipeline/gradio_ui.py`:

```bash
python -m pipeline.gradio_ui
```

**Features:**
- Extract entities from journal text
- Toggle deduplication on/off
- View extraction statistics (exact/fuzzy matches, new entities)
- Inspect UUID mappings (provisional → resolved)
- Write results to database
- Reset database for clean testing

**Use cases:**
- Validate entity extraction accuracy
- Test deduplication behavior with/without existing graph data
- Inspect DSPy signature outputs before committing to database

## Debugging

### Debug Logging

Set log level to see detailed execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Shows DSPy adapter fallbacks, generation params
logging.basicConfig(level=logging.INFO)   # Default: shows stage progress, extraction results
```

### Failed Generations

When constrained generation fails (truncated JSON, validation errors), the full output is saved to:

```
debug/failed_generation_YYYYMMDD_HHMMSS.json
```

**Common issues:**
- **Truncated JSON**: Check if `max_tokens` in `settings.py` is too low
- **Invalid schema**: Model may need better prompt or examples
- **Hallucinations**: Model extracting entities from `previous_context` instead of `episode_content`

The `debug/` directory is gitignored.

## Implementation Notes

- **One DSPy config**: Configure once at module import, shared across all stages
- **Async/sync hybrid**: Database queries async, DSPy signatures sync (MLX_LOCK in adapter)
- **Single event loop**: All async stages share one event loop (no conflicts)
- **Code reuse first**: Import graphiti-core utilities before writing custom code
- **Test coverage**: Both end-to-end (`test_add_journal.py`) and per-stage (`test_extract_nodes.py`) tests
