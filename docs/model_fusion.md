# Model Fusion: NER + LLM Knowledge Graph Extraction

## Overview

Charlie demonstrates hybrid model fusion by combining fast NER (DistilBERT-ONNX) with LLM-based knowledge graph extraction (DSPy+Outlines). The NER model provides deterministic entity detection (PER, ORG, LOC, MISC), while the LLM handles relationship extraction and context understanding.

## Architecture

**Prompt Injection Strategy**: NER entities are passed as optional hints to the LLM, allowing it to use them as context without hard constraints.

### Flow

1. **NER Extraction**: `distilbert_ner.predict_entities(text)` detects entities with confidence scores
2. **Hint Formatting**: `format_entities()` converts entities to strings with configurable detail:
   - Plain text: `["Microsoft", "Satya Nadella"]`
   - With labels: `["Microsoft (Organization)", "Satya Nadella (Person)"]`
   - With confidence: `["Microsoft (entity_type:Organization, conf:0.99)"]`
3. **LLM Context**: Hints passed to DSPy signature's `entity_hints: Optional[List[str]]` field
4. **Graph Generation**: LLM generates full knowledge graph (nodes + edges) using hints as context

### Implementation

**DSPy Signature** (`gradio_app.py:45-54`):
- Added optional `entity_hints` input field to `ExtractKnowledgeGraph` signature
- LLM receives hints via prompt but can override or ignore them (flexibility vs constraints)

**Extraction Function** (`gradio_app.py:61-117`):
- Runs NER first, always
- Formats hints based on UI toggle states (use_hints, include_labels, include_confidence)
- Passes formatted hints (or `None`) to extractor
- Returns NER output, formatted hints, graph visualization, and final JSON

**UI Controls** (`gradio_app.py:133-146`):
- Toggle: Enable/disable hint usage
- Toggle: Include entity type labels in hints
- Toggle: Include confidence scores (requires labels enabled)
- Separate viewers for NER output, hints sent to LLM, graph viz, and JSON

## Why Prompt Injection?

- **No framework modifications**: Works with existing DSPy+Outlines three-tier fallback
- **Flexibility**: LLM can correct NER errors or add missed entities
- **Speed**: Single LLM call (vs sequential enrichment requiring two calls)
- **Simplicity**: Optional field - system works with or without hints
