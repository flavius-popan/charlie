# Entity Extraction Runaway Fix

## Problem

When processing journal entry 4/33, the MLX model got stuck in a recursive loop generating hundreds of repetitive entities:

```
"desire to feel good"
"desire to feel balanced"
"desire to feel safe"
"desire to feel calm"
... (hundreds more)
```

This caused:
1. **JSON truncation**: Generation exceeded token limit, producing incomplete JSON
2. **Validation failure**: Truncated JSON couldn't be parsed by Pydantic
3. **Processing failure**: Journal entry 4 failed to load

The raw JSON output was 27,486 characters with ~400+ entities before being cut off mid-string.

## Root Cause

The Graphiti `ExtractedEntities` Pydantic schema has no constraint on list length:

```python
class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]  # No max_length!
```

Small/quantized models (like Qwen2.5-3B-4bit) can get stuck in repetitive patterns during structured generation, especially when:
- The prompt is complex
- The model has limited capacity
- The schema allows unbounded lists

## Solution

**Applied runtime schema patch** to limit entity extraction before it goes off the rails.

### Implementation

Created `app/llm/schema_patches.py`:

```python
MAX_ENTITIES_PER_EXTRACTION = 50

def apply_entity_extraction_limits():
    """Monkey-patch ExtractedEntities to add max_length constraint."""

    class ConstrainedExtractedEntities(BaseModel):
        extracted_entities: list[ExtractedEntity] = Field(
            ...,
            max_length=MAX_ENTITIES_PER_EXTRACTION
        )

    # Replace in graphiti_core module
    extract_nodes.ExtractedEntities = ConstrainedExtractedEntities
```

### Integration

Patches are applied in `load_journals.py` **before** Graphiti initialization:

```python
from app.llm.schema_patches import apply_all_patches
apply_all_patches()
```

This ensures:
1. Outlines generates with constrained schema
2. Model forced to stop at 50 entities
3. JSON always completes within token budget
4. Validation succeeds

## Impact

**Before:**
- Journal 4: ❌ Validation error (truncated JSON at 27K chars)
- Entities extracted: ~400+ (incomplete)

**After:**
- Journal 4: ✅ Successfully processes
- Entities extracted: ≤50 (complete, valid)
- Generation time: ~3 min (same)

## Trade-offs

**Pros:**
- ✅ Prevents runaway generation
- ✅ Ensures valid JSON output
- ✅ No model changes needed
- ✅ Quick fix (~50 lines)

**Cons:**
- ⚠️ May truncate legitimate entity lists >50
- ⚠️ Monkey-patching is fragile across graphiti_core updates

### Is 50 Enough?

For journal entries:
- Most entries: 10-30 entities
- Complex entries: 30-50 entities
- **>50 entities**: Usually indicates extraction quality issue

If you hit the limit legitimately, increase `MAX_ENTITIES_PER_EXTRACTION` in `schema_patches.py`.

## Validation

Test that constraint works:

```bash
python3 -c "
from app.llm.schema_patches import apply_all_patches
from graphiti_core.prompts import extract_nodes
apply_all_patches()
schema = extract_nodes.ExtractedEntities.model_json_schema()
print(f'maxItems: {schema[\"properties\"][\"extracted_entities\"][\"maxItems\"]}')
"
```

Expected output: `maxItems: 50`

## Future Improvements

1. **Upstream fix**: Submit PR to graphiti-core adding configurable limits
2. **Prompt tuning**: Improve extraction prompts to discourage repetition
3. **Model upgrade**: Test larger/better quantized models (e.g., Q5, Q6)
4. **Detection**: Log warning when hitting limit to identify problem entries
