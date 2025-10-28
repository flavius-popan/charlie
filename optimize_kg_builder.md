# Knowledge Graph Builder Optimization Plan

## Current State

The three-tier adapter is working correctly but **always falls back to tier 3 (constrained generation)** due to zero-shot prompting:

- **Tier 1 (Chat)**: Fails - model outputs JSON without field markers `[[ ## graph ## ]]`
- **Tier 2 (JSON)**: Fails - model outputs `{"nodes": [...], "edges": [...]}` instead of `{"graph": {"nodes": [...], "edges": [...]}}`
- **Tier 3 (Constrained)**: Succeeds - Outlines FSM forces correct structure

**Problem**: No few-shot examples → model doesn't learn output format → slow tier 3 every time

## Solution: DSPy Prompt Optimization

Use DSPy's optimizers to generate few-shot examples that teach the model the correct output format.

## Implementation Steps

### 1. Decouple DSPy Module from Gradio (Priority)

**Current**: `gradio_app.py` has tightly coupled:
- Pydantic models (Node, Edge, KnowledgeGraph)
- DSPy signature (ExtractKnowledgeGraph)
- Predictor instantiation

**Goal**: Extract into standalone module (e.g., `kg_extraction.py`) with:
```python
# kg_extraction.py
class ExtractKnowledgeGraph(dspy.Signature):
    """..."""
    text: str = dspy.InputField(...)
    known_entities: Optional[List[str]] = dspy.InputField(...)
    graph: KnowledgeGraph = dspy.OutputField(...)

class KGExtractor:
    def __init__(self, optimized_prompts_path=None):
        self.predictor = dspy.Predict(ExtractKnowledgeGraph)
        if optimized_prompts_path:
            self.predictor.load(optimized_prompts_path)

    def extract(self, text, known_entities=None):
        return self.predictor(text=text, known_entities=known_entities)
```

### 2. Create Optimization Script

**File**: `scripts/optimize_kg_prompts.py`

**Steps**:
1. Create labeled dataset (10-20 examples):
   ```python
   training_data = [
       {"text": "Alice met Bob at coffee shop.", "graph": KnowledgeGraph(...)},
       {"text": "John works at Microsoft.", "graph": KnowledgeGraph(...)},
       # ... more examples
   ]
   ```

2. Use DSPy optimizer:
   ```python
   from dspy.teleprompt import BootstrapFewShot

   optimizer = BootstrapFewShot(
       metric=kg_quality_metric,  # Define metric for valid graphs
       max_bootstrapped_demos=5,
       max_labeled_demos=3
   )

   optimized_extractor = optimizer.compile(
       student=dspy.Predict(ExtractKnowledgeGraph),
       trainset=training_data
   )

   # Export optimized prompts
   optimized_extractor.save("prompts/kg_extraction_optimized.json")
   ```

3. Test tier success rates before/after optimization

### 3. Metric Function

Define what makes a "good" knowledge graph:
```python
def kg_quality_metric(example, prediction, trace=None):
    """Returns True if graph is valid and matches expected structure."""
    if not prediction.graph:
        return False

    # Check structure
    if not prediction.graph.nodes or not prediction.graph.edges:
        return False

    # Check format (should have field wrapper)
    # Additional validation as needed

    return True
```

### 4. Integration

Update `gradio_app.py`:
```python
from kg_extraction import KGExtractor

# Load optimized prompts if available
extractor = KGExtractor(optimized_prompts_path="prompts/kg_extraction_optimized.json")
result = extractor.extract(text, known_entities=hints)
```

## Expected Outcomes

**Before optimization**:
- Tier 1: 0% success
- Tier 2: 0% success
- Tier 3: 100% success (slow)

**After optimization**:
- Tier 1: 60-80% success (fast)
- Tier 2: 15-30% success (medium)
- Tier 3: 5-10% fallback (slow safety net)

## References

- DSPy BootstrapFewShot: https://dspy-docs.vercel.app/docs/building-blocks/optimizers#teleprompt-bootstrapfewshot
- DSPy saving/loading: https://dspy-docs.vercel.app/docs/building-blocks/modules#serialization

## Notes

- Start with small dataset (10-20 examples) - DSPy bootstrap generates synthetic examples
- Monitor adapter metrics (already tracked in `adapter.metrics`) to measure improvement
- Consider `COPRO` optimizer if BootstrapFewShot doesn't improve tier 1/2 enough
