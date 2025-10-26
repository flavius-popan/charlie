# Model Fusion Architecture: NER-ONNX + DSPy+Outlines

## Executive Summary

This document outlines strategies for fusing DistilBERT-NER (ONNX) with DSPy+Outlines knowledge graph extraction. The NER model provides fast, deterministic entity extraction (PER, ORG, LOC, MISC), while the LLM adds relationships and enrichment. By leveraging each model's strengths, we can achieve better speed and reliability than either alone.

**Key Finding**: DSPy's three-tier adapter (ChatAdapter → JSONAdapter → OutlinesJSON) makes prompt injection the most pragmatic fusion approach, requiring no framework modifications.

## Component Analysis

### NER-ONNX Model (inference_onnx.py)
- **Output**: `[{"label": "PER", "text": "Alice", "start_token": 0, "end_token": 0}, ...]`
- **Entities**: PER (person), ORG (organization), LOC (location), MISC (miscellaneous)
- **Performance**: ~10ms for 128 tokens, deterministic
- **Location**: `inference_onnx.py:108-157` (predict method)

### DSPy+Outlines Stack
- **Signature**: Pydantic-typed input/output fields (dspy-poc.py:34-40)
- **Adapter Flow**: ChatAdapter → JSONAdapter → OutlinesJSON (7x slower fallback)
- **Constraint**: JSONAdapter uses OpenAI structured outputs or JSON mode (fast, preferred)
- **Validation**: All output fields currently required (`adapter.py:269`)
- **Location**: `dspy_outlines/adapter.py:41-116` (three-tier __call__)

### Pydantic Capabilities
- **Partial Models**: `Optional[T] = None` for optional fields
- **Construction**: `model_construct()` (no validation), `model_copy(update={...})` for incremental updates
- **Tracking**: `model_fields_set` tracks explicitly set fields
- **Serialization**: `model_dump(exclude_unset=True)` omits unset fields

## Fusion Strategies

### Strategy 1: Prompt Injection (Recommended)

**Concept**: Run NER first, inject entities into LLM prompt as structured context.

**Architecture**:
```python
# Pseudocode flow
def extract_graph(text: str) -> KnowledgeGraph:
    # 1. NER extraction
    ner_entities = ner_model.predict(text)  # Fast: ~10ms

    # 2. Format as context
    entity_context = format_entities_for_prompt(ner_entities)
    # Example: "Detected entities: Alice (PER), Apple Inc. (ORG), Cupertino (LOC)"

    # 3. Enhanced DSPy signature
    result = extractor(
        text=text,
        entity_hints=entity_context  # New input field
    )

    # 4. LLM generates full graph (nodes + edges)
    return result.graph
```

**Implementation Changes**:
- Add `entity_hints: str = dspy.InputField()` to signature
- Format NER output as readable text (not JSON)
- LLM uses hints to generate accurate nodes + edges

**Pros**:
- No adapter modifications required
- Works with existing three-tier fallback
- LLM can override NER errors (flexible)
- Single model call (fast)

**Cons**:
- NER entities are "hints" not guarantees
- Prompt token overhead (~50-100 tokens)
- LLM might ignore or hallucinate over hints

**Performance**: NER (10ms) + LLM (300-2000ms) = **~350ms total** (ChatAdapter tier)

---

### Strategy 2: Sequential Enrichment

**Concept**: NER extracts nodes → construct partial Pydantic model → LLM fills edges only.

**Architecture**:
```python
# Pseudocode
def extract_graph(text: str) -> KnowledgeGraph:
    # 1. NER → nodes
    ner_entities = ner_model.predict(text)
    nodes = convert_ner_to_nodes(ner_entities)

    # 2. New signature for edges only
    class ExtractEdges(dspy.Signature):
        text: str = dspy.InputField()
        known_nodes: str = dspy.InputField(desc="JSON of known nodes")
        edges: List[Edge] = dspy.OutputField()

    # 3. LLM generates edges referencing nodes
    result = edge_extractor(
        text=text,
        known_nodes=json.dumps([n.model_dump() for n in nodes])
    )

    # 4. Merge
    return KnowledgeGraph(nodes=nodes, edges=result.edges)
```

**Implementation Changes**:
- Create new `ExtractEdges` signature (edges only)
- Convert NER entities to `Node` objects: `inference_onnx.py:258` entity format → `Node(id=text, label=text, properties={"type": ner_label})`
- Merge NER nodes + LLM edges

**Pros**:
- NER nodes guaranteed present (deterministic)
- Smaller LLM output (edges only, faster)
- Clear separation of concerns

**Cons**:
- Two-step process (latency)
- LLM must reference node IDs correctly (potential mismatch)
- Requires new signature definition

**Performance**: NER (10ms) + LLM (200-1500ms) = **~250ms total** (smaller output helps)

---

### Strategy 3: Partial Schema Constraint

**Concept**: Make `nodes` field optional in Pydantic schema, pre-populate in prompt.

**Architecture**:
```python
# Modified Pydantic models
class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Node]] = None  # Optional!
    edges: List[Edge]  # Required

# Signature remains same
class ExtractKnowledgeGraph(dspy.Signature):
    text: str = dspy.InputField()
    graph: KnowledgeGraph = dspy.OutputField()

# Execution
def extract_graph(text: str) -> KnowledgeGraph:
    ner_entities = ner_model.predict(text)
    nodes = convert_ner_to_nodes(ner_entities)

    # Inject nodes into prompt
    result = extractor(
        text=f"{text}\n\nKnown nodes: {json.dumps([n.model_dump() for n in nodes])}"
    )

    # Merge: use LLM edges + NER nodes (override if LLM provided nodes)
    return KnowledgeGraph(
        nodes=result.graph.nodes or nodes,  # Fallback to NER
        edges=result.graph.edges
    )
```

**Implementation Changes**:
- Change `nodes: List[Node]` → `nodes: Optional[List[Node]] = None` in Pydantic model
- Modify `adapter.py:269` to allow missing optional fields:
  ```python
  # Current: strict check
  if fields.keys() != signature.output_fields.keys():
      raise AdapterParseError(...)

  # New: allow missing optional fields
  required_fields = {k for k, v in signature.output_fields.items() if not is_optional(v.annotation)}
  if not required_fields.issubset(fields.keys()):
      raise AdapterParseError(...)
  ```

**Pros**:
- LLM focuses on edges (smaller output)
- NER nodes serve as fallback
- Single signature, single call

**Cons**:
- Requires adapter modification (moderate complexity)
- LLM might generate nodes anyway (wasted tokens)
- Validation complexity increases

**Performance**: NER (10ms) + LLM (200-1500ms) = **~250ms total**

---

### Strategy 4: Two-Signature Pipeline

**Concept**: Parallel execution of NER and LLM with separate signatures, merge results.

**Architecture**:
```python
# Signatures
class ExtractNodes(dspy.Signature):
    text: str = dspy.InputField()
    nodes: List[Node] = dspy.OutputField()

class ExtractEdges(dspy.Signature):
    text: str = dspy.InputField()
    edges: List[Edge] = dspy.OutputField()

# Execution
def extract_graph(text: str) -> KnowledgeGraph:
    # Run in parallel (if threading supported)
    ner_entities = ner_model.predict(text)  # ~10ms
    llm_edges = edge_extractor(text=text)   # ~200ms

    # Convert & merge
    nodes = convert_ner_to_nodes(ner_entities)
    return KnowledgeGraph(nodes=nodes, edges=llm_edges.edges)
```

**Implementation Changes**:
- Create `ExtractNodes` and `ExtractEdges` signatures
- NER replaces `ExtractNodes` LLM call (faster)
- Optional: parallel execution with threading

**Pros**:
- Clean separation
- NER replaces slower LLM node extraction
- Can optimize each signature independently

**Cons**:
- More boilerplate (two signatures)
- Edges must reference node IDs (coordination needed)
- No LLM validation of nodes

**Performance**: Max(NER=10ms, LLM=200ms) = **~200ms** (if parallel)

---

### Strategy 5: Post-Processing Merge

**Concept**: Run both NER and full LLM extraction, merge results with conflict resolution.

**Architecture**:
```python
def extract_graph(text: str) -> KnowledgeGraph:
    # 1. Both models run independently
    ner_entities = ner_model.predict(text)
    llm_graph = llm_extractor(text=text).graph

    # 2. Merge with priority rules
    merged_nodes = merge_nodes(
        ner_nodes=convert_ner_to_nodes(ner_entities),
        llm_nodes=llm_graph.nodes,
        strategy="prefer_ner"  # or "prefer_llm", "union"
    )

    # 3. Use LLM edges as-is
    return KnowledgeGraph(nodes=merged_nodes, edges=llm_graph.edges)
```

**Implementation Changes**:
- Write `merge_nodes()` function with conflict resolution
- Strategies: `prefer_ner`, `prefer_llm`, `union`, `intersect`
- Node matching by text similarity (fuzzy matching)

**Pros**:
- Maximum robustness (two independent sources)
- Can compare NER vs LLM accuracy
- Flexible conflict resolution

**Cons**:
- Slowest (both models run fully)
- Merge complexity (fuzzy matching, deduplication)
- Wastes LLM tokens on node generation

**Performance**: NER (10ms) + LLM (300-2000ms) = **~350ms** (no speed gain)

## Performance & Reliability Matrix

| Strategy | Speed (est.) | Reliability | Complexity | Adapter Changes |
|----------|--------------|-------------|------------|-----------------|
| 1. Prompt Injection | ~350ms | Medium | Low | None |
| 2. Sequential Enrichment | ~250ms | High | Medium | None (new signature) |
| 3. Partial Schema | ~250ms | High | Medium | Moderate (validation) |
| 4. Two-Signature Pipeline | ~200ms | High | Medium | None (new signatures) |
| 5. Post-Processing Merge | ~350ms | Highest | High | None (merge logic) |

**Speed Factors**:
- ChatAdapter tier (no Outlines): ~300ms (typical)
- JSONAdapter tier: ~400ms (JSON mode)
- OutlinesJSON tier: ~2100ms (7x slower, constrained)
- NER always: ~10ms (negligible)

## Recommendations

### Recommended Implementation Order

**Phase 1: Proof of Concept (Strategy 1)**
- Implement Prompt Injection first
- Zero adapter changes
- Validates NER integration
- Measures actual LLM response to entity hints

**Phase 2: Optimization (Strategy 2 or 4)**
- If Phase 1 shows LLM ignores hints → Strategy 2 (Sequential Enrichment)
- If Phase 1 works but slow → Strategy 4 (Two-Signature Pipeline)
- Both achieve ~200-250ms with high reliability

**Phase 3: Advanced (Strategy 3 or 5)**
- Strategy 3 if complex graphs need flexible node handling
- Strategy 5 for research/comparison (not production due to cost)

### Key Implementation Notes

**NER-to-Node Conversion** (all strategies need this):
```python
# Reference: inference_onnx.py:158-257 (_extract_entities)
def convert_ner_to_nodes(ner_entities: list[dict]) -> list[Node]:
    nodes = []
    for i, entity in enumerate(ner_entities):
        nodes.append(Node(
            id=f"ner_{i}",  # or hash of text
            label=entity["text"],
            properties={"type": entity["label"]}  # PER/ORG/LOC/MISC
        ))
    return nodes
```

**Adapter Modification for Optional Fields** (Strategy 3):
```python
# In adapter.py:269, replace strict check with:
from typing import get_origin, get_args

def is_optional(annotation):
    return get_origin(annotation) is Union and type(None) in get_args(annotation)

# Then in _parse_json():
required_fields = {
    k for k, v in signature.output_fields.items()
    if not is_optional(v.annotation)
}
if not required_fields.issubset(fields.keys()):
    raise AdapterParseError(...)
```

**Entity Context Formatting** (Strategy 1):
```python
def format_entities_for_prompt(ner_entities: list[dict]) -> str:
    entity_groups = {}
    for e in ner_entities:
        entity_groups.setdefault(e["label"], []).append(e["text"])

    parts = []
    for label, entities in entity_groups.items():
        parts.append(f"{label}: {', '.join(entities)}")

    return "Detected entities: " + "; ".join(parts)
```

## Architecture Decision Points

**Choose Strategy 1 (Prompt Injection) if**:
- Want fastest implementation
- LLM is reliable with hints
- Flexibility > guarantees

**Choose Strategy 2 (Sequential Enrichment) if**:
- Need guaranteed NER nodes
- LLM must only generate edges
- Clear separation of concerns preferred

**Choose Strategy 4 (Two-Signature) if**:
- Performance critical (fastest)
- Can handle parallel execution
- Want independent optimization

## References

**Existing Code Locations**:
- NER predict: `inference_onnx.py:108-157`
- NER entity extraction: `inference_onnx.py:158-257`
- DSPy signature example: `dspy-poc.py:34-40`
- OutlinesAdapter three-tier: `dspy_outlines/adapter.py:41-116`
- JSONAdapter parsing: `dspy_outlines/adapter.py:229-277`
- Pydantic models: `dspy-poc.py:15-30`

**DSPy/Outlines Internals**:
- DSPy Signature fields: `.venv/.../dspy/signatures/signature.py:200-221`
- JSONAdapter structured outputs: `.venv/.../dspy/adapters/json_adapter.py:210-288`
- Outlines constraint generation: `.venv/.../outlines/backends/outlines_core.py:235-252`
- Pydantic model_construct: `.venv/.../pydantic/main.py:307-385`
