# DSPy + Outlines + MLX Hybrid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a hybrid LM client that combines DSPy's signature-based programming with Outlines' constrained generation on MLX, enabling structured output extraction (knowledge graphs) with guaranteed JSON schema compliance.

**Architecture:** Build a custom `dspy.LM` subclass that intercepts generation calls and routes them through Outlines' constrained decoding engine. This mirrors the working Outlines+MLX bridge pattern from `app/llm/client.py` but for DSPy. The hybrid approach lets DSPy handle prompt structure/optimization while Outlines enforces Pydantic schemas at generation time.

**Tech Stack:** DSPy 3.0.3, Outlines 1.2.7, MLX-LM 0.28.2, Pydantic

---

## Implementation Status

**✓ COMPLETED (Tasks 1-3):**
- Task 1: DSPy LM interface researched → `research/dspy-lm-interface.md`
- Task 2: PassthroughLM created → `dspy_outlines/base_lm.py` (proves interception works)
- Task 3: MLX loading implemented → `dspy_outlines/mlx_loader.py` (both tests passing)

**→ NEXT (Tasks 4-6):** Schema extraction, hybrid LM implementation, integration

**PENDING (Tasks 7-10):** Async support, Gradio UI, docs, integration tests

---

## Background & Context

### What We Discovered

1. **DSPy's structured output is prompt-based only** - uses JSON schema in prompts but doesn't guarantee valid output
2. **Outlines provides true constrained decoding** - guarantees valid JSON/regex/grammar at token generation level
3. **MLX-LM is already running** via LM Studio on `http://127.0.0.1:8000/v1` serving `qwen/qwen3-4b-2507`
4. **Existing working pattern** in `load_journals.py` + `app/llm/client.py` shows how to bridge Outlines+MLX with a framework

### Key Reference Files

**Working Outlines+MLX Integration (reference patterns only):**
- `load_journals.py:99-104` - Shows how to create Outlines model wrapper from MLX
- `app/llm/client.py:24-164` - Bridge pattern: framework ↔ Outlines ↔ MLX (use as reference, we're ditching Graphiti)
- `app/llm/prompts.py:6-38` - Message formatting using tokenizer chat templates
- `app/settings.py:16-18` - Model paths and configuration

**Current DSPy PoC:**
- `dspy-poc.py` - Working DSPy knowledge graph extraction (prompt-based only)
- Uses `dspy.Predict` + Pydantic models (Node, Edge, KnowledgeGraph)
- Connects to LM Studio OpenAI-compatible endpoint

**Langtrace Blog Post Insights (from screenshots):**
The Langtrace approach (now outdated, but useful pattern) showed:
1. **Custom `OutlinesLM(dspy.LM)` class** that wraps DSPy's LM interface
2. **`__call__` method intercepts** DSPy generation calls
3. **Routes based on `generate_fn` parameter**: `"json"`, `"choice"`, `"regex"`, or `"text"`
4. **Extracts `schema_object`** from kwargs and passes to Outlines' `generate.json()`, `generate.choice()`, etc.
5. **Uses Pydantic `model_config = ConfigDict(extra='forbid')`** to prevent extra fields
6. **Example usage pattern**:
   ```python
   outlines_lm = OutlinesLM(
       model="gpt-4o-mini",
       generate_fn="json",
       schema_object=ClassificationType  # Pydantic model
   )
   ```

**Key takeaway**: We intercept at the LM level, extract schema, route to Outlines constrained generation, return validated result.

### The Problem to Solve

Current `dspy-poc.py` works but:
- No guarantee of valid JSON output (DSPy uses prompts, not constrained generation)
- Using LM Studio as separate service (want to consolidate with MLX direct)
- Missing Outlines' powerful features (regex, grammar, guaranteed schemas)

### The Solution

Create `OutlinesDSPyLM` that:
1. Implements `dspy.LM` interface (or subclasses it)
2. Loads MLX model directly via `outlines.models.mlxlm()`
3. Intercepts DSPy's generation calls
4. Extracts Pydantic schema from DSPy signature output fields
5. Routes to Outlines' `model(prompt, output_type=PydanticModel, max_tokens=N)`
6. Returns validated results to DSPy

---

## ✓ Task 1: Research DSPy LM Interface (COMPLETED)

**Result:** Research documented in `research/dspy-lm-interface.md`

**Key Findings:**
- Inherit from `dspy.BaseLM` (simpler than `dspy.LM`)
- Override `forward(prompt, messages, **kwargs)` method
- Return OpenAI-format response: `{"choices": [...], "usage": {...}, "model": "..."}`
- Signature passed in kwargs by adapters - easy to extract Pydantic schemas
- Adapters call `lm(...)` → `__call__()` → `forward()` - intercept at forward level

---

## ✓ Task 2: Create Minimal Custom DSPy LM (Passthrough) (COMPLETED)

**Result:** Created `PassthroughLM` class that proves DSPy call interception works

**Files Created:**
- `dspy_outlines/base_lm.py` - PassthroughLM that inherits from `dspy.LM` and logs interceptions
- `dspy_outlines/__init__.py` - Package exports
- `tests/test_base_lm.py` - Test that PassthroughLM handles basic DSPy calls

**Updated:**
- `dspy-poc.py` - Now uses `PassthroughLM` instead of `dspy.LM`
- `pyproject.toml` - Added `dspy_outlines*` to package includes

**Test Status:** ✓ PASSING

---

## ✓ Task 3: Load MLX Model via Outlines (COMPLETED)

**Result:** Created MLX model loading utilities with Outlines wrapper

**Files Created:**
- `dspy_outlines/mlx_loader.py` - `load_mlx_model()` and `create_outlines_model()` functions
- `tests/test_mlx_loader.py` - Tests for both MLX loading and Outlines structured generation

**Key Implementation Detail:**
- Correct API: `outlines.from_mlxlm(mlx_model, mlx_tokenizer)` (not `outlines.models.mlxlm()`)
- Default model path: `.models/mlx-community--Qwen3-4B-Instruct-2507-8bit`
- Outlines wrapper supports constrained generation with Pydantic `output_type` parameter

**Test Status:** ✓ 2 PASSING (load model + structured generation)

---

## Task 4: Extract Pydantic Schema from DSPy Signature

**Goal:** Build utility to extract Pydantic models from DSPy signature output fields.

**Files:**
- Create: `dspy_outlines/schema_extractor.py`
- Create: `tests/test_schema_extractor.py`

**Step 1: Write failing test**

Create: `tests/test_schema_extractor.py`

```python
import dspy
from pydantic import BaseModel, Field
from typing import List

from dspy_outlines.schema_extractor import extract_output_schema

def test_extract_simple_schema():
    """Test extracting Pydantic model from simple signature."""

    class Answer(BaseModel):
        text: str
        confidence: float

    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: Answer = dspy.OutputField()

    schema = extract_output_schema(QASignature)

    assert schema == Answer

def test_extract_complex_schema():
    """Test extracting complex nested Pydantic model."""

    class Node(BaseModel):
        id: str
        label: str

    class Edge(BaseModel):
        source: str
        target: str

    class Graph(BaseModel):
        nodes: List[Node]
        edges: List[Edge]

    class GraphSignature(dspy.Signature):
        text: str = dspy.InputField()
        graph: Graph = dspy.OutputField()

    schema = extract_output_schema(GraphSignature)

    assert schema == Graph
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema_extractor.py -v
```

Expected: FAIL (module not found)

**Step 3: Implement schema extractor**

Create: `dspy_outlines/schema_extractor.py`

```python
"""Extract Pydantic schemas from DSPy signatures."""

import logging
from typing import Type
from pydantic import BaseModel
import dspy

logger = logging.getLogger(__name__)

def extract_output_schema(signature: Type[dspy.Signature]) -> Type[BaseModel] | None:
    """
    Extract Pydantic model from DSPy signature's output field.

    Args:
        signature: DSPy Signature class

    Returns:
        Pydantic BaseModel class if found, None otherwise
    """
    # Get signature's output fields
    output_fields = signature.output_fields

    if not output_fields:
        logger.warning(f"No output fields in signature {signature.__name__}")
        return None

    # For now, assume single output field (most common case)
    # Future: handle multiple output fields
    if len(output_fields) > 1:
        logger.warning(f"Multiple output fields in {signature.__name__}, using first")

    output_field = output_fields[0]
    field_type = output_field.annotation

    # Check if it's a Pydantic model
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        logger.info(f"Extracted schema: {field_type.__name__}")
        return field_type

    logger.warning(f"Output field is not a Pydantic model: {field_type}")
    return None
```

**Step 4: Run test**

```bash
pytest tests/test_schema_extractor.py -v
```

Expected: PASS (or debug/fix as needed)

**Step 5: Commit**

```bash
git add dspy_outlines/schema_extractor.py tests/test_schema_extractor.py
git commit -m "feat: add Pydantic schema extraction from DSPy signatures"
```

---

## Task 5: Implement OutlinesDSPyLM (Hybrid LM)

**Goal:** Create the full hybrid LM that routes DSPy calls through Outlines constrained generation.

**Files:**
- Create: `dspy_outlines/hybrid_lm.py`
- Create: `tests/test_hybrid_lm.py`

**Step 1: Write failing test**

Create: `tests/test_hybrid_lm.py`

```python
import dspy
from pydantic import BaseModel, Field
from typing import List

from dspy_outlines.hybrid_lm import OutlinesDSPyLM

def test_hybrid_lm_knowledge_graph_extraction():
    """Test full hybrid LM with knowledge graph extraction."""

    # Define Pydantic models (same as dspy-poc.py)
    class Node(BaseModel):
        id: str = Field(description="Unique identifier for the node")
        label: str = Field(description="Name of the entity")
        properties: dict = Field(default_factory=dict)

    class Edge(BaseModel):
        source: str = Field(description="Source node ID")
        target: str = Field(description="Target node ID")
        label: str = Field(description="Type of relationship")
        properties: dict = Field(default_factory=dict)

    class KnowledgeGraph(BaseModel):
        nodes: List[Node]
        edges: List[Edge]

    # Define DSPy signature
    class ExtractKnowledgeGraph(dspy.Signature):
        """Extract knowledge graph of people and relationships."""
        text: str = dspy.InputField()
        graph: KnowledgeGraph = dspy.OutputField()

    # Initialize hybrid LM
    lm = OutlinesDSPyLM()
    dspy.configure(lm=lm)

    # Create predictor
    extractor = dspy.Predict(ExtractKnowledgeGraph)

    # Test extraction
    text = "Alice met Bob at the coffee shop. Charlie joined them."
    result = extractor(text=text)

    # Verify result
    assert hasattr(result, 'graph')
    assert isinstance(result.graph, KnowledgeGraph)
    assert len(result.graph.nodes) >= 3  # At least Alice, Bob, Charlie
    assert all(isinstance(n, Node) for n in result.graph.nodes)
    assert all(isinstance(e, Edge) for e in result.graph.edges)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_hybrid_lm.py::test_hybrid_lm_knowledge_graph_extraction -v
```

Expected: FAIL (module not found)

**Step 3: Implement OutlinesDSPyLM**

Create: `dspy_outlines/hybrid_lm.py`

```python
"""Hybrid DSPy LM using Outlines for constrained generation."""

import asyncio
import logging
from typing import Any

import dspy
from pydantic import BaseModel

from .mlx_loader import create_outlines_model
from .schema_extractor import extract_output_schema
from .prompt_formatter import format_dspy_prompt

logger = logging.getLogger(__name__)

# MLX thread safety lock (same pattern as GraphitiLM)
MLX_LOCK = asyncio.Lock()

class OutlinesDSPyLM(dspy.LM):
    """
    Hybrid LM: DSPy ↔ Outlines ↔ MLX

    Architecture:
    1. Receives DSPy signature + inputs
    2. Extracts Pydantic schema from output field
    3. Formats prompt using DSPy's formatting
    4. Generates via Outlines with schema constraint
    5. Returns validated Pydantic object to DSPy

    This gives us:
    - DSPy's signature-based programming + optimization
    - Outlines' guaranteed constrained generation
    - MLX's efficient local inference
    """

    def __init__(self, model_path: str = None):
        """
        Initialize hybrid LM.

        Args:
            model_path: Path to MLX model (uses default if None)
        """
        # Don't call super().__init__ - we're replacing the backend entirely
        self.outlines_model, self.tokenizer = create_outlines_model(model_path)
        self.history = []
        logger.info("OutlinesDSPyLM initialized with Outlines+MLX backend")

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Main generation interface called by DSPy.

        Args:
            prompt: String prompt (if using prompt-based)
            messages: List of message dicts (if using chat format)
            **kwargs: Additional generation params (max_tokens, temperature, etc.)

        Returns:
            List of completion strings (DSPy expects list)
        """
        # Extract generation params
        max_tokens = kwargs.get('max_tokens', 512)

        # Get the signature from kwargs if available (DSPy passes it)
        signature = kwargs.get('signature', None)

        # Extract Pydantic schema from signature
        schema = None
        if signature:
            schema = extract_output_schema(signature)

        # Format the prompt
        if messages:
            formatted_prompt = self._format_messages(messages)
        else:
            formatted_prompt = prompt

        logger.info(f"Generating with schema: {schema.__name__ if schema else 'None'}")

        # Generate using Outlines
        if schema:
            # Constrained generation with Pydantic schema
            result_json = self.outlines_model(
                formatted_prompt,
                output_type=schema,
                max_tokens=max_tokens
            )
            # Parse and re-serialize to ensure valid JSON
            parsed = schema.model_validate_json(result_json)
            completion = parsed.model_dump_json()
        else:
            # Fallback: unconstrained text generation
            completion = self.outlines_model(
                formatted_prompt,
                max_tokens=max_tokens
            )

        # Store in history
        self.history.append({
            "prompt": formatted_prompt[:200],
            "completion": completion[:200]
        })

        # DSPy expects list of completions
        return [completion]

    def _format_messages(self, messages: list[dict]) -> str:
        """
        Format chat messages using tokenizer's chat template.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts

        Returns:
            Formatted prompt string
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
            prompt += "Assistant:"
            return prompt
```

**Step 4: Create prompt_formatter helper (if needed)**

If DSPy doesn't expose its prompt formatting, create:

`dspy_outlines/prompt_formatter.py`:

```python
"""Utilities for formatting DSPy prompts."""

def format_dspy_prompt(signature, inputs: dict) -> str:
    """
    Format DSPy signature + inputs into prompt string.

    For now, simple template. Future: use DSPy's internal formatter.
    """
    prompt = f"{signature.__doc__}\n\n" if signature.__doc__ else ""

    for field_name, value in inputs.items():
        prompt += f"{field_name}: {value}\n"

    prompt += "\nOutput:"
    return prompt
```

**Step 5: Run test**

```bash
pytest tests/test_hybrid_lm.py::test_hybrid_lm_knowledge_graph_extraction -v
```

Expected: PASS (or debug signature/schema extraction issues)

**Step 6: Fix any issues**

Common issues:
- Schema extraction might need to handle DSPy's field annotation format differently
- Prompt formatting might need DSPy's actual internal format
- May need to handle DSPy's adapter layer

Debug and iterate until test passes.

**Step 7: Commit**

```bash
git add dspy_outlines/hybrid_lm.py dspy_outlines/prompt_formatter.py tests/test_hybrid_lm.py
git commit -m "feat: implement OutlinesDSPyLM hybrid LM with constrained generation"
```

---

## Task 6: Update dspy-poc.py to Use Hybrid LM

**Goal:** Replace LM Studio API calls with direct MLX via OutlinesDSPyLM.

**Files:**
- Modify: `dspy-poc.py`

**Step 1: Backup current version**

```bash
cp dspy-poc.py dspy-poc-lmstudio.py
git add dspy-poc-lmstudio.py
git commit -m "chore: backup LM Studio version of PoC"
```

**Step 2: Update imports and LM initialization**

Modify: `dspy-poc.py:1-12`

```python
import dspy
from pydantic import BaseModel, Field
from typing import List
import json

from dspy_outlines import OutlinesDSPyLM

# Configure Outlines+MLX hybrid LM (no LM Studio needed!)
lm = OutlinesDSPyLM()  # Uses default model path from mlx_loader
dspy.configure(lm=lm)

# Rest of file stays the same...
```

**Step 3: Test manually**

```bash
python dspy-poc.py
```

Paste test text:
```
Alice met Bob at the coffee shop. Dr. Charlie Smith joined them later to discuss the research project with Professor Diana Lee.
```

Expected: Same JSON output but generated via MLX directly, not LM Studio.

**Step 4: Verify LM Studio is NOT being called**

```bash
# Stop LM Studio or kill the server
# Then run the PoC again - it should still work
python dspy-poc.py
```

Expected: Works without LM Studio running.

**Step 5: Compare output quality**

Run both versions side-by-side:
```bash
# Terminal 1: LM Studio version
python dspy-poc-lmstudio.py

# Terminal 2: MLX direct version
python dspy-poc.py
```

Compare:
- Generation speed
- Output quality
- JSON validity (new version should NEVER produce invalid JSON)

**Step 6: Commit**

```bash
git add dspy-poc.py
git commit -m "feat: switch dspy-poc to Outlines+MLX hybrid (no LM Studio)"
```

---

## Task 7: Add Async Support (Optional Performance Enhancement)

**Goal:** Add async generation to match GraphitiLM pattern for better performance.

**Files:**
- Modify: `dspy_outlines/hybrid_lm.py`

**Step 1: Add async wrapper**

Modify: `dspy_outlines/hybrid_lm.py`

Add method:
```python
async def _generate_async(self, prompt: str, schema: type[BaseModel] | None, max_tokens: int) -> str:
    """
    Async generation using Outlines (runs in thread pool).

    Pattern from GraphitiLM: offload synchronous Outlines to thread.
    """
    async with MLX_LOCK:
        if schema:
            result_json = await asyncio.to_thread(
                self.outlines_model,
                prompt,
                output_type=schema,
                max_tokens=max_tokens
            )
            parsed = schema.model_validate_json(result_json)
            return parsed.model_dump_json()
        else:
            return await asyncio.to_thread(
                self.outlines_model,
                prompt,
                max_tokens=max_tokens
            )
```

**Step 2: Update __call__ to use async if available**

Modify: `OutlinesDSPyLM.__call__`

```python
def __call__(self, prompt=None, messages=None, **kwargs):
    # ... existing code to extract schema and format prompt ...

    # Check if we're in async context
    try:
        loop = asyncio.get_running_loop()
        # We're in async context - but DSPy __call__ is sync
        # Use asyncio.run_coroutine_threadsafe? Or just use sync?
        # For now, keep sync (DSPy interface is sync)
        completion = self._generate_sync(formatted_prompt, schema, max_tokens)
    except RuntimeError:
        # No event loop - use sync
        completion = self._generate_sync(formatted_prompt, schema, max_tokens)

    return [completion]

def _generate_sync(self, prompt: str, schema: type[BaseModel] | None, max_tokens: int) -> str:
    """Synchronous generation (existing logic)."""
    if schema:
        result_json = self.outlines_model(
            prompt,
            output_type=schema,
            max_tokens=max_tokens
        )
        parsed = schema.model_validate_json(result_json)
        return parsed.model_dump_json()
    else:
        return self.outlines_model(prompt, max_tokens=max_tokens)
```

**Step 3: Test performance**

Add benchmark test:
```python
import time

def test_hybrid_lm_performance():
    """Benchmark generation speed."""
    lm = OutlinesDSPyLM()
    # ... setup signature and predictor ...

    start = time.time()
    for i in range(5):
        result = extractor(text="Alice met Bob.")
    elapsed = time.time() - start

    print(f"5 generations in {elapsed:.2f}s ({elapsed/5:.2f}s each)")
```

**Step 4: Commit**

```bash
git add dspy_outlines/hybrid_lm.py tests/test_hybrid_lm.py
git commit -m "feat: add async support to hybrid LM for better performance"
```

---

## Task 8: Add Gradio UI for Interactive Knowledge Graph Extraction

**Goal:** Build simple Gradio UI to input text and visualize extracted knowledge graphs.

**Files:**
- Create: `gradio_app.py`
- Update: `requirements.txt` or `pyproject.toml`

**Step 1: Install Gradio**

```bash
uv add gradio graphviz
```

**Step 2: Create minimal Gradio app**

Create: `gradio_app.py`

```python
"""Gradio UI for knowledge graph extraction."""

import gradio as gr
import dspy
from pydantic import BaseModel, Field
from typing import List
import json

from dspy_outlines import OutlinesDSPyLM

# Initialize LM
lm = OutlinesDSPyLM()
dspy.configure(lm=lm)

# Pydantic models
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Name of the entity")
    properties: dict = Field(default_factory=dict)

class Edge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    label: str = Field(description="Type of relationship")
    properties: dict = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# DSPy signature
class ExtractKnowledgeGraph(dspy.Signature):
    """Extract knowledge graph of people and relationships."""
    text: str = dspy.InputField()
    graph: KnowledgeGraph = dspy.OutputField()

# Create predictor
extractor = dspy.Predict(ExtractKnowledgeGraph)

def extract_and_display(text: str, max_tokens: int = 512):
    """
    Extract knowledge graph from text and return JSON + visualization.

    Args:
        text: Input text to analyze
        max_tokens: Maximum tokens for generation

    Returns:
        tuple: (json_output, graph_image)
    """
    if not text.strip():
        return "Please enter some text.", None

    try:
        # Extract graph
        result = extractor(text=text)
        graph = result.graph

        # Format JSON
        json_output = json.dumps(graph.model_dump(), indent=2)

        # Create graph visualization (TODO: implement)
        graph_image = None  # Placeholder for now

        return json_output, graph_image

    except Exception as e:
        return f"Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Knowledge Graph Extractor") as demo:
    gr.Markdown("# Knowledge Graph Extraction")
    gr.Markdown("Extract people and relationships from text using DSPy + Outlines + MLX")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze (e.g., 'Alice met Bob at the coffee shop.')",
                lines=10
            )
            max_tokens_slider = gr.Slider(
                minimum=128,
                maximum=2048,
                value=512,
                step=128,
                label="Max Tokens"
            )
            extract_btn = gr.Button("Extract Knowledge Graph", variant="primary")

        with gr.Column():
            json_output = gr.Code(label="Extracted Graph (JSON)", language="json")
            # graph_viz = gr.Image(label="Graph Visualization")  # TODO

    extract_btn.click(
        fn=extract_and_display,
        inputs=[text_input, max_tokens_slider],
        outputs=[json_output]  # Add graph_viz when implemented
    )

    # Example
    gr.Examples(
        examples=[
            ["Alice met Bob at the coffee shop. Dr. Charlie Smith joined them later to discuss the research project with Professor Diana Lee."],
            ["John works at Microsoft. He reports to Sarah, the VP of Engineering. Sarah previously worked with David at Google."],
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch()
```

**Step 3: Test Gradio app**

```bash
python gradio_app.py
```

Open browser to http://localhost:7860, test extraction.

**Step 4: Add graph visualization (optional)**

If time permits, add Graphviz rendering:

```python
import graphviz

def create_graph_image(graph: KnowledgeGraph) -> str:
    """Create Graphviz visualization of knowledge graph."""
    dot = graphviz.Digraph(comment='Knowledge Graph')

    # Add nodes
    for node in graph.nodes:
        dot.node(node.id, node.label)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge.source, edge.target, label=edge.label)

    # Render to PNG
    output_path = "/tmp/knowledge_graph"
    dot.render(output_path, format='png', cleanup=True)

    return f"{output_path}.png"
```

Update `extract_and_display` to generate image.

**Step 5: Commit**

```bash
git add gradio_app.py
git commit -m "feat: add Gradio UI for interactive knowledge graph extraction"
```

---

## Task 9: Documentation and Examples

**Goal:** Document the hybrid approach and provide usage examples.

**Files:**
- Create: `docs/dspy-outlines-hybrid.md`
- Update: `README.md`

**Step 1: Write architecture documentation**

Create: `docs/dspy-outlines-hybrid.md`

```markdown
# DSPy + Outlines + MLX Hybrid Architecture

## Overview

This project combines three powerful libraries:
- **DSPy**: Signature-based LLM programming with optimization
- **Outlines**: Constrained generation with guaranteed schema compliance
- **MLX**: Efficient local inference on Apple Silicon

## Architecture

```
DSPy Signature → OutlinesDSPyLM → Outlines → MLX → GPU
     ↓                ↓              ↓
  Program         Schema       Constrained
  Structure      Extraction    Generation
```

### Components

**OutlinesDSPyLM** (`dspy_outlines/hybrid_lm.py`)
- Implements `dspy.LM` interface
- Extracts Pydantic schemas from DSPy signatures
- Routes to Outlines for constrained generation
- Returns validated results to DSPy

**Schema Extractor** (`dspy_outlines/schema_extractor.py`)
- Parses DSPy signature output fields
- Identifies Pydantic BaseModel types
- Passes schema to Outlines

**MLX Loader** (`dspy_outlines/mlx_loader.py`)
- Loads local MLX models
- Creates Outlines wrapper
- Manages model lifecycle

## Benefits

1. **Guaranteed Valid Output**: Outlines ensures JSON schema compliance
2. **Local Inference**: No API calls, runs entirely on-device
3. **DSPy Optimization**: Can use DSPy's optimizers to improve prompts
4. **Type Safety**: Pydantic validation end-to-end

## Usage

```python
import dspy
from pydantic import BaseModel
from dspy_outlines import OutlinesDSPyLM

# Initialize hybrid LM
lm = OutlinesDSPyLM()
dspy.configure(lm=lm)

# Define Pydantic output model
class Answer(BaseModel):
    text: str
    confidence: float

# Define DSPy signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()

# Use as normal DSPy
qa = dspy.Predict(QA)
result = qa(question="What is 2+2?")

print(result.answer)  # Guaranteed valid Answer object
```

## See Also

- `dspy-poc.py` - Knowledge graph extraction example
- `gradio_app.py` - Interactive UI
- `load_journals.py` - Graphiti integration (similar pattern)
```

**Step 2: Update main README**

Add section to `README.md`:

```markdown
## DSPy + Outlines + MLX Integration

This project uses a hybrid approach combining:
- **DSPy** for structured LLM programming
- **Outlines** for constrained generation
- **MLX** for local Apple Silicon inference

See `docs/dspy-outlines-hybrid.md` for architecture details.

### Quick Start

```bash
# Run knowledge graph extraction PoC
python dspy-poc.py

# Or launch interactive Gradio UI
python gradio_app.py
```
```

**Step 3: Commit**

```bash
git add docs/dspy-outlines-hybrid.md README.md
git commit -m "docs: add hybrid architecture documentation"
```

---

## Task 10: Testing and Validation

**Goal:** Ensure the hybrid system works correctly and produces valid output.

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create: `tests/test_integration.py`

```python
"""Integration tests for DSPy + Outlines + MLX hybrid."""

import dspy
from pydantic import BaseModel, Field
from typing import List
import json

from dspy_outlines import OutlinesDSPyLM

def test_json_validity_always():
    """Test that output is ALWAYS valid JSON (key benefit of Outlines)."""

    class Result(BaseModel):
        value: int

    class Signature(dspy.Signature):
        prompt: str = dspy.InputField()
        result: Result = dspy.OutputField()

    lm = OutlinesDSPyLM()
    dspy.configure(lm=lm)

    predictor = dspy.Predict(Signature)

    # Run 10 times - should NEVER fail JSON parsing
    for i in range(10):
        result = predictor(prompt="Count to 5")
        assert isinstance(result.result, Result)
        # Should be able to serialize
        json.dumps(result.result.model_dump())

def test_complex_nested_schema():
    """Test with deeply nested Pydantic models."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        age: int
        address: Address

    class Team(BaseModel):
        name: str
        members: List[Person]

    class Signature(dspy.Signature):
        description: str = dspy.InputField()
        team: Team = dspy.OutputField()

    lm = OutlinesDSPyLM()
    dspy.configure(lm=lm)

    predictor = dspy.Predict(Signature)
    result = predictor(description="A team of 2 engineers")

    assert isinstance(result.team, Team)
    assert len(result.team.members) >= 1
    assert all(isinstance(m, Person) for m in result.team.members)
    assert all(isinstance(m.address, Address) for m in result.team.members)

def test_knowledge_graph_extraction_full():
    """Full end-to-end test of knowledge graph extraction."""
    # ... (same as earlier test but in integration suite)
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS

**Step 3: Run with coverage**

```bash
pytest tests/ --cov=dspy_outlines --cov-report=html
open htmlcov/index.html
```

Target: >80% coverage

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for hybrid system"
```

---

## Completion Checklist

- [ ] Custom DSPy LM class implemented
- [ ] MLX model loading via Outlines working
- [ ] Schema extraction from DSPy signatures
- [ ] Constrained generation with Pydantic models
- [ ] `dspy-poc.py` updated to use hybrid LM
- [ ] Gradio UI for interactive extraction
- [ ] Documentation written
- [ ] Tests passing
- [ ] No dependency on LM Studio

## Success Criteria

1. **Run `python dspy-poc.py`** - extracts knowledge graph using MLX directly
2. **Output is ALWAYS valid JSON** - Outlines guarantees schema compliance
3. **No LM Studio needed** - runs entirely local via MLX
4. **Gradio UI works** - interactive extraction with visualization
5. **Tests pass** - integration tests validate hybrid approach

## Future Enhancements

- Use DSPy optimizers (BootstrapFewShot, MIPRO) to improve extraction quality
- Add graph visualization in Gradio using Graphviz
- Support multiple output fields in signatures
- Batch processing for multiple texts
- Export graphs to various formats (GraphML, Cypher, etc.)
