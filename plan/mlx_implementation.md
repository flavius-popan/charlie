# Graphiti + MLX via Outlines - Phase 1 Implementation Plan

## Overview

**Goal**: Transition from prototype (`mlx_test.py`) to production architecture integrating Graphiti with local MLX inference through Outlines.

**Success Criteria**: `load_journals.py` runs successfully using MLX+Outlines backend and produces results equivalent to OpenAI/Anthropic implementation.

**Scope**: MLX backend only. llama.cpp support deferred to Phase 2.

---

## ⚠️ IMPORTANT: Manual Confirmation Required

**Each implementation step requires explicit authorization before proceeding.**

After completing each step:
1. Verify the work is correct
2. Run any applicable tests or checks
3. Explicitly authorize the next step before continuing

Do not proceed to the next step without confirmation.

---

## Architecture

```
load_journals.py
    ↓
Graphiti.add_episode()
    ↓
GraphitiLM._generate_response()  [implements LLMClient interface]
    ↓
Outlines.generate.json()          [structured generation with Pydantic]
    ↓
MLX model                         [local inference on Apple Silicon]
```

**Key Interfaces**:
- `LLMClient._generate_response(messages, response_model, max_tokens, model_size) -> dict`
- `EmbedderClient.create(input_data) -> list[float]`

---

## Step 1: Create Module Structure

**Task**: Create `app/llm/` directory and placeholder files

**Files to create**:
- `app/llm/__init__.py` (empty for now)
- `app/llm/prompts.py` (empty, will contain message formatting)
- `app/llm/embedder.py` (empty, will contain MLXEmbedder)
- `app/llm/client.py` (empty, will contain GraphitiLM)

**Verification**:
- Directory structure exists
- All files are present
- Import works: `from app.llm import prompts`

**⏸️ STOP - Await confirmation before Step 2**

---

## Step 2: Implement Message Formatting

**File**: `app/llm/prompts.py`

**Task**: Convert Graphiti `Message` objects (role + content) to prompt strings

**Implementation**:

```python
"""Message formatting utilities for Graphiti LLM bridge."""

from graphiti_core.prompts.models import Message


def format_messages(messages: list[Message], tokenizer) -> str:
    """
    Convert Graphiti messages to chat prompt string.

    Strategy:
    1. Try native chat template if available (preferred)
    2. Fallback to simple System:/User:/Assistant: format

    Args:
        messages: List of Graphiti Message objects
        tokenizer: MLX tokenizer (may have apply_chat_template)

    Returns:
        Formatted prompt string ready for generation
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        # Use native chat template from model
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple text format
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
        prompt += "Assistant:"
        return prompt
```

**Verification**:
- File imports without errors
- Function signature matches expected usage
- Test with sample Message objects if possible

**⏸️ STOP - Await confirmation before Step 3**

---

## Step 3: Implement MLX Embedder

**File**: `app/llm/embedder.py`

**Task**: Implement `EmbedderClient` interface for MLX embeddings

**Based on**: `mlx_test.py:79-138` (MLXEmbedder class)

**Key changes from prototype**:
- Use `asyncio.to_thread()` for async safety
- Separate sync helper method for cleaner code
- Match `EmbedderClient.create()` signature exactly

**Implementation**:

```python
"""MLX embedder for Graphiti."""

import asyncio
from typing import Iterable

import mlx.core as mx
from graphiti_core.embedder import EmbedderClient


class MLXEmbedder(EmbedderClient):
    """
    Local embedder using MLX for Apple Silicon.

    Uses mean pooling over the last hidden state to generate embeddings
    compatible with Graphiti's vector storage.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize MLX embedder.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embedding for text input.

        Args:
            input_data: Text string to embed (or list with single string)

        Returns:
            List of floats representing the embedding vector
        """
        # Handle list input (take first element)
        if isinstance(input_data, list):
            input_data = input_data[0]

        # Offload blocking operation to thread
        return await asyncio.to_thread(self._embed_sync, input_data)

    def _embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding computation.

        Process:
        1. Tokenize input text
        2. Forward pass through model
        3. Mean pool over sequence dimension
        4. Convert to list and return
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens = mx.array([tokens])

        # Forward pass - get last hidden state
        outputs = self.model(tokens)

        # Mean pooling over sequence length
        # Shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
        embedding = mx.mean(outputs, axis=1)

        # Convert to list
        return embedding[0].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for batch of text inputs.

        Args:
            input_data_list: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input_data_list:
            embeddings.append(await self.create(text))
        return embeddings
```

**Verification**:
- File imports without errors
- `MLXEmbedder` is subclass of `EmbedderClient`
- Method signatures match abstract interface
- Test embedding generation with sample text if possible

**⏸️ STOP - Await confirmation before Step 4**

---

## Step 4: Implement GraphitiLM Bridge

**File**: `app/llm/client.py`

**Task**: Implement `LLMClient` interface to bridge Graphiti ↔ Outlines + MLX

**Key requirement**: `_generate_response()` must return `dict[str, Any]`

**Implementation**:

```python
"""GraphitiLM: LLM client bridge for Graphiti + Outlines + MLX."""

import asyncio
import logging
from typing import Any

import outlines
from pydantic import BaseModel

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

from .prompts import format_messages


logger = logging.getLogger(__name__)


class GraphitiLM(LLMClient):
    """
    Bridge: Graphiti ↔ Outlines ↔ MLX

    Design:
    1. Receives: list[Message] + Pydantic schema (response_model)
    2. Formats: messages → prompt string using chat template
    3. Generates: Outlines structured generation with schema constraint
    4. Returns: validated Pydantic object as dict

    This enables all Graphiti operations (entity extraction, deduplication,
    summarization, reflexion, etc.) to work with local MLX inference.
    """

    def __init__(
        self,
        outlines_model,
        tokenizer,
        config: LLMConfig | None = None,
        mode: str = "direct"
    ):
        """
        Initialize GraphitiLM bridge.

        Args:
            outlines_model: Outlines model wrapping MLX (from outlines.models.mlxlm)
            tokenizer: MLX tokenizer (for chat template formatting)
            config: Optional LLM configuration
            mode: "direct" (current) or "dspy" (Phase 2)
        """
        super().__init__(config, cache=False)
        self.outlines_model = outlines_model
        self.tokenizer = tokenizer
        self.mode = mode

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate structured response using Outlines + MLX.

        Core bridge logic:
        1. Format messages → chat prompt string
        2. Create Outlines generator with Pydantic schema
        3. Run generation (offloaded to thread for async safety)
        4. Return Pydantic object as dict

        Args:
            messages: Graphiti message list (system + user prompts)
            response_model: Pydantic schema for structured output
            max_tokens: Maximum tokens to generate (currently unused)
            model_size: Model size hint (currently unused - single model)

        Returns:
            Dict representation of the Pydantic response object
        """
        if self.mode == "dspy":
            # Phase 2: DSPy-based generation
            return await self._dspy_generate(messages, response_model, max_tokens)
        else:
            # Phase 1: Direct Outlines generation
            return await self._direct_generate(messages, response_model)

    async def _direct_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None
    ) -> dict[str, Any]:
        """
        Direct generation via Outlines.

        Process:
        1. Format messages using chat template
        2. Create JSON generator with Pydantic schema
        3. Generate in thread (Outlines is synchronous)
        4. Return result as dict
        """
        # Format prompt
        prompt = format_messages(messages, self.tokenizer)

        # Create structured generator
        if response_model is None:
            raise ValueError("response_model is required for structured generation")

        generator = outlines.generate.json(self.outlines_model, response_model)

        # Generate in thread (Outlines blocks)
        logger.debug(f"Generating with prompt length: {len(prompt)}")
        result = await asyncio.to_thread(generator, prompt)

        # Outlines returns Pydantic instance - convert to dict
        return result.model_dump()

    async def _dspy_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        max_tokens: int
    ) -> dict[str, Any]:
        """
        DSPy-based generation (Phase 2 placeholder).

        Future: Route through DSPy optimizers for improved quality.
        """
        raise NotImplementedError("DSPy mode not yet implemented (Phase 2)")
```

**Verification**:
- File imports without errors
- `GraphitiLM` is subclass of `LLMClient`
- `_generate_response()` signature matches abstract method
- Constructor takes expected arguments
- Test instantiation if possible (without full model load)

**⏸️ STOP - Await confirmation before Step 5**

---

## Step 5: Update Configuration

**File**: `app/settings.py`

**Task**: Add MLX model configuration setting

**Changes**:

Add to end of file:
```python
# LLM Backend Configuration (Phase 1: MLX only)
MLX_MODEL_NAME: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
```

**Verification**:
- Setting is accessible: `from app import settings; print(settings.MLX_MODEL_NAME)`
- No syntax errors in settings file

**⏸️ STOP - Await confirmation before Step 6**

---

## Step 6: Update load_journals.py

**File**: `load_journals.py`

**Task**: Replace default Graphiti initialization with MLX backend

**Location**: Around line 140-142 in `load_journals()` function

**Original code**:
```python
kuzu_driver = KuzuDriver(db=db_path)
graphiti = Graphiti(graph_driver=kuzu_driver)
```

**New code**:
```python
# Initialize MLX model and tokenizer
logger.info("Loading MLX model and tokenizer...")
import mlx_lm
mlx_model, mlx_tokenizer = mlx_lm.load(settings.MLX_MODEL_NAME)
logger.info("MLX model loaded successfully")

# Create Outlines model wrapper
logger.info("Initializing Outlines structured generation...")
import outlines
outlines_model = outlines.models.mlxlm(mlx_model, mlx_tokenizer)
logger.info("Outlines model ready")

# Initialize bridge components
from app.llm.client import GraphitiLM
from app.llm.embedder import MLXEmbedder

llm_client = GraphitiLM(outlines_model, mlx_tokenizer)
embedder = MLXEmbedder(mlx_model, mlx_tokenizer)
logger.info("LLM client and embedder initialized")

# Initialize Graphiti with local MLX backend
kuzu_driver = KuzuDriver(db=db_path)
graphiti = Graphiti(
    graph_driver=kuzu_driver,
    llm_client=llm_client,
    embedder=embedder
)
```

**Verification**:
- File has no syntax errors
- Imports are at correct scope
- Settings imported: `from app import settings` exists at top of file

**⏸️ STOP - Await confirmation before Step 7**

---

## Step 7: Manual End-to-End Testing

**Task**: Verify complete integration with real journal data

**Prerequisites**:
- All previous steps completed
- MLX model downloaded locally (via `models.py` or auto-download)
- Test journal subset available

**Test Procedure**:

1. **Create test subset** (if needed):
   ```bash
   # Extract first 2 entries from Journal.json for quick testing
   python -c "import json; d=json.load(open('raw_data/Journal.json')); d['entries']=d['entries'][:2]; json.dump(d, open('raw_data/test_subset.json','w'), indent=2)"
   ```

2. **Run journal loader**:
   ```bash
   python load_journals.py load \
     --journal-file raw_data/test_subset.json \
     --skip-verification \
     --db-path brain/test_mlx.kuzu
   ```

3. **Verify output**:
   - Model loads without errors
   - Journal entries process successfully
   - Entities and relationships extracted
   - No async-related errors or crashes
   - Graphiti operations complete (extraction, deduplication, etc.)

4. **Inspect database** (optional):
   ```python
   # Quick inspection script
   from graphiti_core import Graphiti
   from graphiti_core.driver.kuzu_driver import KuzuDriver

   driver = KuzuDriver("brain/test_mlx.kuzu")
   # Query entities, edges, episodes...
   ```

5. **Compare with baseline** (if OpenAI/Anthropic results available):
   - Entity names and types similar?
   - Relationship facts semantically equivalent?
   - No major quality degradation?

**Success Criteria**:
- ✅ No crashes or exceptions
- ✅ Structured outputs generated correctly
- ✅ Entities and relationships saved to database
- ✅ Async operations work smoothly
- ✅ Quality comparable to cloud LLM baseline

**⏸️ STOP - Report results and await next steps**

---

## Post-Verification Actions

After successful Step 7 verification:

### Option A: Archive Prototype
```bash
mkdir -p archive
git mv mlx_test.py archive/
git commit -m "Archive mlx_test.py prototype after successful refactor"
```

### Option B: Keep for Reference
Leave `mlx_test.py` in place temporarily for comparison testing

### Next Phase Preparation
Document findings for Phase 2:
- MLX model performance (speed, quality)
- Memory usage
- Any quirks or issues discovered
- Readiness for llama.cpp backend addition

---

## Rollback Plan

If testing fails at Step 7:

1. **Identify failure point**: Parse error? Async issue? Generation quality?
2. **Isolate component**: Test embedder, client, prompts separately
3. **Compare with prototype**: Does `mlx_test.py` still work?
4. **Debug and iterate**: Fix issues in specific module
5. **Re-verify**: Repeat Step 7 after fixes

**Fallback**: Revert `load_journals.py` changes and continue using cloud LLM while debugging

---

## Dependencies

**Already satisfied**:
- ✅ `mlx-lm>=0.28.2` (in pyproject.toml)
- ✅ `outlines>=1.2.6` (in pyproject.toml)
- ✅ `outlines-core` override for MLX fix (in tool.uv section)
- ✅ `graphiti-core>=0.21.0` (in pyproject.toml)

**No new dependencies required for Phase 1**

---

## Estimated Effort

- **New code**: ~150 lines across 3 modules
- **Modified code**: ~40 lines (load_journals.py + settings.py)
- **Testing**: 1-2 hours (model download + verification)
- **Total**: ~3-4 hours end-to-end

---

## Phase 2 Preview (Post-MLX Verification)

Once MLX implementation is confirmed working:

1. **llama.cpp Backend Support**
   - Create `app/llm/backend.py` for backend selection
   - Add `LlamaCppEmbedder` class
   - Update `GraphitiLM` to handle backend differences

2. **Unified Model Management**
   - Extend `models.py` for GGUF downloads
   - Create model registry/config
   - Streamline model selection

3. **DSPy Integration** (if quality issues)
   - Implement `_dspy_generate()` in GraphitiLM
   - Add DSPy optimizers
   - A/B test quality improvements

4. **Testing & Documentation**
   - Unit tests for components
   - Integration tests
   - User documentation
   - Performance benchmarks
