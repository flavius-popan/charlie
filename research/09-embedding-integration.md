# Embedding Integration with Qwen3

## Overview

Embeddings are **foundational** to Graphiti's search and deduplication systems. Every entity name and fact text is embedded for semantic search. This document covers the embedding subsystem and integration with Qwen3-Embedding-4B-4bit-DWQ.

**Key Locations:**
- `graphiti_core/embedder/client.py` - Abstract interface
- `graphiti_core/embedder/openai_embedder.py` - OpenAI implementation
- `graphiti_core/embedder/voyageai_embedder.py` - VoyageAI implementation
- `graphiti_core/embedder/bge_embedder.py` - BGE local implementation

## Qwen3 Models for Graphiti

### 1. Qwen3-Embedding-4B-4bit-DWQ (for embeddings)
**HuggingFace**: https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ

**Key Characteristics:**
- 4-bit quantized for MLX (Apple Silicon optimized)
- 4B parameters (high quality)
- Output dimension: 768 or 1024 (check model config)
- Optimized for semantic similarity tasks
- Local inference via MLX

### 2. Qwen3-Reranker-0.6B-seq-cls (for reranking)
**HuggingFace**: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls

**Key Characteristics:**
- Sequence classification model
- 0.6B parameters (lightweight, fast)
- Direct relevance scoring (not boolean classification)
- Suitable for cross-encoder reranking

## EmbedderClient Interface

### Abstract Base Class

**Location**: `graphiti_core/embedder/client.py`

```python
from abc import ABC, abstractmethod

EMBEDDING_DIM = 1024  # Default dimension

class EmbedderClient(ABC):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def create(self, input_data: list[str]) -> list[list[float]]:
        """
        Create embeddings for input strings.

        Args:
            input_data: List of strings to embed

        Returns:
            List of embedding vectors (each is list[float])
        """
        pass

    @abstractmethod
    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        """
        Create embeddings in batch mode.

        Args:
            input_data: List of strings to embed

        Returns:
            List of embedding vectors
        """
        pass
```

**Key Points:**
- Both `create()` and `create_batch()` return same format
- `create_batch()` is for optimization (batching multiple inputs)
- Embeddings used for cosine similarity search
- Critical for node/edge deduplication

## Where Embeddings Are Used

### 1. Entity Node Embeddings

**Field**: `EntityNode.name_embedding: list[float]`

**Purpose**: Semantic search for entity deduplication

**Generation**:
```python
# From graphiti_core/nodes.py
async def create_entity_node_embeddings(
    embedder: EmbedderClient,
    nodes: list[EntityNode]
):
    texts = [node.name.replace('\n', ' ') for node in nodes]
    name_embeddings = await embedder.create_batch(texts)
    for node, name_embedding in zip(nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding
```

**Usage**: When extracting entities, search existing graph for similar entity names using cosine similarity.

### 2. Entity Edge Embeddings

**Field**: `EntityEdge.fact_embedding: list[float]`

**Purpose**: Semantic search for fact deduplication and retrieval

**Generation**:
```python
# From graphiti_core/edges.py
async def create_entity_edge_embeddings(
    embedder: EmbedderClient,
    edges: list[EntityEdge]
):
    filtered_edges = [edge for edge in edges if edge.fact]
    fact_embeddings = await embedder.create_batch(
        [edge.fact for edge in filtered_edges]
    )
    for edge, fact_embedding in zip(filtered_edges, fact_embeddings, strict=True):
        edge.fact_embedding = fact_embedding
```

**Usage**:
- Find similar facts for deduplication
- Semantic search during query time
- Detect potential contradictions

### 3. Search Query Embeddings

**Purpose**: Real-time embedding of user queries for semantic search

**Generation**:
```python
# From graphiti_core/search/search.py
search_vector = await embedder.create(
    input_data=[query.replace('\n', ' ')]
)
```

**Usage**: Cosine similarity search against stored node/edge embeddings

## Qwen3 Embedder Implementation

### MLX-Based Implementation

```python
from graphiti_core.embedder.client import EmbedderClient
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
import numpy as np

class Qwen3EmbedderClient(EmbedderClient):
    """
    Qwen3 embedding client using MLX.

    Uses mlx-community/Qwen3-Embedding-4B-4bit-DWQ for local embeddings.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    ):
        from mlx_lm import load

        # Load Qwen3 embedding model
        self.model, self.tokenizer = load(model_name)

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size

    async def create(self, input_data: list[str]) -> list[list[float]]:
        """Create embeddings for input strings."""
        return await self.create_batch(input_data)

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        """Create embeddings in batch mode."""
        # Tokenize all inputs
        encoded = self.tokenizer(
            input_data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded['input_ids'])
        attention_mask = mx.array(encoded['attention_mask'])

        # Get model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract embeddings (mean pooling over sequence)
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        # Normalize embeddings
        embeddings = mx.nn.normalize(embeddings, axis=1)

        # Convert to list of lists
        return embeddings.tolist()

    def _mean_pooling(
        self,
        token_embeddings: mx.array,
        attention_mask: mx.array
    ) -> mx.array:
        """
        Mean pooling over sequence length.

        Takes attention mask into account for proper averaging.
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = mx.expand_dims(
            attention_mask,
            axis=-1
        ).astype(token_embeddings.dtype)

        # Sum embeddings, weighted by mask
        sum_embeddings = mx.sum(
            token_embeddings * input_mask_expanded,
            axis=1
        )

        # Sum mask values
        sum_mask = mx.clip(
            mx.sum(input_mask_expanded, axis=1),
            a_min=1e-9,
            a_max=None
        )

        # Average
        return sum_embeddings / sum_mask
```

### Alternative: Using sentence-transformers Interface

If Qwen3 embedding model supports sentence-transformers API:

```python
from graphiti_core.embedder.client import EmbedderClient
from sentence_transformers import SentenceTransformer
import asyncio

class Qwen3EmbedderClient(EmbedderClient):
    """Qwen3 embedder using sentence-transformers interface."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        self.model = SentenceTransformer(model_name, device="mps")  # MPS for Apple Silicon
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def create(self, input_data: list[str]) -> list[list[float]]:
        """Create embeddings."""
        # Run in executor for async compatibility
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.model.encode,
            input_data
        )
        return embeddings.tolist()

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        """Batch embedding (same as create)."""
        return await self.create(input_data)
```

## Qwen3 Reranker Implementation

Based on the sequence classification model structure:

```python
from graphiti_core.cross_encoder.client import CrossEncoderClient
import mlx.core as mx
from transformers import AutoTokenizer
import numpy as np

class Qwen3RerankerClient(CrossEncoderClient):
    """
    Qwen3 reranker using sequence classification.

    Uses tomaarsen/Qwen3-Reranker-0.6B-seq-cls for local reranking.
    """

    def __init__(
        self,
        model_name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    ):
        from mlx_lm import load

        # Load Qwen3 reranker model
        self.model, self.tokenizer = load(model_name)

    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        """Rank passages by relevance to query."""
        # Create query-passage pairs
        pairs = [f"{query} [SEP] {passage}" for passage in passages]

        # Tokenize
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded['input_ids'])
        attention_mask = mx.array(encoded['attention_mask'])

        # Get logits from sequence classification head
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, 2) for binary classification

        # Extract scores (probability of "relevant" class)
        # Assuming index 1 is the "relevant" class
        scores = mx.softmax(logits, axis=-1)[:, 1]

        # Convert to list
        scores_list = scores.tolist()

        # Create (passage, score) tuples and sort
        results = list(zip(passages, scores_list))
        results.sort(key=lambda x: x[1], reverse=True)

        return results
```

### Alternative: Using sentence-transformers CrossEncoder

If model compatible with sentence-transformers:

```python
from graphiti_core.cross_encoder.client import CrossEncoderClient
from sentence_transformers import CrossEncoder
import asyncio

class Qwen3RerankerClient(CrossEncoderClient):
    """Qwen3 reranker using CrossEncoder interface."""

    def __init__(
        self,
        model_name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    ):
        self.model = CrossEncoder(model_name, device="mps")

    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        """Rank passages."""
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]

        # Run in executor for async
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self.model.predict,
            pairs
        )

        # Create results and sort
        results = list(zip(passages, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        return results
```

## Integration in Graphiti

### Complete Initialization

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.driver.falkordb_driver import FalkorDriver
from your_module import (
    Qwen3EmbedderClient,
    Qwen3RerankerClient,
    DSPyLLMClient
)

# Initialize components
embedder = Qwen3EmbedderClient()
reranker = Qwen3RerankerClient()
llm_client = DSPyLLMClient()  # From DSPy integration
driver = FalkorDriver(host="localhost", port=6379)

# Create Graphiti instance
graphiti = Graphiti(
    uri="falkordb://localhost:6379",
    llm_client=llm_client,
    embedder=embedder,                 # ← Qwen3 embeddings
    cross_encoder=reranker,            # ← Qwen3 reranking
    graph_driver=driver
)
```

## Embedding Dimension Configuration

### Important: Match FalkorDB Vector Index

When creating FalkorDB vector indexes, dimension must match embedder:

```python
# Get embedding dimension from embedder
embedding_dim = embedder.embedding_dim  # e.g., 1024 for Qwen3-Embedding-4B

# Create vector index in FalkorDB
await driver.execute_query(f"""
    CALL db.idx.vector.create(
        'Entity',
        'name_embedding',
        'FP32',
        {embedding_dim},
        'COSINE'
    )
""")

await driver.execute_query(f"""
    CALL db.idx.vector.create(
        'RelatesToNode_',
        'fact_embedding',
        'FP32',
        {embedding_dim},
        'COSINE'
    )
""")
```

### Updating EMBEDDING_DIM Constant

If Qwen3 embedding dimension differs from default:

```python
# In your initialization
from graphiti_core.embedder import client as embedder_client

# Override global constant
embedder_client.EMBEDDING_DIM = embedder.embedding_dim
```

## Performance Considerations

### Batch Size

```python
# Good: Batch process multiple entities
embeddings = await embedder.create_batch([
    "Alice",
    "Bob",
    "Stanford University",
    "Microsoft"
])

# Less efficient: One at a time
embeddings = []
for entity in entities:
    emb = await embedder.create([entity])
    embeddings.append(emb[0])
```

### Caching

Embeddings are **persistent** and cached in the graph:

```python
# First time: Generate and save
node = EntityNode(name="Alice", ...)
await create_entity_node_embeddings(embedder, [node])
await node.save(driver)  # Embedding saved to FalkorDB

# Later: Load from database
loaded_node = await EntityNode.get_by_uuid(driver, node.uuid)
# loaded_node.name_embedding is already populated from DB
```

### MLX Optimization

For Apple Silicon, MLX provides:
- **Unified memory**: Efficient CPU-GPU data transfer
- **Quantization**: 4-bit models for speed
- **Batching**: Process multiple inputs efficiently

```python
# Optimize batch size for MLX
# Typical sweet spot: 16-32 for 4-bit models
OPTIMAL_BATCH_SIZE = 32

async def embed_in_batches(
    embedder: Qwen3EmbedderClient,
    texts: list[str]
) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), OPTIMAL_BATCH_SIZE):
        batch = texts[i:i + OPTIMAL_BATCH_SIZE]
        embeddings = await embedder.create_batch(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings
```

## Testing

### Unit Test for Embedder

```python
import pytest
from your_module import Qwen3EmbedderClient

@pytest.mark.asyncio
async def test_qwen3_embedder():
    embedder = Qwen3EmbedderClient()

    texts = ["Alice", "Bob", "Stanford University"]
    embeddings = await embedder.create_batch(texts)

    # Check dimension
    assert all(len(emb) == embedder.embedding_dim for emb in embeddings)

    # Check normalization (if using normalized embeddings)
    import numpy as np
    norms = [np.linalg.norm(emb) for emb in embeddings]
    assert all(abs(norm - 1.0) < 1e-5 for norm in norms)

    # Check similarity makes sense
    # "Alice" and "Bob" should be more similar than "Alice" and "Stanford"
    alice_bob_sim = np.dot(embeddings[0], embeddings[1])
    alice_stanford_sim = np.dot(embeddings[0], embeddings[2])
    assert alice_bob_sim > alice_stanford_sim
```

### Unit Test for Reranker

```python
@pytest.mark.asyncio
async def test_qwen3_reranker():
    reranker = Qwen3RerankerClient()

    query = "What is Alice's position?"
    passages = [
        "Alice works as a research scientist",
        "Bob is a software engineer",
        "The weather is sunny"
    ]

    results = await reranker.rank(query, passages)

    # Check sorting (descending scores)
    assert results[0][1] >= results[1][1] >= results[2][1]

    # Check most relevant passage is first
    assert "Alice" in results[0][0]
    assert "research scientist" in results[0][0]
```

## Summary

### Key Integration Points

1. **EmbedderClient Implementation**
   - Implement `create()` and `create_batch()` methods
   - Use Qwen3-Embedding-4B-4bit-DWQ via MLX
   - Return embeddings as `list[list[float]]`

2. **CrossEncoderClient Implementation**
   - Implement `rank()` method
   - Use Qwen3-Reranker-0.6B-seq-cls via MLX
   - Return sorted `list[tuple[str, float]]`

3. **Graphiti Initialization**
   - Pass custom embedder and reranker to `Graphiti()`
   - Ensure embedding dimension matches FalkorDB indexes
   - Configure search to use cross-encoder reranking

### Benefits of Qwen3 Models

✅ **Local Inference**: No API costs, no rate limits
✅ **Apple Silicon Optimized**: Fast via MLX
✅ **Small Models**: 0.6B (reranker) and 4B (embedder) fit in memory
✅ **High Quality**: Qwen3 family has strong performance
✅ **Quantized**: 4-bit for speed without major accuracy loss

### Next Steps

1. Test Qwen3 models separately for embedding/reranking quality
2. Benchmark performance on Apple Silicon
3. Compare accuracy vs OpenAI/BGE baselines
4. Tune batch sizes for optimal MLX throughput
5. Integrate with custom DSPy pipeline
