# Cross-Encoder and Reranking Subsystem

## Overview

The cross-encoder reranking subsystem is a **critical post-processing step** in Graphiti's search pipeline that improves result relevance by rescoring passages based on query-passage similarity. This operates after initial retrieval (embedding search + BM25) to provide more accurate final rankings.

**Key Locations:**
- `graphiti_core/cross_encoder/client.py` - Abstract interface
- `graphiti_core/cross_encoder/openai_reranker_client.py` - OpenAI implementation
- `graphiti_core/cross_encoder/gemini_reranker_client.py` - Gemini implementation
- `graphiti_core/cross_encoder/bge_reranker_client.py` - BGE local model
- `graphiti_core/search/search.py` - Integration in search pipeline

## Cross-Encoder vs Bi-Encoder Architecture

### Comparison Table

| Aspect | Bi-Encoder (Embeddings) | Cross-Encoder (Reranking) |
|--------|-------------------------|---------------------------|
| **Processing** | Query and passages encoded separately | Query and passage processed **together** |
| **Computational Cost** | Lower (cached embeddings) | Higher (real-time processing) |
| **Accuracy** | Good for broad semantic similarity | **Better for precise relevance** |
| **Scalability** | High (vector similarity) | Limited (pairwise scoring) |
| **Use Case** | Initial retrieval | Final reranking |
| **When Run** | Offline (embeddings cached) | Online (per query) |

### Why Cross-Encoders Are More Accurate

**Bi-Encoder (Embedding Search):**
```
Query → Encoder → [q1, q2, q3, ...] (query embedding)
Passage → Encoder → [p1, p2, p3, ...] (passage embedding)

Similarity = cosine(query_embedding, passage_embedding)
```

**Cross-Encoder (Reranking):**
```
[Query + Passage] → Joint Encoder → Relevance Score

"How does Alice know Bob?" + "Alice met Bob at Stanford in 2020"
    ↓
Transformer processes ENTIRE sequence together
    ↓
Direct relevance score (not embedding similarity)
```

**Key Difference**: Cross-encoders see the query and passage together, allowing attention mechanisms to capture query-specific relevance patterns that bi-encoders miss.

## CrossEncoderClient Interface

### Abstract Base Class

**Location**: `graphiti_core/cross_encoder/client.py:20-40`

```python
class CrossEncoderClient(ABC):
    """
    CrossEncoderClient is an abstract base class that defines the interface
    for cross-encoder models used for ranking passages based on their
    relevance to a query.
    """

    @abstractmethod
    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        """
        Rank the given passages based on their relevance to the query.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage
                                     and its score, sorted in descending
                                     order of relevance.
        """
        pass
```

**Key Points:**
- Simple interface: `query` + `passages` → scored and sorted results
- Async for concurrent processing
- Scores are normalized (typically 0.0 to 1.0)
- Results **already sorted** in descending order (best first)

## Implementation: OpenAI Reranker

### Architecture

**Location**: `graphiti_core/cross_encoder/openai_reranker_client.py:61-118`

Uses OpenAI's API with a **boolean classification approach** leveraging log probabilities for scoring.

### Scoring Method

**Prompt Template:**
```
Is the following passage relevant to this query? Answer 'True' or 'False'.

Query: {query}

Passage: {passage}
```

**Logit Bias:**
```python
logit_bias = {
    6432: 100,  # Token for 'True'
    7983: 100,  # Token for 'False'
}
```

This encourages the model to respond with exactly "True" or "False".

### Score Calculation

```python
# Extract log probability from response
top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs

# Normalize to probability (0-1)
norm_logprobs = np.exp(top_logprobs[0].logprob)

# Score based on response
if top_logprobs[0].token.strip().split(' ')[0].lower() == 'true':
    score = norm_logprobs  # High probability of "True" = high relevance
else:
    score = 1 - norm_logprobs  # High probability of "False" = low relevance
```

**Key Features:**
- Model: `gpt-4o-mini` by default
- Concurrent processing with `semaphore_gather`
- Returns scores normalized to [0, 1]
- Sorted results (best first)

### Example Usage

```python
from graphiti_core.cross_encoder import OpenAIRerankerClient

reranker = OpenAIRerankerClient()

query = "What is Alice's role at Stanford?"
passages = [
    "Alice works at Stanford as a research scientist",
    "Bob founded Microsoft in 1975",
    "Stanford is located in California",
]

results = await reranker.rank(query, passages)
# [
#   ("Alice works at Stanford as a research scientist", 0.92),
#   ("Stanford is located in California", 0.45),
#   ("Bob founded Microsoft in 1975", 0.12)
# ]
```

## Implementation: Gemini Reranker

### Architecture

**Location**: `graphiti_core/cross_encoder/gemini_reranker_client.py:73-147`

Uses Gemini API with **direct relevance scoring** on a 0-100 scale.

### Scoring Prompt

```
Rate the relevance of the following passage to the query on a scale
from 0 to 100. Only return the numeric score.

Query: {query}

Passage: {passage}

Relevance Score (0-100):
```

### Score Extraction

```python
# Extract score from response using regex
score_match = re.search(r'\b\d+\b', response.text)

if score_match:
    score = float(score_match.group()) / 100.0  # Normalize to [0, 1]
else:
    score = 0.0  # Fallback for unparseable responses
```

**Key Features:**
- Direct scoring (no log probabilities available in Gemini API)
- Regex extraction handles various response formats
- Concurrent processing
- Fallback to 0.0 for errors

**Trade-off**: Less precise than log probability approach, but works with Gemini API constraints.

## Implementation: BGE Reranker (Local)

### Architecture

**Location**: `graphiti_core/cross_encoder/bge_reranker_client.py:34-54`

Uses **BAAI BGE reranker model** via sentence-transformers library for **local neural reranking**.

### Model Details

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('BAAI/bge-reranker-v2-m3')
```

**Model Characteristics:**
- Multi-lingual support (m3 = multilingual, multimodal, multi-granularity)
- ~550M parameters
- Runs locally (no API calls)
- Requires ~2GB GPU memory (or CPU fallback)

### Scoring Process

```python
# Create query-passage pairs
pairs = [[query, passage] for passage in passages]

# Batch score
scores = model.predict(pairs)

# Sort and return
results = sorted(
    zip(passages, scores),
    key=lambda x: x[1],
    reverse=True
)
```

**Key Features:**
- No external API dependencies
- Consistent scoring (deterministic)
- Async-compatible (runs in thread executor)
- Auto-downloads model on first use

**Trade-off**: Slower than API-based solutions for small batches, but no rate limits or costs.

## Integration in Search Pipeline

### Search Flow with Reranking

```
User Query
    ↓
1. Initial Retrieval (Hybrid Search)
    ├─ BM25 (keyword matching)
    └─ Cosine Similarity (semantic embedding search)
    ↓
2. Reciprocal Rank Fusion (RRF)
    - Combines BM25 and semantic results
    - Produces top K candidates (e.g., K=50)
    ↓
3. Cross-Encoder Reranking  ← THIS STEP
    - Rescore top K candidates
    - More expensive but more accurate
    - Produces final ranked list (e.g., top 10)
    ↓
Final Results (sorted by cross-encoder scores)
```

### Configuration in Search Configs

**Location**: `graphiti_core/search/search_config_recipes.py`

Cross-encoder reranking is configured via `SearchConfig`:

```python
from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig
from graphiti_core.search.search_config import EdgeSearchMethod, EdgeReranker

# With cross-encoder reranking
EDGE_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity
        ],
        reranker=EdgeReranker.cross_encoder,  # ← Enables cross-encoder
        limit=10
    )
)

# Without reranking (RRF only)
EDGE_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity
        ],
        reranker=EdgeReranker.rrf,  # ← RRF only, no cross-encoder
        limit=10
    )
)
```

### When Reranking Is Used

**Typical Pattern:**
1. Retrieve top 50-100 results with BM25 + embedding search
2. Rerank using cross-encoder to get best 10-20
3. Return final results to user

**Cost Trade-off:**
- Reranking 100 passages = 100 LLM calls (OpenAI/Gemini) or 100 forward passes (BGE)
- Expensive, so only applied to top candidates
- Improves precision@10 significantly

## Custom Implementation for Qwen3

### Requirements

To implement a custom cross-encoder using Qwen3, you need to:

1. **Implement `CrossEncoderClient` interface**
2. **Define scoring method** (e.g., boolean classification, direct scoring, or regression)
3. **Handle concurrent processing** (async compatibility)
4. **Normalize scores** to [0, 1] range
5. **Sort results** by score (descending)

### Qwen3 Reranker Implementation Pattern

```python
from graphiti_core.cross_encoder.client import CrossEncoderClient
import mlx.core as mx
from mlx_lm import load, generate

class Qwen3RerankerClient(CrossEncoderClient):
    """Cross-encoder reranker using Qwen3 via MLX."""

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"):
        self.model, self.tokenizer = load(model_name)

    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        """
        Rank passages using Qwen3.

        Strategy: Use boolean classification with logit extraction
        similar to OpenAI reranker.
        """
        scores = []

        for passage in passages:
            # Create prompt
            prompt = self._create_prompt(query, passage)

            # Get logits for "Yes" and "No" tokens
            score = await self._score_passage(prompt)
            scores.append((passage, score))

        # Sort by score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _create_prompt(self, query: str, passage: str) -> str:
        """Create relevance classification prompt."""
        return f"""<|im_start|>system
You are a relevance classifier. Determine if the passage is relevant to the query.
Answer with ONLY 'Yes' or 'No'.<|im_end|>
<|im_start|>user
Query: {query}

Passage: {passage}

Is the passage relevant to the query? (Yes/No)<|im_end|>
<|im_start|>assistant
"""

    async def _score_passage(self, prompt: str) -> float:
        """
        Score passage using logit extraction.

        Extract logits for "Yes" and "No" tokens, compute
        probability of "Yes" as relevance score.
        """
        # Tokenize
        tokens = self.tokenizer.encode(prompt)

        # Get logits (run model forward pass)
        logits = self.model(mx.array([tokens]))

        # Get logits for next token (the response position)
        next_token_logits = logits[0, -1, :]

        # Find token IDs for "Yes" and "No"
        yes_token_id = self.tokenizer.encode("Yes")[0]
        no_token_id = self.tokenizer.encode("No")[0]

        # Extract logits for these tokens
        yes_logit = next_token_logits[yes_token_id].item()
        no_logit = next_token_logits[no_token_id].item()

        # Compute softmax probability for "Yes"
        import math
        exp_yes = math.exp(yes_logit)
        exp_no = math.exp(no_logit)
        prob_yes = exp_yes / (exp_yes + exp_no)

        return prob_yes
```

### Alternative: Direct Scoring with Constrained Generation

```python
from pydantic import BaseModel, Field
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

class RelevanceScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)"
    )

class ScorePassageSignature(dspy.Signature):
    """Score passage relevance to query."""

    query: str = dspy.InputField()
    passage: str = dspy.InputField()

    score: RelevanceScore = dspy.OutputField()

class Qwen3RerankerDSPy(CrossEncoderClient):
    """DSPy-based reranker with constrained generation."""

    def __init__(self):
        lm = OutlinesLM(model_name="mlx-community/Qwen2.5-3B-Instruct-4bit")
        adapter = OutlinesAdapter()
        dspy.configure(lm=lm, adapter=adapter)

        self.scorer = dspy.Predict(ScorePassageSignature)

    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        # Score all passages
        results = []
        for passage in passages:
            result = self.scorer(query=query, passage=passage)
            score = result.score.score
            results.append((passage, score))

        # Sort by score
        return sorted(results, key=lambda x: x[1], reverse=True)
```

## Performance Considerations

### Batch Size

Cross-encoders process each query-passage pair independently. For efficiency:

```python
# Good: Process top 20 candidates
candidates = initial_search(query, limit=100)
top_candidates = candidates[:20]
final_results = reranker.rank(query, [c.passage for c in top_candidates])

# Bad: Rerank all 10,000 results
all_results = graph.get_all_edges()
reranked = reranker.rank(query, [r.fact for r in all_results])  # Too slow!
```

**Typical Settings:**
- Initial retrieval: 50-100 results
- Reranking: Top 20-50 of those
- Final return: Top 10-20

### Concurrent Processing

All implementations use `semaphore_gather` for parallel processing:

```python
from graphiti_core.helpers import semaphore_gather

# Process passages in parallel
scores = await semaphore_gather(
    *[self._score_passage(query, passage) for passage in passages]
)
```

This is critical for performance with API-based rerankers (OpenAI, Gemini).

### Caching Strategies

**Don't Cache:**
- Cross-encoder scores (query-dependent)

**Do Cache:**
- Embeddings (query-independent)
- Initial retrieval results (if queries repeat)

## Integration Points in Graphiti

### 1. Graphiti Initialization

```python
from graphiti_core import Graphiti
from graphiti_core.cross_encoder import BGERerankerClient

graphiti = Graphiti(
    uri="falkordb://localhost:6379",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=BGERerankerClient(),  # ← Custom reranker here
    graph_driver=driver
)
```

### 2. Search Configuration

Cross-encoder is used when `EdgeReranker.cross_encoder` is specified:

```python
from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig
from graphiti_core.search.search_config import EdgeReranker

config = SearchConfig(
    edge_config=EdgeSearchConfig(
        reranker=EdgeReranker.cross_encoder  # ← Triggers cross-encoder
    )
)

results = await graphiti.search_(query, config=config)
```

### 3. Custom Search Without Reranking

```python
# Use RRF only (faster, less accurate)
config = SearchConfig(
    edge_config=EdgeSearchConfig(
        reranker=EdgeReranker.rrf  # ← Skip cross-encoder
    )
)
```

## Testing Custom Reranker

### Unit Test Pattern

```python
import pytest
from your_module import Qwen3RerankerClient

@pytest.mark.asyncio
async def test_qwen3_reranker():
    reranker = Qwen3RerankerClient()

    query = "What is Alice's position at Stanford?"
    passages = [
        "Alice works at Stanford as a research scientist",
        "Bob founded Microsoft",
        "Stanford is in California"
    ]

    results = await reranker.rank(query, passages)

    # Check sorting (descending)
    assert results[0][1] >= results[1][1] >= results[2][1]

    # Check scores in valid range
    for passage, score in results:
        assert 0.0 <= score <= 1.0

    # Check most relevant is first
    assert "Alice" in results[0][0]
    assert "Stanford" in results[0][0]
```

### Integration Test with Search

```python
@pytest.mark.asyncio
async def test_search_with_custom_reranker():
    # Setup graphiti with custom reranker
    graphiti = Graphiti(
        uri="falkordb://localhost:6379",
        cross_encoder=Qwen3RerankerClient(),
        # ... other params
    )

    # Add some test data
    await graphiti.add_episode(...)

    # Search with cross-encoder reranking
    config = SearchConfig(
        edge_config=EdgeSearchConfig(
            reranker=EdgeReranker.cross_encoder
        )
    )

    results = await graphiti.search_("test query", config=config)

    # Verify results are reranked
    assert len(results.edges) > 0
```

## Summary

### Key Points

1. **Cross-encoders improve search accuracy** by processing query-passage pairs jointly
2. **Simple interface**: `rank(query, passages) → [(passage, score)]`
3. **Three implementations**: OpenAI (log probs), Gemini (direct scoring), BGE (local)
4. **Used in search pipeline** as final reranking step
5. **Computational cost**: O(n) LLM calls for n passages
6. **Integration**: Via `cross_encoder` parameter in Graphiti initialization

### For Custom Pipeline with Qwen3

**Requirements:**
1. Implement `CrossEncoderClient` abstract class
2. Define `rank(query, passages)` method
3. Choose scoring strategy (logit extraction OR constrained generation)
4. Handle async processing
5. Normalize scores to [0, 1]
6. Sort results (descending)

**Recommended Approach:**
- Use **logit extraction** for boolean classification (similar to OpenAI)
- More efficient than full generation
- More reliable than regex parsing
- Leverages Qwen3's instruction-following capabilities

**Integration:**
```python
graphiti = Graphiti(
    cross_encoder=Qwen3RerankerClient(),
    # ... other params
)
```

This enables high-quality reranking using local Qwen3 models via MLX.
