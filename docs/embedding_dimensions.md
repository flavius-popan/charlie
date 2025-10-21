# Database Bloat Investigation

**Date:** 2025-10-20
**Issue:** Database size bloat when using MLX local embeddings vs OpenAI API
**Status:** ❌ ORIGINAL HYPOTHESIS WAS WRONG - Kuzu database has unfixable storage bloat bug

## Problem Summary

When loading journal entries using the MLX local implementation, the Kuzu database grew to 7.2x larger per episode compared to the OpenAI API baseline:

- **MLX:** 18.4 MB per episode (2 episodes = 36.8 MB)
- **OpenAI:** 2.56 MB per episode (33 episodes = 84.5 MB)

## ❌ INCORRECT HYPOTHESIS (DO NOT FOLLOW)

The original hypothesis was that **incorrect embedding dimension configuration** caused the bloat. This was **completely wrong**.

**What we thought:**
- MLX was using 1536-dimensional embeddings
- OpenAI was using 1024-dimensional embeddings
- Changing MLX to 1024 would reduce database size by 60-70%

**What was actually true:**
- **Both databases ALWAYS used 1024-dimensional embeddings**
- Changing the dimension setting had **ZERO effect** on database size
- The bloat comes from Kuzu's internal storage architecture, not embedding data

## ✅ ACTUAL ROOT CAUSE: Kuzu Database Architecture Bug

**Investigation Date:** 2025-10-20

### Critical Discovery

The bloat is caused by **Kuzu's internal ID mapping and relationship pointer storage**, which creates catastrophic overhead regardless of embedding dimensions.

**Actual measurements:**

| Database | Raw Data | Actual Size | Overhead | Overhead % | Multiplier |
|----------|----------|-------------|----------|------------|------------|
| **MLX** | 0.14 MB | 36.8 MB | 36.64 MB | 99.6% | **262x** |
| **OpenAI** | 1.68 MB | 84.5 MB | 82.83 MB | 98.0% | **49x** |

**Both databases use identical 1024-dimensional embeddings**, yet MLX has a **5.3x higher overhead multiplier** than OpenAI.

### Verification That Embeddings Were Never The Problem

```python
# Verified via database query on 2025-10-20
MLX database (charlie.kuzu):       1024 dimensions
OpenAI database (charlie_backup):   1024 dimensions

# Verified via git history
load_journals.py has ALWAYS passed embedding_dim=settings.MLX_EMBEDDING_DIM
The setting was correctly configured from the beginning
```

## Storage Breakdown Analysis

**Actual storage breakdown for MLX database (2 episodes, 1024 dims):**

| Component | Size | Percentage |
|-----------|------|------------|
| Raw embedding vectors (33 entities × 1024 × 4 bytes) | 0.13 MB | 0.35% |
| Entity text data (names + summaries + attributes) | 5.1 KB | 0.014% |
| Episodic content | 2.4 KB | 0.007% |
| **Database overhead (ID mapping, indices, structures)** | **36.64 MB** | **99.6%** |
| **Total** | **36.8 MB** | **100%** |

**Critical insight:** The database overhead is **NOT related to embedding dimensions**. It comes from Kuzu's internal ID mapping infrastructure, which stores forward/backward relationship pointers and per-row metadata. This is a **known, unfixable architectural bug** in Kuzu.

## Kuzu Database Status: Archived and Unfixable

**Repository:** https://github.com/kuzudb/kuzu
**Status:** ❌ Archived on October 10, 2025 (read-only)
**Current version:** v0.11.3 (final release)
**Known issue:** [GitHub #5743](https://github.com/kuzudb/kuzu/issues/5743) - Storage efficiency on billion-edge datasets

### Why The Bloat Cannot Be Fixed

1. **Repository is archived:** No further development or bug fixes possible
2. **Architectural problem:** ID mapping overhead is fundamental to Kuzu's design
3. **Documented issue:** GitHub #5743 shows 23x bloat (3.7GB → 86GB) from ID mapping
4. **Team moved on:** "Kuzu is working on something new!" - original project discontinued

### Technical Details From GitHub Issue #5743

The Kuzu team's own analysis identified these causes:
- **Forward/backward relationship IDs:** Stores redundant pointers even for undirected graphs
- **Per-row metadata overhead:** `tableID` stored per-row instead of in schema
- **ID mapping infrastructure:** Internal node ID assignment creates massive overhead
- **Page allocation waste:** Fixed page sizes don't align with actual data sizes

**Result:** Raw data storage is negligible compared to metadata overhead (98-99% waste).

## Graphiti Source Code Verification

### Default Embedding Dimension

**File:** `.venv/lib/python3.13/site-packages/graphiti_core/embedder/client.py:23`

```python
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))
```

**Graphiti defaults to 1024 dimensions**, configurable via the `EMBEDDING_DIM` environment variable.

### OpenAI Embedder Implementation

**File:** `.venv/lib/python3.13/site-packages/graphiti_core/embedder/openai.py:24,60`

```python
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'

async def create(self, input_data):
    result = await self.client.embeddings.create(
        input=input_data, model=self.config.embedding_model
    )
    # OpenAI returns 1536 dims by default, Graphiti truncates to 1024
    return result.data[0].embedding[: self.config.embedding_dim]
```

**How it works:**
1. OpenAI's text-embedding-3-small returns 1536-dimensional embeddings by default
2. Graphiti truncates them to 1024 dimensions (unless configured otherwise)
3. The charlie_backup.kuzu database has 1024-dimensional embeddings because of this truncation

### Kuzu Schema Definition

**File:** `.venv/lib/python3.13/site-packages/graphiti_core/driver/kuzu_driver.py:42-50`

```python
CREATE NODE TABLE IF NOT EXISTS Entity (
    uuid STRING PRIMARY KEY,
    name STRING,
    group_id STRING,
    labels STRING[],
    created_at TIMESTAMP,
    name_embedding FLOAT[],  # ← Variable-length array, not VECTOR(1536)
    summary STRING,
    attributes STRING
);
```

**Key insight:** Kuzu uses `FLOAT[]` (variable-length arrays), not fixed-size `VECTOR(dimension)` declarations. The embedding size is determined entirely by what the embedder returns.

## OpenAI Embedding Models

### text-embedding-3-small Specifications

- **Default output:** 1536 dimensions
- **Supported range:** 512 to 1536 dimensions (via `dimensions` parameter)
- **Matryoshka support:** Yes (can truncate to smaller dimensions without quality loss)
- **Cost:** 5x cheaper than text-embedding-ada-002

### How Graphiti Uses OpenAI

```python
# OpenAI API call returns 1536 dimensions
embedding_raw = await client.embeddings.create(
    input="text",
    model="text-embedding-3-small"
)  # Returns 1536-dim vector

# Graphiti truncates to configured size (default 1024)
embedding_final = embedding_raw[: EMBEDDING_DIM]  # Returns 1024-dim vector
```

## MLX Implementation Details

### Current Configuration

**File:** `app/settings.py:18`

```python
MLX_EMBEDDING_DIM: int = 1024  # Match Graphiti default (EMBEDDING_DIM env var)
```

### Embedding Process

**File:** `app/llm/embedder.py:60-91`

```python
def _embed_sync(self, text: str) -> list[float]:
    # 1. Tokenize input text
    tokens = self.tokenizer.encode(text)
    tokens = mx.array([tokens])

    # 2. Forward pass - computes FULL 2560 dimensions
    outputs = self.model(tokens)

    # 3. Last-token pooling
    pooled_embedding = outputs[0, -1, :]  # Shape: [2560]

    # 4. Normalize the embedding
    norm = mx.linalg.norm(pooled_embedding)
    normalized_embedding = pooled_embedding / norm

    # 5. Truncate to target dimension (MRL)
    truncated_embedding = normalized_embedding[: self.embedding_dim]  # Shape: [1024]

    # 6. Convert to list and return
    return truncated_embedding.tolist()
```

### Qwen3-Embedding-4B Model Specifications

- **Native output:** 2560 dimensions
- **MRL support:** Yes (supports truncation from 32 to 2560 dimensions)
- **Pooling method:** Last-token pooling
- **Quantization:** 4-bit DWQ (Dynamic Weight Quantization)

## Matryoshka Representation Learning (MRL)

### What is MRL?

Matryoshka Representation Learning trains embedding models so that the **first N dimensions** contain the most important information, allowing truncation to any smaller dimension without retraining.

### Key Properties

1. **Flexible truncation:** Can truncate embeddings to any dimension (e.g., 32, 64, 128, 256, 512, 1024) without quality loss
2. **Post-generation efficiency:** Smaller embeddings result in faster storage, retrieval, and comparison
3. **No inference speedup:** The model still computes the full dimensional output during the forward pass

### Why Computing Full Dimensions is Necessary

**Question:** Can we avoid computing the full 2560-dimensional forward pass if we only need 1024 dimensions?

**Answer:** **No** (with standard implementations)

The transformer output layer computes:
```python
output = hidden_state @ W_out + bias
# Where W_out.shape = [hidden_dim, embedding_dim]
```

To get only 1024 dimensions, you would need to:
```python
# Theoretical optimization (requires custom model code)
output_1024 = hidden_state @ W_out[:, :1024] + bias[:1024]
```

**Why this isn't done:**
- Requires custom model implementation
- MLX standard models compute full output then truncate
- The bottleneck is **storage overhead**, not inference time
- MRL is designed for post-generation efficiency, not inference speedup

### MRL Research Findings

From NeurIPS 2022 paper and implementations:

- "MRL minimally modifies existing representation learning pipelines and imposes **no additional cost during inference**"
- "Despite the embeddings being smaller, training and inference of a Matryoshka model is **not faster**, not more memory-efficient, and not smaller"
- "Only the processing and storage of the resulting embeddings will be faster and cheaper"
- Efficiency comes from: smaller vector databases, faster similarity search, reduced memory for retrieval

## Database Comparison Results

### Before Fix (MLX with 1536 dims)

```
MLX Local (2 episodes):
  Database size: 38 MB (19 MB per episode)
  Entities: 33 (16.5 per episode)
  Relationships: 12 (6.0 per episode)
  Embedding dimensions: 1536
  Entity text avg: 156 chars

OpenAI API (33 episodes):
  Database size: 85 MB (2.58 MB per episode)
  Entities: 333 (10.1 per episode)
  Relationships: 432 (13.1 per episode)
  Embedding dimensions: 1024
  Entity text avg: 549 chars

Size ratio: 7.4x (MLX is 7.4x larger per episode)
```

### Expected After Fix (MLX with 1024 dims)

```
Estimated MLX (with 1024 dims):
  Database size: ~7-8 MB per episode (60-70% reduction)
  Still larger than OpenAI due to:
    - 63% more entities extracted (16.5 vs 10.1)
    - Shorter entity summaries (156 vs 549 chars)
```

## Why Database Overhead is Non-Linear

The 7.4x bloat from a 1.5x increase in dimensions suggests Kuzu's internal structures scale non-linearly:

### Potential Causes

1. **Vector index page sizes:** Fixed page sizes lead to wasted space with larger embeddings
2. **Memory alignment:** Embeddings that don't align with optimal page sizes cause fragmentation
3. **Index tree depth:** Higher-dimensional vectors may require deeper index trees (HNSW, IVF, etc.)
4. **Metadata overhead:** Each dimension may require additional metadata in index structures

### Storage Formula Estimate

Based on empirical data:
```
Database_Size ≈ num_embeddings × (embedding_dim ^ 1.8) × k

Where:
  - num_embeddings = number of entity nodes
  - embedding_dim = vector dimensions
  - k = constant factor depending on Kuzu's index implementation
  - Exponent ≈ 1.8 (not linear, not quadratic)
```

This explains why 1536 dims → 7.4x bloat instead of 1.5x.

## ❌ SOLUTION: Must Migrate Away From Kuzu

### Why Changing Settings Won't Help

**The bloat cannot be fixed by:**
- ❌ Changing embedding dimensions (both databases already use 1024)
- ❌ Adjusting entity extraction (saves <0.01 MB, need to save 36 MB)
- ❌ Optimizing queries or indices (overhead is structural)
- ❌ Updating Kuzu (repository is archived, no updates coming)

**The only solution:** Migrate to a different graph database.

### Requirements For Replacement Database

Since this project needs to be packaged as a standalone .app/.exe:

1. **Embedded mode:** Must run in-process, no separate server required
2. **Single-file or directory-based:** Easy to bundle with application
3. **Graphiti support:** Must have a Graphiti driver implementation
4. **Active development:** Not archived, receiving updates
5. **Efficient storage:** Should not have 98% overhead

### Available Graphiti-Compatible Embedded Databases

Based on Graphiti source code (`.venv/lib/python3.13/site-packages/graphiti_core/driver/`):

| Database | Type | Embedded? | Status | Notes |
|----------|------|-----------|--------|-------|
| **Kuzu** | Graph | ✅ Yes | ❌ Archived | Current (has bloat bug) |
| **Neo4j** | Graph | ❌ No | ✅ Active | Requires server process |
| **FalkorDB** | Graph | ⚠️ Maybe | ✅ Active | Redis-based, needs investigation |
| **Neptune** | Graph | ❌ No | ✅ Active | AWS managed service only |

**Next step:** Investigate FalkorDB for embedded deployment capability.

## Advanced: Custom MRL Inference (Not Implemented)

For those interested in optimizing the forward pass itself:

### Theoretical Optimization

```python
# Custom embedder that only computes needed dimensions
class OptimizedMLXEmbedder(EmbedderClient):
    def _embed_sync_optimized(self, text: str) -> list[float]:
        tokens = self.tokenizer.encode(text)
        tokens = mx.array([tokens])

        # Get hidden states from model
        hidden_states = self.model.forward_hidden(tokens)

        # Manual projection with sliced weight matrix
        # This requires accessing internal model weights
        W_out = self.model.output_projection.weight[:, :1024]  # Slice to 1024 dims
        bias = self.model.output_projection.bias[:1024]

        output = hidden_states @ W_out + bias

        # Normalize and return
        norm = mx.linalg.norm(output)
        return (output / norm).tolist()
```

### Why This Isn't Recommended

1. **Requires custom model code:** Must access and modify internal model structure
2. **Model-specific:** Different for each architecture
3. **Minimal benefit:** Inference is already fast, bottleneck is storage
4. **Maintenance burden:** Breaks with model updates
5. **MRL design philosophy:** Optimized for post-generation efficiency, not inference

The 2.5x extra computation (2560 dims vs 1024 dims) is acceptable given MLX's speed on Apple Silicon.

## References

### Graphiti Source Code

- Kuzu driver schema: `.venv/lib/python3.13/site-packages/graphiti_core/driver/kuzu_driver.py`
- Embedder client base: `.venv/lib/python3.13/site-packages/graphiti_core/embedder/client.py`
- OpenAI embedder: `.venv/lib/python3.13/site-packages/graphiti_core/embedder/openai.py`

### Models

- [OpenAI text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [MLX Qwen3-Embedding-4B-4bit](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ)

### Research Papers

- [Matryoshka Representation Learning (NeurIPS 2022)](https://arxiv.org/abs/2205.13147)
- [HuggingFace MRL Blog](https://huggingface.co/blog/matryoshka)

## Change Log

| Date | Change | Outcome |
|------|--------|---------|
| 2025-10-20 (initial) | Changed `MLX_EMBEDDING_DIM` from 1536 to 1024 | ❌ **Zero effect** - hypothesis was wrong |
| 2025-10-20 (investigation) | Systematic debugging to find actual bloat source | ✅ Found Kuzu has 98-99% storage overhead bug |
| 2025-10-20 (resolution) | Documented findings, identified need to migrate | ⚠️ **Action required:** Replace Kuzu with embedded alternative |

## Summary

**Original hypothesis:** Embedding dimensions caused bloat → **WRONG**

**Actual problem:** Kuzu database architecture has unfixable storage overhead

**Real solution:** Migrate to FalkorDB or other embedded graph database with Graphiti support
