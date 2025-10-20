"""Quick test to check actual embedding dimensions."""

import asyncio
import mlx_lm
from app import settings
from app.llm.embedder import MLXEmbedder


async def test_embedding_size():
    print(f"Loading embedding model: {settings.MLX_EMBEDDING_MODEL_NAME}")
    mlx_embedding_model, mlx_embedding_tokenizer = mlx_lm.load(
        settings.MLX_EMBEDDING_MODEL_NAME
    )

    print(f"Initializing embedder with dimension: {settings.MLX_EMBEDDING_DIM}")
    embedder = MLXEmbedder(
        mlx_embedding_model,
        mlx_embedding_tokenizer,
        embedding_dim=settings.MLX_EMBEDDING_DIM,
    )

    test_text = "This is a test sentence."
    print(f"\nGenerating embedding for: '{test_text}'")

    embedding = await embedder.create(test_text)

    print(f"\nEmbedding dimensions: {len(embedding)}")
    print(f"Embedding size in bytes (float32): {len(embedding) * 4}")
    print(f"Embedding size in KB: {len(embedding) * 4 / 1024:.2f}")
    print(f"Embedding size in MB: {len(embedding) * 4 / 1024 / 1024:.4f}")

    # Estimate size for a typical journal entry
    # Graphiti typically creates embeddings for: episodes, entities, edges, communities
    # A single episode might generate ~20-50 embeddings
    print("\n--- Size Estimates ---")
    print(f"Per embedding: {len(embedding) * 4 / 1024:.2f} KB")
    print(f"Per episode (~30 embeddings): {len(embedding) * 4 * 30 / 1024:.2f} KB")
    print(
        f"33 episodes (~990 embeddings): {len(embedding) * 4 * 990 / 1024 / 1024:.2f} MB"
    )


if __name__ == "__main__":
    asyncio.run(test_embedding_size())
