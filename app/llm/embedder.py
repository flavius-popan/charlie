import asyncio
import logging
from typing import Iterable

import mlx.core as mx
from graphiti_core.embedder import EmbedderClient

# Import the global MLX lock from client module
from .client import MLX_LOCK

logger = logging.getLogger(__name__)


class MLXEmbedder(EmbedderClient):
    """
    Local embedder using MLX for Apple Silicon.

    Uses last-token pooling, normalization, and MRL truncation to generate
    embeddings compatible with Graphiti's vector storage.
    """

    def __init__(
        self,
        model,
        tokenizer,
        embedding_dim: int = 1536,
    ):
        """
        Initialize MLX embedder.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
            embedding_dim: Target dimension for embeddings (MRL truncation)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embedding for text input (no batching).

        Args:
            input_data: Text string to embed (or list with single string)

        Returns:
            List of floats representing the embedding vector
        """
        # Handle list input (take first element)
        if isinstance(input_data, list):
            input_data = input_data[0]

        # Simple serial processing with MLX lock
        async with MLX_LOCK:
            return await asyncio.to_thread(self._embed_sync, input_data)

    def _embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding computation for single input.

        Process:
        1. Tokenize input text
        2. Forward pass through model
        3. Pool using the last token's hidden state
        4. Normalize the embedding vector
        5. Truncate to the target dimension (MRL)
        6. Convert to list and return
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens = mx.array([tokens])

        # Forward pass - get last hidden state
        outputs = self.model(tokens)

        # Last-token pooling
        # Shape: [1, seq_len, hidden_dim] -> [hidden_dim]
        pooled_embedding = outputs[0, -1, :]

        # Normalize the embedding
        norm = mx.linalg.norm(pooled_embedding)
        normalized_embedding = pooled_embedding / norm

        # Truncate the embedding (Matryoshka Representation Learning)
        truncated_embedding = normalized_embedding[: self.embedding_dim]

        # Convert to list
        return truncated_embedding.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for batch of text inputs (serial processing).

        Args:
            input_data_list: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input_data_list:
            embeddings.append(await self.create(text))
        return embeddings
