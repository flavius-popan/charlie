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
