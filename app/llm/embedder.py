"""MLX embedder for Graphiti."""

import asyncio
import logging
from typing import Iterable

import mlx.core as mx
from graphiti_core.embedder import EmbedderClient

# Import the global MLX lock from client module
from .client import MLX_LOCK
from .batcher import RequestBatcher

logger = logging.getLogger(__name__)


class MLXEmbedder(EmbedderClient):
    """
    Local embedder using MLX for Apple Silicon.

    Uses mean pooling over the last hidden state to generate embeddings
    compatible with Graphiti's vector storage.
    """

    def __init__(self, model, tokenizer, enable_batching: bool = True):
        """
        Initialize MLX embedder.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
            enable_batching: Enable request batching for better throughput
        """
        self.model = model
        self.tokenizer = tokenizer
        self.enable_batching = enable_batching

        # Initialize batcher if enabled
        if enable_batching:
            self.batcher = RequestBatcher(
                batch_fn=self._embed_batch_sync,
                batch_window=0.01,  # 10ms collection window
                max_batch_size=32,
            )
        else:
            self.batcher = None

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

        # Use batcher if enabled
        if self.batcher:
            return await self.batcher.submit(input_data)
        else:
            # Fallback to serial processing
            async with MLX_LOCK:
                return await asyncio.to_thread(self._embed_sync, input_data)

    def _embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding computation for single input.

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

    async def _embed_batch_sync(
        self, batch_args: list[tuple], batch_kwargs: list[dict]
    ) -> list[list[float]]:
        """
        Batched embedding computation.

        Processes multiple texts in a single forward pass for better throughput.

        Args:
            batch_args: List of (text,) tuples
            batch_kwargs: List of empty dicts (unused)

        Returns:
            List of embedding vectors, one per input
        """
        texts = [args[0] for args in batch_args]

        logger.info(f"[MLXEmbedder] Batching {len(texts)} embeddings")

        async with MLX_LOCK:
            return await asyncio.to_thread(self._embed_batch_impl, texts)

    def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronous batched embedding implementation.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        # Tokenize all inputs
        all_tokens = [self.tokenizer.encode(text) for text in texts]

        # Pad to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        for tokens in all_tokens:
            padded = tokens + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)

        # Convert to MLX array: [batch_size, seq_len]
        batch_tokens = mx.array(padded_tokens)

        # Single forward pass for entire batch
        outputs = self.model(batch_tokens)

        # Mean pooling over sequence dimension for each item
        # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        embeddings = mx.mean(outputs, axis=1)

        # Convert each embedding to list
        return [emb.tolist() for emb in embeddings]

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
