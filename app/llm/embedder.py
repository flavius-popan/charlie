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

    Uses last-token pooling, normalization, and MRL truncation to generate
    embeddings compatible with Graphiti's vector storage.
    """

    def __init__(
        self,
        model,
        tokenizer,
        embedding_dim: int = 1536,
        enable_batching: bool = True,
    ):
        """
        Initialize MLX embedder.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
            embedding_dim: Target dimension for embeddings (MRL truncation)
            enable_batching: Enable request batching for better throughput
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
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

        Process:
        1. Tokenize all inputs and pad to the same length
        2. Create an attention mask to identify padding
        3. Forward pass through the model for the entire batch
        4. Perform last-token pooling using the attention mask
        5. Normalize all embedding vectors
        6. Truncate all embeddings to the target dimension (MRL)
        7. Convert to a list of lists and return

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
            # Right-padding
            padded = tokens + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)

        # Convert to MLX array: [batch_size, seq_len]
        batch_tokens = mx.array(padded_tokens)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = batch_tokens != 0

        # Single forward pass for entire batch
        outputs = self.model(batch_tokens)

        # Last-token pooling
        # Get the index of the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(axis=1).astype(mx.int32) - 1
        batch_size = outputs.shape[0]
        # Select the hidden states of the last tokens
        pooled_embeddings = outputs[mx.arange(batch_size), sequence_lengths]

        # Normalize the embeddings
        norm = mx.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
        normalized_embeddings = pooled_embeddings / norm

        # Truncate embeddings to the target dimension
        truncated_embeddings = normalized_embeddings[:, : self.embedding_dim]

        # Convert each embedding to list
        return [emb.tolist() for emb in truncated_embeddings]

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
