"""
Request batching for MLX operations.

Collects concurrent requests and processes them together for improved throughput.
Addresses MLX thread-safety while maximizing GPU utilization.
"""

import asyncio
import logging
from typing import Any, Callable, TypeVar
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchRequest:
    """Single request in a batch."""
    args: tuple
    kwargs: dict
    future: asyncio.Future


class RequestBatcher:
    """
    Batches concurrent requests to improve throughput.

    Strategy:
    1. Collect requests arriving within batch_window
    2. Process entire batch together with MLX
    3. Distribute results back to individual futures

    This converts serial processing (due to MLX lock) into batched processing.
    """

    def __init__(
        self,
        batch_fn: Callable,
        batch_window: float = 0.01,  # 10ms collection window
        max_batch_size: int = 32,
    ):
        """
        Initialize request batcher.

        Args:
            batch_fn: Function that processes a batch (receives lists of args)
            batch_window: Time to wait for more requests before processing
            max_batch_size: Maximum requests per batch
        """
        self.batch_fn = batch_fn
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size

        self.pending: deque[BatchRequest] = deque()
        self.lock = asyncio.Lock()
        self.processing = False

    async def submit(self, *args, **kwargs) -> Any:
        """
        Submit a request for batched processing.

        Returns:
            Result for this specific request
        """
        # Create future for this request
        future = asyncio.Future()
        request = BatchRequest(args=args, kwargs=kwargs, future=future)

        async with self.lock:
            self.pending.append(request)

            # If not already processing, start batch collection
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """
        Collect requests for batch_window, then process together.
        """
        # Wait for batch window to collect concurrent requests
        await asyncio.sleep(self.batch_window)

        async with self.lock:
            # Collect batch
            batch = []
            while self.pending and len(batch) < self.max_batch_size:
                batch.append(self.pending.popleft())

            # Reset processing flag if queue empty
            if not self.pending:
                self.processing = False
            else:
                # More requests pending - schedule next batch
                asyncio.create_task(self._process_batch())

        if not batch:
            return

        logger.info(f"[Batcher] Processing batch of {len(batch)} requests")

        try:
            # Extract args for batch processing
            # Assumes all requests have same signature
            batch_args = [req.args for req in batch]
            batch_kwargs = [req.kwargs for req in batch]

            # Process batch together
            results = await self.batch_fn(batch_args, batch_kwargs)

            # Distribute results
            for request, result in zip(batch, results):
                request.future.set_result(result)

        except Exception as e:
            logger.error(f"[Batcher] Batch processing failed: {e}")
            # Propagate error to all waiting futures
            for request in batch:
                request.future.set_exception(e)
