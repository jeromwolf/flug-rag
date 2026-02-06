"""Batch processor for grouping LLM/embedding requests for efficient inference."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Single item in a batch queue."""

    data: Any
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    submitted_at: float = field(default_factory=time.time)


class BatchProcessor:
    """Collects individual requests into batches for efficient processing.

    Useful for batching embedding or LLM inference requests (e.g., vLLM batch mode).

    Usage::

        processor = BatchProcessor(
            process_fn=my_batch_fn,
            batch_size=8,
            max_wait_ms=100,
        )
        result = await processor.submit(single_input)
    """

    def __init__(
        self,
        process_fn: Callable,
        batch_size: int = 8,
        max_wait_ms: int = 100,
        name: str = "batch",
    ):
        """
        Args:
            process_fn: Async callable that takes a list of items and returns a list of results.
                        len(results) must equal len(items).
            batch_size: Maximum number of items per batch.
            max_wait_ms: Maximum time to wait for a full batch before processing.
            name: Name for logging purposes.
        """
        self._process_fn = process_fn
        self._batch_size = batch_size
        self._max_wait_ms = max_wait_ms
        self._name = name
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background batch worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("BatchProcessor '%s' started (batch_size=%d, max_wait=%dms)", self._name, self._batch_size, self._max_wait_ms)

    async def stop(self) -> None:
        """Stop the background batch worker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("BatchProcessor '%s' stopped", self._name)

    async def submit(self, data: Any) -> Any:
        """Submit a single item for batch processing.

        Returns the result for this specific item once the batch is processed.
        """
        if not self._running:
            await self.start()

        item = BatchItem(data=data)
        await self._queue.put(item)
        return await item.future

    async def _worker_loop(self) -> None:
        """Background loop that collects items and processes them in batches."""
        while self._running:
            batch: list[BatchItem] = []

            try:
                # Wait for the first item
                first_item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch.append(first_item)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Collect more items up to batch_size or max_wait_ms
            deadline = time.time() + self._max_wait_ms / 1000.0
            while len(batch) < self._batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Process what we have and exit
                    self._running = False
                    break

            if batch:
                await self._process_batch(batch)

    async def _process_batch(self, batch: list[BatchItem]) -> None:
        """Process a collected batch and resolve individual futures."""
        inputs = [item.data for item in batch]
        try:
            results = await self._process_fn(inputs)

            if len(results) != len(batch):
                raise ValueError(
                    f"Batch function returned {len(results)} results for {len(batch)} inputs"
                )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as exc:
            logger.error("BatchProcessor '%s' batch failed: %s", self._name, exc)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    @property
    def pending_count(self) -> int:
        """Number of items waiting in the queue."""
        return self._queue.qsize()
