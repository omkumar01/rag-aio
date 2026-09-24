"""Concurrency primitives for batch ingestion.

:class:`StageWorker` provides per-stage semaphores so that, for example, the
"read" stage can run more concurrent file reads than the heavier "embed"
stage.  :class:`BoundedQueue` is a thin async-first wrapper around
:class:`asyncio.Queue` used for producer-consumer dispatching of file paths.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from .config import MassInjectConfig

__all__ = ["BoundedQueue", "StageWorker"]


class StageWorker:
    """Bounded async worker with per-stage concurrency semaphores.

    Each stage name in ``config.concurrency`` gets its own
    :class:`asyncio.Semaphore`.  :meth:`run_stage` acquires the semaphore,
    awaits the supplied coroutine, and releases — providing back-pressure at
    each pipeline stage.

    Stages not present in ``config.concurrency`` run unbounded (semaphore of
    unlimited size), so ad-hoc stages work out of the box.
    """

    def __init__(self, config: MassInjectConfig) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {
            stage: asyncio.Semaphore(n) for stage, n in config.concurrency.items()
        }

    async def run_stage(self, stage: str, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run *coro* while holding the semaphore for *stage*.

        Returns the coroutine's result.  If *stage* has no configured
        semaphore the coroutine runs unbounded.
        """
        sem = self._sems.get(stage)
        if sem is None:
            return await coro
        async with sem:
            return await coro

    def semaphore(self, stage: str) -> asyncio.Semaphore | None:
        """Return the semaphore for *stage* (or ``None`` if unbounded)."""
        return self._sems.get(stage)


class BoundedQueue:
    """Async bounded queue wrapping :class:`asyncio.Queue`.

    Provides a minimal put/get/join API suitable for producer-consumer
    dispatch.  ``maxsize`` bounds the in-flight items, applying back-pressure
    to the producer when workers cannot keep up.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: Any) -> None:
        """Block until *item* can be enqueued."""
        await self._queue.put(item)

    async def get(self) -> Any:
        """Block until an item is available, then return it."""
        return await self._queue.get()

    def task_done(self) -> None:
        """Mark a previously-enqueued item as fully processed."""
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all items have been marked done."""
        await self._queue.join()

    def qsize(self) -> int:
        """Approximate number of items currently in the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return ``True`` if the queue has no items."""
        return self._queue.empty()
