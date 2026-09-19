"""``ObservabilityHub``: an :class:`rag_core.protocols.Observer` implementation.

The hub fans stage events out to three sinks:

* a structured log record (info level),
* an OpenTelemetry counter ``rag.event.<event>`` (no-op without an SDK),
* a list of async listener callbacks.

Listeners run with ``asyncio.gather(..., return_exceptions=True)`` so a failing
listener is logged but never breaks the caller. The synchronous ``record``
entry point (matching the ``Observer`` protocol) dispatches listeners on the
running event loop when available; :meth:`emit` is the awaited equivalent that
always runs listeners inline.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .instrumentation import record_counter
from .logging import _LOGRECORD_ATTRS, get_logger

__all__ = ["ObservabilityHub"]

_Listener = Callable[[str, "dict[str, Any] | None"], Awaitable[None]]

_logger = get_logger("rag_observe.hub")


class ObservabilityHub:
    """Implements :class:`rag_core.protocols.Observer`."""

    def __init__(self) -> None:
        self._listeners: list[_Listener] = []
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def listeners(self) -> Sequence[_Listener]:
        return list(self._listeners)

    def subscribe(self, callback: _Listener) -> Callable[[], None]:
        """Register an async listener; returns a callable to unsubscribe it."""

        if callback in self._listeners:
            return lambda: None
        self._listeners.append(callback)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(callback)

        return unsubscribe

    def record(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Sync ``Observer`` entry point: log + metric, then best-effort listeners.

        Mirrors ``rag_core.protocols.Observer.record``. When called from within a
        running event loop, listeners are dispatched on a background task so this
        method never blocks or raises on a listener failure. With no running loop,
        only the log + metric are emitted.
        """
        self._log_and_count(event, attributes)
        if not self._listeners:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._notify(event, attributes))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def emit(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Async entry point: log + metric, then await all listeners.

        Listener failures are captured (``return_exceptions=True``) and logged,
        never propagated.
        """
        self._log_and_count(event, attributes)
        await self._notify(event, attributes)

    def _log_and_count(self, event: str, attributes: dict[str, Any] | None) -> None:
        safe = {k: v for k, v in (attributes or {}).items() if k not in _LOGRECORD_ATTRS}
        _logger.info(event, extra={"fields": {"event": event, **safe}})
        record_counter(f"rag.event.{event}", 1, {"event": event})

    async def _notify(self, event: str, attributes: dict[str, Any] | None) -> None:
        if not self._listeners:
            return
        results = await asyncio.gather(
            *(cb(event, attributes) for cb in self._listeners),
            return_exceptions=True,
        )
        for listener, result in zip(self._listeners, results, strict=False):
            if isinstance(result, BaseException):
                _logger.warning("listener %r failed", listener, exc_info=result)
