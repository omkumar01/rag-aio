"""SSE (Server-Sent Events) progress bridge for :class:`ObservabilityHub`.

A tiny, framework-free async generator that converts hub events into SSE
frames so consumers (e.g. a FastAPI ``StreamingResponse``) can stream live
pipeline progress to browsers.

Frame schema — every yielded string is one SSE frame::

    data: {"event": "<event name>", "attributes": <event attributes or null>}\n\n

The JSON payload always has exactly two keys: ``event`` (the event name as
passed to :meth:`ObservabilityHub.record` / :meth:`ObservabilityHub.emit`) and
``attributes`` (the event's attribute mapping, or ``null`` when none were
given). No framework imports are used; the generator is pure asyncio.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observer import ObservabilityHub

__all__ = ["sse_stream"]


async def sse_stream(
    hub: ObservabilityHub,
    *,
    event_filter: Callable[[str, dict[str, object]], bool] | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for every event observed on *hub*.

    Subscribes an ``asyncio.Queue``-based listener to *hub* and yields one
    ``data: {...}\\n\\n`` frame per event (see the module docstring for the
    frame schema). The listener enqueues with ``put_nowait`` so it never blocks
    the hub's dispatch — safe under both :meth:`ObservabilityHub.emit`
    (listeners awaited inline) and :meth:`ObservabilityHub.record` (listeners
    dispatched on a background task).

    Args:
        hub: The observability hub to bridge.
        event_filter: Optional predicate ``(event_name, attributes) -> bool``;
            events for which it returns ``False`` are skipped (no frame is
            emitted). ``attributes`` is the event's mapping, or ``{}`` when
            the event carried none.

    The stream runs until the consumer closes the generator (``break`` out of
    an ``async for``, ``aclose()``, or client disconnect); the hub listener is
    always unsubscribed in a ``finally`` block, so closing the stream never
    leaks a listener.
    """
    queue: asyncio.Queue[tuple[str, dict[str, object] | None]] = asyncio.Queue()

    async def listener(event: str, attributes: dict[str, object] | None) -> None:
        queue.put_nowait((event, attributes))

    unsubscribe = hub.subscribe(listener)
    try:
        while True:
            event, attributes = await queue.get()
            if event_filter is not None and not event_filter(event, attributes or {}):
                continue
            payload = json.dumps({"event": event, "attributes": attributes})
            yield f"data: {payload}\n\n"
    finally:
        unsubscribe()
