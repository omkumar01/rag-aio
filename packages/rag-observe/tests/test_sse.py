"""Unit tests for :mod:`rag_observe.sse` (SSE progress bridge).

Subscription happens lazily when the async generator body first runs, so every
test starts a consumer task (and yields control) *before* emitting events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import pytest
from rag_observe import ObservabilityHub, sse_stream

pytestmark = pytest.mark.unit


def _payload(frame: str) -> dict[str, Any]:
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    body = frame[len("data: ") : -2]
    return json.loads(body)


def _pull_task(gen: AsyncIterator[str], n: int) -> asyncio.Task[list[str]]:
    """Start a task pulling *n* frames from *gen* (subscribing as a side effect)."""

    async def consume() -> list[str]:
        return [await gen.__anext__() for _ in range(n)]

    return asyncio.create_task(consume())


async def test_frames_from_emit() -> None:
    hub = ObservabilityHub()
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 2)
        await asyncio.sleep(0.01)  # let the consumer subscribe
        await hub.emit("retrieval.done", {"hits": 3})
        await hub.emit("ingest.started", {"uri": "s3://bucket/doc.pdf"})
        frames = await pull
    assert _payload(frames[0]) == {"event": "retrieval.done", "attributes": {"hits": 3}}
    assert _payload(frames[1]) == {
        "event": "ingest.started",
        "attributes": {"uri": "s3://bucket/doc.pdf"},
    }


async def test_frames_from_record_background_dispatch() -> None:
    hub = ObservabilityHub()
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 1)
        await asyncio.sleep(0.01)  # let the consumer subscribe
        hub.record("stage.tick", {"stage": "retrieval"})
        await asyncio.sleep(0.05)  # let the scheduled listener task run
        (frame,) = await pull
    assert _payload(frame) == {"event": "stage.tick", "attributes": {"stage": "retrieval"}}


async def test_record_with_none_attributes_yields_null_attributes() -> None:
    hub = ObservabilityHub()
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 1)
        await asyncio.sleep(0.01)
        await hub.emit("done")
        (frame,) = await pull
    assert _payload(frame) == {"event": "done", "attributes": None}


async def test_event_filter_excludes_events() -> None:
    hub = ObservabilityHub()
    gen = sse_stream(hub, event_filter=lambda name, _attrs: not name.startswith("noise"))
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 1)
        await asyncio.sleep(0.01)
        await hub.emit("noise.ping", {"n": 1})
        await hub.emit("retrieval.done", {"hits": 1})
        (frame,) = await pull
        assert _payload(frame) == {"event": "retrieval.done", "attributes": {"hits": 1}}
        # The noise event never produced a frame.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream.__anext__(), timeout=0.05)


async def test_unsubscribe_on_generator_close() -> None:
    hub = ObservabilityHub()
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 1)
        await asyncio.sleep(0.01)
        assert len(hub.listeners) == 1
        await hub.emit("e1", {})
        await pull
    assert len(hub.listeners) == 0


async def test_unsubscribe_on_break_out_of_async_for() -> None:
    hub = ObservabilityHub()
    received: list[str] = []
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:

        async def consume_two() -> None:
            async for frame in stream:
                received.append(str(_payload(frame)["event"]))
                if len(received) == 2:
                    break

        task = asyncio.create_task(consume_two())
        await asyncio.sleep(0.01)
        await hub.emit("e1", {})
        await hub.emit("e2", {})
        await task
    # aclosing() closes the generator despite the consumer's break.
    assert received == ["e1", "e2"]
    assert len(hub.listeners) == 0


async def test_listener_failure_does_not_break_stream() -> None:
    hub = ObservabilityHub()

    async def bad(event: str, attributes: dict | None) -> None:
        raise RuntimeError("boom")

    hub.subscribe(bad)
    gen = sse_stream(hub)
    async with aclosing(gen) as stream:
        pull = _pull_task(stream, 1)
        await asyncio.sleep(0.01)
        await hub.emit("e1", {"k": "v"})
        (frame,) = await pull
    assert _payload(frame) == {"event": "e1", "attributes": {"k": "v"}}
