"""Unit tests for :mod:`rag_observe.observer`."""

from __future__ import annotations

import asyncio
import logging

import pytest
from rag_core import protocols
from rag_observe import ObservabilityHub

pytestmark = pytest.mark.unit


def test_implements_observer_protocol() -> None:
    hub = ObservabilityHub()
    assert isinstance(hub, protocols.Observer)


def test_record_with_zero_listeners_does_not_raise() -> None:
    hub = ObservabilityHub()
    hub.record("started", {"stage": "retrieval"})


def test_record_default_attributes_does_not_raise() -> None:
    hub = ObservabilityHub()
    hub.record("started")


async def test_subscribed_async_listeners_receive_events() -> None:
    hub = ObservabilityHub()
    received: list[tuple[str, dict | None]] = []

    async def listener(event: str, attributes: dict | None) -> None:
        received.append((event, attributes))

    hub.subscribe(listener)
    await hub.emit("done", {"n": 1})
    assert received == [("done", {"n": 1})]


async def test_failing_listener_does_not_break_emit() -> None:
    hub = ObservabilityHub()
    received: list[str] = []

    async def bad(event: str, attributes: dict | None) -> None:
        raise RuntimeError("boom")

    async def good(event: str, attributes: dict | None) -> None:
        received.append(event)

    hub.subscribe(bad)
    hub.subscribe(good)
    await hub.emit("evt", {})
    assert received == ["evt"]


async def test_failing_listener_does_not_break_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    hub = ObservabilityHub()
    received: list[str] = []

    async def bad(event: str, attributes: dict | None) -> None:
        raise RuntimeError("boom")

    async def good(event: str, attributes: dict | None) -> None:
        received.append(event)

    hub.subscribe(bad)
    hub.subscribe(good)
    hub.record("evt", {})  # sync, must not raise
    await asyncio.sleep(0.05)  # let the scheduled listener task run
    assert received == ["evt"]


def test_subscribe_returns_unsubscribe_callable() -> None:
    hub = ObservabilityHub()

    async def listener(event: str, attributes: dict | None) -> None:
        pass

    unsub = hub.subscribe(listener)
    assert callable(unsub)
    assert len(hub.listeners) == 1
    unsub()
    assert len(hub.listeners) == 0


def test_subscribe_deduplicates() -> None:
    hub = ObservabilityHub()

    async def listener(event: str, attributes: dict | None) -> None:
        pass

    hub.subscribe(listener)
    hub.subscribe(listener)
    assert len(hub.listeners) == 1
