"""Unit tests for :mod:`rag_observe.instrumentation`.

These run with **no** OpenTelemetry SDK installed: every span/metric call is a
no-op, so the helpers must never raise.
"""

from __future__ import annotations

import asyncio

import pytest
from rag_observe import StageTimer, observe, record_counter, record_histogram, timed

pytestmark = pytest.mark.unit


def test_observe_reraises_on_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with observe("retrieval", {"rag.test": "v"}):
            raise ValueError("boom")


def test_observe_records_attributes_without_crashing() -> None:
    with observe("parsing", {"model": "x", "n": 3, "ok": True, "nested": {"a": 1}, "tags": ["a"]}):
        pass


def test_observe_supports_async_body() -> None:
    async def _run() -> None:
        with observe("generation"):
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_timed_reraises_on_exception() -> None:
    with pytest.raises(RuntimeError, match="inner"):
        with timed("rerank"):
            raise RuntimeError("inner")


def test_timed_records_duration_histogram() -> None:
    with timed("retrieval"):
        pass


def test_record_counter_and_histogram_without_sdk() -> None:
    record_counter("rag.test.counter", 1, {"stage": "x"})
    record_histogram("rag.test.histogram", 42.0, {"stage": "x"})


async def test_stage_timer_measures_elapsed_time() -> None:
    timer = StageTimer()
    timer.start()
    await asyncio.sleep(0.01)
    assert timer.elapsed_ms > 0
    elapsed = timer.stop()
    assert elapsed > 0
    assert timer.elapsed_ms > 0


async def test_stage_timer_without_start_returns_zero() -> None:
    timer = StageTimer()
    assert timer.elapsed_ms == 0.0
    assert timer.stop() == 0.0
