"""OpenTelemetry API-based instrumentation for rag-aio stages.

This module wraps the OpenTelemetry **API** only (never the SDK). When no SDK
or exporter is configured, every span and metric call is a no-op, so the helpers
here are safe to use unconditionally from hot paths.

Public API:
    observe(stage, attributes=None)  -- context manager wrapping a stage in a span
    timed(stage, attributes=None)    -- span + duration histogram
    record_counter(name, value, ...) -- increment a counter
    record_histogram(name, value, ...) -- record a histogram value
    StageTimer                      -- a dependency-free wall-clock timer
"""

from __future__ import annotations

import contextlib
import enum
import time
from collections.abc import Iterator
from typing import Any

from opentelemetry import metrics, trace

__all__ = [
    "StageTimer",
    "observe",
    "record_counter",
    "record_histogram",
    "timed",
]

_TRACER = trace.get_tracer("rag-aio")
_METER = metrics.get_meter("rag-aio")

_COUNTERS: dict[str, Any] = {}
_HISTOGRAMS: dict[str, Any] = {}


def _coerce_attr_value(value: Any) -> Any:
    """Coerce an arbitrary Python value into an OpenTelemetry attribute value.

    OpenTelemetry attributes must be primitives or (possibly nested) sequences
    of primitives. Anything else is stringified so a real SDK never rejects the
    call.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return _coerce_attr_value(value.value)
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_attr_value(v) for v in value]
        return [v for v in coerced if v is not None]
    return str(value)


def _coerce_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Drop ``None`` values and coerce the rest to OTel-compatible primitives."""
    if not attributes:
        return {}
    coerced: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        coerced[key] = _coerce_attr_value(value)
    return coerced


def _get_counter(name: str) -> Any:
    counter = _COUNTERS.get(name)
    if counter is None:
        counter = _METER.create_counter(name)
        _COUNTERS[name] = counter
    return counter


def _get_histogram(name: str) -> Any:
    hist = _HISTOGRAMS.get(name)
    if hist is None:
        hist = _METER.create_histogram(name)
        _HISTOGRAMS[name] = hist
    return hist


def record_counter(name: str, value: float, attributes: dict[str, Any] | None = None) -> None:
    """Increment the counter ``name`` by ``value`` (no-op without an SDK)."""
    _get_counter(name).add(value, _coerce_attributes(attributes))


def record_histogram(name: str, value: float, attributes: dict[str, Any] | None = None) -> None:
    """Record ``value`` on the histogram ``name`` (no-op without an SDK)."""
    _get_histogram(name).record(value, _coerce_attributes(attributes))


@contextlib.contextmanager
def observe(stage: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    """Context manager that wraps a stage in an OpenTelemetry span.

    Creates a span named ``rag.<stage>``, records the ``rag.stage`` attribute and
    any additional ``attributes``, measures elapsed time as ``rag.duration_ms``,
    records the exception on failure, and re-raises. A no-op when no SDK is
    configured. Usable in ``async`` code via ``with observe(...): await ...``;
    OTel context propagation stays correct across awaits within the same task.
    """
    attrs = _coerce_attributes(attributes)
    start = time.perf_counter()
    with _TRACER.start_as_current_span(f"rag.{stage}") as span:
        span.set_attribute("rag.stage", stage)
        for key, value in attrs.items():
            span.set_attribute(key, value)
        try:
            yield
        except Exception as exc:
            span.set_attribute("rag.duration_ms", (time.perf_counter() - start) * 1000.0)
            span.record_exception(exc)
            raise
        else:
            span.set_attribute("rag.duration_ms", (time.perf_counter() - start) * 1000.0)


@contextlib.contextmanager
def timed(stage: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    """Context manager that records a span **and** a ``rag.<stage>.duration_ms`` histogram.

    Composes :func:`observe` for the span and :func:`record_histogram` for the
    duration metric, so both are emitted even when the body raises.
    """
    start = time.perf_counter()
    try:
        with observe(stage, attributes):
            yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record_histogram(f"rag.{stage}.duration_ms", elapsed_ms, {"rag.stage": stage})


class StageTimer:
    """A minimal, dependency-free wall-clock timer.

    Intended for modules that need timing outside of a span, e.g. to report
    durations even when no OpenTelemetry SDK is configured.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed_ms: float = 0.0

    def start(self) -> StageTimer:
        self._start = time.perf_counter()
        self._elapsed_ms = 0.0
        return self

    def stop(self) -> float:
        if self._start is None:
            return self._elapsed_ms
        elapsed = (time.perf_counter() - self._start) * 1000.0
        self._elapsed_ms = elapsed
        self._start = None
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        if self._start is not None:
            return (time.perf_counter() - self._start) * 1000.0
        return self._elapsed_ms
