"""Tests for CacheStats and the InstrumentedCache wrapper."""

from __future__ import annotations

import pytest
from rag_cache import CacheStats, InstrumentedCache, MemoryCache
from rag_core.protocols import Cache


class _RaisingCache:
    """A minimal Cache that raises on ``get`` (exercises the error counter)."""

    async def get(self, key: str) -> bytes | None:
        raise RuntimeError("boom")

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


async def test_hit_rate_when_no_lookups() -> None:
    assert CacheStats().hit_rate == 0.0


async def test_hits_misses_and_hit_rate() -> None:
    stats = CacheStats()
    ic = InstrumentedCache(MemoryCache(), stats)
    assert await ic.get("miss") is None  # miss
    assert stats.misses == 1
    await ic.set("k", b"v")
    assert await ic.get("k") == b"v"  # hit
    assert stats.hits == 1
    assert stats.sets == 1
    assert stats.hit_rate == pytest.approx(0.5)


async def test_delete_counts() -> None:
    stats = CacheStats()
    ic = InstrumentedCache(MemoryCache(), stats)
    await ic.set("k", b"v")
    await ic.delete("k")
    assert stats.sets == 1
    assert stats.deletes == 1
    assert await ic.get("k") is None


async def test_eviction_counted_via_on_evict() -> None:
    stats = CacheStats()
    inner = MemoryCache(max_items=2)
    ic = InstrumentedCache(inner, stats)
    await ic.set("a", b"1")
    await ic.set("b", b"2")
    await ic.set("c", b"3")  # evicts "a"
    assert stats.evictions == 1


async def test_errors_propagate_and_are_counted() -> None:
    stats = CacheStats()
    ic = InstrumentedCache(_RaisingCache(), stats)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        await ic.get("k")
    assert stats.errors == 1


async def test_stats_property_exposes_shared_stats() -> None:
    stats = CacheStats()
    ic = InstrumentedCache(MemoryCache(), stats)
    assert ic.stats is stats


async def test_protocol_conformance_after_instrumentation() -> None:
    ic = InstrumentedCache(MemoryCache())
    assert isinstance(ic, Cache)
    await ic.set("k", b"v")
    assert await ic.get("k") == b"v"
