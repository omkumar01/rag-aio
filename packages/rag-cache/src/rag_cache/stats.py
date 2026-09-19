"""Cache statistics and an instrumented wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.protocols import Cache


@dataclass
class CacheStats:
    """Mutable counters capturing cache activity.

    ``hit_rate`` is ``hits / (hits + misses)`` and is ``0.0`` before any lookup
    is made (``hits + misses == 0``).
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups > 0 else 0.0


class InstrumentedCache:
    """Wraps a :class:`Cache`, recording every operation into a :class:`CacheStats`.

    On construction, if the wrapped backend exposes a settable ``on_evict``
    callback (currently :class:`~rag_cache.backends.memory.MemoryCache`), it is
    wired to count evictions automatically. Otherwise only hits/misses/sets/
    deletes/errors are recorded. Exceptions from the wrapped cache propagate
    unchanged (after incrementing ``errors``).
    """

    def __init__(self, inner: Cache, stats: CacheStats | None = None) -> None:
        self._inner = inner
        self._stats: CacheStats = stats if stats is not None else CacheStats()
        self._wire_evictions()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def _wire_evictions(self) -> None:
        """Hook eviction counting into backends that expose an ``on_evict`` callback."""
        if hasattr(self._inner, "on_evict") and getattr(self._inner, "on_evict", None) is None:
            self._inner.on_evict = self._on_evict

    def _on_evict(self, _key: str, _value: bytes) -> None:
        self._stats.evictions += 1

    async def get(self, key: str) -> bytes | None:
        try:
            value = await self._inner.get(key)
        except Exception:
            self._stats.errors += 1
            raise
        if value is None:
            self._stats.misses += 1
        else:
            self._stats.hits += 1
        return value

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        try:
            await self._inner.set(key, value, ttl)
        except Exception:
            self._stats.errors += 1
            raise
        self._stats.sets += 1

    async def delete(self, key: str) -> None:
        try:
            await self._inner.delete(key)
        except Exception:
            self._stats.errors += 1
            raise
        self._stats.deletes += 1
