"""In-process LRU + TTL cache backend.

``MemoryCache`` keeps entries in an :class:`~collections.OrderedDict`. Access on
``get`` moves the entry to the most-recently-used end, while insertion beyond the
capacity bounds evicts from the least-recently-used end. Per-entry TTL is
honoured lazily: an expired entry is removed the first time it is read.

There is no I/O here, so the ``async`` methods are trivially non-blocking;
they exist only to satisfy the async :class:`rag_core.protocols.Cache` contract.
Byte accounting is approximate: it uses ``len(value)`` and is only consulted
when ``max_bytes`` is set.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

# Sentinel used to distinguish "no expiry" from a real (always > 0) monotonic time.
_NO_EXPIRY: float | None = None


class _Entry:
    """Stored payload plus its absolute expiry monotonic time (None = never)."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: bytes, expires_at: float | None) -> None:
        self.value = value
        self.expires_at = expires_at


class MemoryCache:
    """Bounded in-process cache with LRU eviction and per-entry TTL.

    Args:
        max_items: Maximum number of entries. When exceeded the least-recently-
            used entry is evicted. ``None``/negative means unbounded by count.
        max_bytes: Approximate byte budget (sum of ``len(value)``). When
            exceeded, least-recently-used entries are evicted. ``None`` disables
            byte-based eviction.
        default_ttl: Default TTL (seconds) applied to entries set without an
            explicit ``ttl``. ``None`` means entries never expire unless a TTL
            is passed to :meth:`set`.
        on_evict: Optional callback ``(key, value)`` invoked for every entry
            evicted by count or byte limits. Wired automatically by
            :class:`~rag_cache.stats.InstrumentedCache` to count evictions.
    """

    def __init__(
        self,
        max_items: int = 10_000,
        max_bytes: int | None = None,
        default_ttl: float | None = None,
        on_evict: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self._max_items = max_items if max_items and max_items > 0 else None
        self._max_bytes = max_bytes if max_bytes and max_bytes > 0 else None
        self._default_ttl = default_ttl
        self._on_evict = on_evict
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes: int = 0

    @property
    def on_evict(self) -> Callable[[str, bytes], None] | None:
        """Eviction callback, settable after construction (e.g. by InstrumentedCache)."""
        return self._on_evict

    @on_evict.setter
    def on_evict(self, value: Callable[[str, bytes], None] | None) -> None:
        self._on_evict = value

    async def get(self, key: str) -> bytes | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
            self._data.pop(key, None)
            self._bytes -= len(entry.value)
            return None
        self._data.move_to_end(key)
        return entry.value

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        if ttl is None:
            ttl = self._default_ttl
        expires_at = time.monotonic() + ttl if ttl is not None else _NO_EXPIRY
        existing = self._data.get(key)
        if existing is not None:
            self._bytes -= len(existing.value)
        self._data[key] = _Entry(value, expires_at)
        self._bytes += len(value)
        self._data.move_to_end(key)
        self._trim()

    async def delete(self, key: str) -> None:
        entry = self._data.pop(key, None)
        if entry is not None:
            self._bytes -= len(entry.value)

    async def clear(self) -> None:
        self._data.clear()
        self._bytes = 0

    async def size(self) -> int:
        return len(self._data)

    def _trim(self) -> None:
        """Evict least-recently-used entries until within the count/byte bounds."""
        while self._max_items is not None and len(self._data) > self._max_items:
            self._pop_oldest()
        while self._max_bytes is not None and self._bytes > self._max_bytes:
            self._pop_oldest()

    def _pop_oldest(self) -> None:
        if not self._data:
            return
        key, entry = self._data.popitem(last=False)
        self._bytes -= len(entry.value)
        cb = self._on_evict
        if cb is not None:
            cb(key, entry.value)
