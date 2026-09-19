"""MemoryCache-specific behaviour: LRU eviction and byte limits."""

from __future__ import annotations

from rag_cache import MemoryCache


async def test_lru_eviction_with_max_items() -> None:
    cache = MemoryCache(max_items=2)
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    # touch "a" so "b" becomes the least-recently-used
    assert await cache.get("a") == b"1"
    await cache.set("c", b"3")  # evicts "b"
    assert await cache.get("b") is None
    assert await cache.get("a") == b"1"
    assert await cache.get("c") == b"3"
    assert await cache.size() == 2


async def test_max_bytes_eviction() -> None:
    # each value is 4 bytes ("AAAA"); cap at 8 bytes -> 2 entries survive
    cache = MemoryCache(max_bytes=8)
    await cache.set("a", b"AAAA")
    await cache.set("b", b"BBBB")
    await cache.set("c", b"CCCC")  # total would be 12, evict oldest
    assert await cache.size() <= 2


async def test_ttl_without_explicit_arg_uses_default() -> None:
    cache = MemoryCache(default_ttl=0.05)
    await cache.set("k", b"v")
    assert await cache.get("k") == b"v"
    import asyncio

    await asyncio.sleep(0.08)
    assert await cache.get("k") is None


async def test_on_evict_callback_invoked() -> None:
    evicted: list[tuple[str, bytes]] = []
    cache = MemoryCache(max_items=1, on_evict=lambda k, v: evicted.append((k, v)))
    await cache.set("a", b"1")
    await cache.set("b", b"2")  # evicts "a"
    assert evicted == [("a", b"1")]
