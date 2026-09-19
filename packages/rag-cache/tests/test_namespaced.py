"""Tests for the NamespacedCache wrapper."""

from __future__ import annotations

import pytest
from rag_cache import MemoryCache, NamespacedCache, cache_key


async def test_prefixes_keys() -> None:
    inner = MemoryCache()
    nc = NamespacedCache(inner, "ns", key_hash=False)
    await nc.set("foo", b"bar")
    assert await inner.get("ns:foo") == b"bar"


async def test_key_hash_namespace() -> None:
    inner = MemoryCache()
    nc = NamespacedCache(inner, "ns", key_hash=True)
    await nc.set("foo", b"bar")
    resolved = cache_key("ns", key="foo")
    assert await inner.get(resolved) == b"bar"


async def test_namespaces_isolated() -> None:
    inner = MemoryCache()
    nc_a = NamespacedCache(inner, "a", key_hash=False)
    nc_b = NamespacedCache(inner, "b", key_hash=False)
    await nc_a.set("k", b"1")
    await nc_b.set("k", b"2")
    assert await nc_a.get("k") == b"1"
    assert await nc_b.get("k") == b"2"


async def test_get_or_set_calls_factory_once_async() -> None:
    calls = 0

    async def factory() -> bytes:
        nonlocal calls
        calls += 1
        return b"v"

    nc = NamespacedCache(MemoryCache(), "ns")
    first = await nc.get_or_set("k", factory)
    second = await nc.get_or_set("k", factory)
    assert first == b"v"
    assert second == b"v"
    assert calls == 1


async def test_get_or_set_supports_sync_factory() -> None:
    nc = NamespacedCache(MemoryCache(), "ns", key_hash=False)
    value = await nc.get_or_set("k", lambda: b"sync")
    assert value == b"sync"
    assert await nc.get("k") == b"sync"


async def test_get_or_set_only_caches_successful_results() -> None:
    inner = MemoryCache()
    nc = NamespacedCache(inner, "ns", key_hash=False)
    attempts = 0

    async def factory() -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return b"ok"

    # first attempt raises -> propagates, not cached
    with pytest.raises(RuntimeError):
        await nc.get_or_set("k", factory)
    # second attempt succeeds and is cached
    result = await nc.get_or_set("k", factory)
    assert result == b"ok"
    assert await inner.get("ns:k") == b"ok"
    assert attempts == 2
