"""Contract tests shared across all cache backends.

Every backend (Memory, Disk, SQLite) must satisfy the same behavioural
contract: the rag-core Cache protocol plus ``clear``/``size``.
"""

from __future__ import annotations

import asyncio

import pytest
from rag_cache import DiskCache, MemoryCache, SQLiteCache
from rag_core.protocols import Cache

Backend = MemoryCache | DiskCache | SQLiteCache
BACKEND_IDS = ["memory", "disk", "sqlite"]


@pytest.fixture(params=BACKEND_IDS, ids=BACKEND_IDS)
def cache(request: pytest.FixtureRequest) -> Backend:
    kind = request.param
    if kind == "memory":
        return MemoryCache()
    if kind == "disk":
        return DiskCache(directory=request.getfixturevalue("tmp_path") / "disk")
    return SQLiteCache(path=request.getfixturevalue("tmp_path") / "cache.db")


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_satisfies_cache_protocol(cache: Backend) -> None:
    assert isinstance(cache, Cache)


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_set_get_roundtrip(cache: Backend) -> None:
    await cache.set("k", b"v")
    assert await cache.get("k") == b"v"


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_get_missing_returns_none(cache: Backend) -> None:
    assert await cache.get("missing") is None


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_ttl_expiry(cache: Backend) -> None:
    await cache.set("k", b"v", ttl=0.05)
    assert await cache.get("k") == b"v"
    await asyncio.sleep(0.08)
    assert await cache.get("k") is None


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_delete(cache: Backend) -> None:
    await cache.set("k", b"v")
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_overwrite(cache: Backend) -> None:
    await cache.set("k", b"1")
    await cache.set("k", b"2")
    assert await cache.get("k") == b"2"


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_clear(cache: Backend) -> None:
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    assert await cache.size() == 2
    await cache.clear()
    assert await cache.size() == 0
    assert await cache.get("a") is None


@pytest.mark.parametrize("cache", BACKEND_IDS, indirect=True, ids=BACKEND_IDS)
async def test_size_counts_entries(cache: Backend) -> None:
    assert await cache.size() == 0
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    assert await cache.size() == 2
