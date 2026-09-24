# rag-cache
> Part of the [rag-aio](../../README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Multi-level caching for rag-aio stages (parsing, OCR, chunking, embeddings,
retrieval, reranking, generation). Cache keys embed a content/config hash via
`rag_core.ids.config_hash`, so stale artifacts are never reused. Backends are
async (file/SQLite I/O runs in `asyncio.to_thread`), TTL/LRU/size-bounded, and
share a single `rag_core.protocols.Cache` protocol.

## Overview

`rag-cache` provides a small, uniform caching layer with three pluggable
backends and a set of composable wrappers:

* **Three backends** -- `MemoryCache` (in-process LRU+TTL), `DiskCache`
  (shard filesystem), `SQLiteCache` (WAL-mode database).
* **NamespacedCache** -- a decorator that prefixes every key with a namespace
  and provides `get_or_set` with a sync or async factory.
* **InstrumentedCache** -- a wrapper that records hit/miss/set/delete/
  eviction/error counters into a `CacheStats` dataclass.
* **cache_key** -- deterministic, namespaced, hash-based key construction that
  guarantees semantic equivalence maps to the same key.

All backends satisfy the `rag_core.protocols.Cache` protocol
(`async get`, `async set`, `async delete`) and additionally provide `clear()`
and `size()`. Cache keys are always `bytes`-valued at the protocol boundary;
serialization to bytes is left to the caller (typically via
`rag_core.serde.to_json`).

## Installation

```bash
pip install rag-cache
# or, within the uv workspace:
uv pip install rag-cache
```

Redis backend (declared as an optional extra but **not yet implemented**):

```bash
pip install rag-cache[redis]
# A future adapter will implement rag_core.protocols.Cache behind this extra,
# selected via CacheConfig(backend="redis", ...).
```

```python
from rag_cache import (
    CacheConfig,
    create_cache,  # typed config + factory
    MemoryCache,
    DiskCache,
    SQLiteCache,  # backends
    NamespacedCache,  # namespace prefix + get_or_set
    CacheStats,
    InstrumentedCache,  # stats + wrapper
    cache_key,  # namespace:<sha256>
)
```

## Architecture / Design Principles

### Single protocol, multiple backends

Every backend (and the `InstrumentedCache` wrapper) satisfies the same
`rag_core.protocols.Cache` interface:

```python
class Cache(Protocol):
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
```

This lets you swap backends via configuration with zero application-code changes.

### Content-hash keys prevent stale hits

Cache keys are never raw strings. `cache_key(namespace, **parts)` produces
`"namespace:<sha256>"` where the digest is `config_hash` over the canonicalized
parts. Because the digest includes configuration and content hashes, changing
the embedding model, chunker version, or source text automatically invalidates
the cache -- stale artifacts are structurally impossible to reuse.

### Lazy expiry on every backend

All backends store an expiry deadline per entry and delete expired entries
lazily on `get`. No background sweeper is needed:

* `MemoryCache` stores an absolute `time.monotonic()` deadline per entry.
* `DiskCache` stores an 8-byte big-endian float64 epoch header (`time.time()`);
  `0.0` means "no expiry".
* `SQLiteCache` stores `expires_at REAL` (`NULL` = no expiry).

### Async I/O off the event loop

File and database I/O always runs inside `asyncio.to_thread`, so the event
loop is never blocked. `MemoryCache` has no I/O, so its `async` methods are
trivial no-ops on the concurrency front (they exist only to satisfy the async
protocol contract).

### Disk atomicity

`DiskCache._atomic_write` writes to a temp file and then `os.replace`s it into
place, so readers never observe a partial write. On failure the temp file is
cleaned up.

### SQLite connection strategy

A fresh `sqlite3.connect` is opened and closed per operation inside
`asyncio.to_thread`. Because no connection is shared across threads, SQLite's
cross-thread constraint is satisfied without `check_same_thread=False` and
without a process-wide lock. The database runs in WAL journal mode with a 5-second
`busy_timeout`, so readers never block writers and concurrent writers back off
instead of erroring. This is the simplest, safest option for a single-writer cache.

## Source tree

```
src/rag_cache/
  __init__.py           Public API re-exports; version = "0.1.0"
  config.py             CacheConfig (Pydantic) + create_cache() factory
  keys.py               cache_key() -- deterministic namespaced key builder
  namespaced.py         NamespacedCache (prefix + get_or_set)
  stats.py              CacheStats dataclass + InstrumentedCache wrapper
  backends/
    __init__.py         Module docstring (no exports)
    memory.py           MemoryCache (OrderedDict LRU + TTL)
    disk.py             DiskCache (sharded filesystem + float64 header)
    sqlite.py           SQLiteCache (WAL mode, per-op connections)
```

## Public API

### Key construction (`keys.py`)

#### `cache_key(namespace, **parts) -> str`

Builds a deterministic cache key: `"namespace:<sha256-of-canonical-parts>"`.

The digest is computed with `rag_core.ids.config_hash`, which canonicalizes
`parts` via `json.dumps(sort_keys=True, separators=(",", ":"), default=str)`.
This makes the key:

* **Deterministic** -- identical inputs always produce the same key.
* **Order-insensitive** for dict/parameter parts (key order does not matter).
* **Namespace-scoped** -- the `namespace` is a literal prefix, not folded into
  the hash; two distinct namespaces never collide even with identical parts.

```python
from rag_cache import cache_key

# Simple
key = cache_key("embeddings", text="hello", model="fastembed-BAAI")
# -> "embeddings:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b5a8e0e..."

# Order-insensitive
assert cache_key("ns", a=1, b=2) == cache_key("ns", b=2, a=1)

# Different namespaces never collide
assert cache_key("ns1", a=1) != cache_key("ns2", a=1)

# Pydantic models accepted as parts (canonicalized via model_dump)
assert cache_key("ns", config=CacheConfig(backend="memory")) == cache_key(
    "ns", config=CacheConfig(backend="memory")
)
```

### Configuration (`config.py`)

#### `CacheConfig`

A `RagBaseModel` (strict, `extra="forbid"`) that selects and configures a
backend:

| Field | Default | Type | Description |
|---|---|---|---|
| `backend` | `"memory"` | `Literal["memory", "disk", "sqlite"]` | Which implementation to build |
| `namespace` | `"default"` | `str` | Logical grouping (used with `cache_key`/`NamespacedCache`) |
| `max_items` | `10_000` | `int` | Maximum entries (memory + disk LRU) |
| `max_bytes` | `None` | `int \| None` | Approximate byte budget (memory + disk) |
| `directory` | `None` | `str \| None` | Required when `backend="disk"` |
| `path` | `None` | `str \| None` | Required when `backend="sqlite"` |
| `default_ttl` | `None` | `float \| None` | Default TTL in seconds (none = never expires) |

```python
from rag_cache import CacheConfig

# In-memory for development/testing
mem = CacheConfig(backend="memory", max_items=1000, default_ttl=300)

# Disk-backed for persistence across restarts
disk = CacheConfig(
    backend="disk", directory="/var/cache/rag", max_bytes=1073741824, default_ttl=86400
)

# SQLite for shared multi-process access
db = CacheConfig(backend="sqlite", path="/var/cache/rag/cache.db", default_ttl=3600)
```

#### `create_cache(config: CacheConfig) -> Cache`

Factory that validates backend-specific required fields and returns the
appropriate backend. Raises `rag_core.errors.ConfigError` when a required field
is missing:

```python
from rag_cache import CacheConfig, create_cache

config = CacheConfig(backend="disk", directory="/tmp/rag-cache")
cache = create_cache(config)

# Missing required field -> ConfigError
try:
    bad = CacheConfig(backend="disk")  # no directory
    create_cache(bad)
except ConfigError as e:
    print(e.code)  # config_error
```

### Backends

| Backend | Persistence | Eviction | TTL | Notes |
|---|---|---|---|---|
| `MemoryCache` | in-process | LRU (count + approximate bytes via `len(value)`) | lazy, on `get` | `on_evict` callback hook; no I/O |
| `DiskCache` | filesystem | by size (oldest mtime) when `max_bytes` set | lazy, on `get` | sharded `<hash[:2]>/<sanitized-key>`; 8-byte float64 expiry header |
| `SQLiteCache` | SQLite (WAL) | none | lazy, on `get` | per-operation connection in `to_thread`; `expires_at REAL` column |

#### `MemoryCache`

```python
from rag_cache import MemoryCache

cache = MemoryCache(max_items=500, max_bytes=50_000_000, default_ttl=300)

# LRU eviction: accessing "a" makes "b" the LRU candidate
await cache.set("a", b"1")
await cache.set("b", b"2")
assert await cache.get("a") == b"1"  # "a" is now MRU
await cache.set("c", b"3")  # if max_items=2, "b" is evicted

# on_evict callback
evicted = []
cache = MemoryCache(max_items=1, on_evict=lambda k, v: evicted.append((k, v)))
await cache.set("a", b"1")
await cache.set("b", b"2")  # evicts "a"
assert evicted == [("a", b"1")]

# Additional methods
await cache.clear()
n = await cache.size()
```

#### `DiskCache`

```python
from rag_cache import DiskCache
from pathlib import Path

cache = DiskCache(directory=Path("/tmp/rag-cache"), max_bytes=1_000_000, default_ttl=3600)

# File layout: <directory>/<hash[:2]>/<sanitized-key>
# Entry format: [8-byte float64 expiry][payload bytes]
# 0.0 expiry header = "no expiry"

await cache.set("embeddings:abc123", b"...", ttl=600)
val = await cache.get("embeddings:abc123")

# Clear and size
await cache.clear()  # removes entire directory tree, recreates empty
n = await cache.size()  # counts files (excluding .tmp)
```

#### `SQLiteCache`

```python
from rag_cache import SQLiteCache

cache = SQLiteCache(path="/tmp/rag-cache.db", default_ttl=86400)

# Schema (created at construction):
#   CREATE TABLE IF NOT EXISTS cache (
#     key TEXT PRIMARY KEY,
#     value BLOB NOT NULL,
#     expires_at REAL    -- NULL = no expiry
#   )
# Opened in WAL mode with 5s busy_timeout.

await cache.set("retrieval:q1", b"...", ttl=300)
val = await cache.get("retrieval:q1")
await cache.delete("retrieval:q1")

# SQLiteCache does NOT support max_bytes eviction (no in-process byte limit).
```

### `NamespacedCache`

A decorator that prefixes every key with a namespace. Lets a single backend be
shared across stages/namespaces without the caller remembering to prefix:

```python
from rag_cache import NamespacedCache, MemoryCache

inner = MemoryCache()
ns = NamespacedCache(inner, namespace="embeddings", key_hash=True)

# key_hash=True (default): uses cache_key(namespace, key=<key>)
await ns.set("doc1-chunk3", b"...")
assert await ns.get("doc1-chunk3") == b"..."

# key_hash=False: uses literal "namespace:key"
ns2 = NamespacedCache(inner, namespace="embeddings", key_hash=False)
await ns2.set("doc1-chunk3", b"...")
# resolves to "embeddings:doc1-chunk3" on the inner store

# Namespaces are isolated
ns_a = NamespacedCache(inner, namespace="A", key_hash=False)
ns_b = NamespacedCache(inner, namespace="B", key_hash=False)
await ns_a.set("k", b"1")
await ns_b.set("k", b"2")
assert await ns_a.get("k") == b"1"
assert await ns_b.get("k") == b"2"
```

#### `get_or_set(key, factory) -> bytes`

Resolves the key, returns the cached value when present, and otherwise invokes
`factory` (sync or `async`) -- storing and returning only the successful
result. Exceptions from the factory propagate untouched, leaving the cache
unchanged.

```python
from rag_cache import NamespacedCache, MemoryCache

ns = NamespacedCache(MemoryCache(), "embeddings")

# Async factory
call_count = 0


async def compute_embedding() -> bytes:
    global call_count
    call_count += 1
    return b"dense-vector-bytes"


first = await ns.get_or_set("doc1-chunk3", compute_embedding)
second = await ns.get_or_set("doc1-chunk3", compute_embedding)
assert first == b"dense-vector-bytes"
assert second == b"dense-vector-bytes"
assert call_count == 1  # factory called only once

# Sync factory also works
value = await ns.get_or_set("config-hash", lambda: b"config-bytes")
assert value == b"config-bytes"

# Failed factory is NOT cached -- retried on next call
attempts = 0


async def flaky() -> bytes:
    global attempts
    attempts += 1
    if attempts == 1:
        raise RuntimeError("upstream unavailable")
    return b"ok"


try:
    await ns.get_or_set("k", flaky)  # raises, not cached
except RuntimeError:
    pass
result = await ns.get_or_set("k", flaky)  # succeeds, cached
assert result == b"ok"
assert attempts == 2
```

> **Note:** `NamespacedCache` intentionally does **not** expose `clear()` or
> `size()`. A namespace should not wipe or size the whole shared backend. Callers
> that need those operate on `inner` directly.

### Statistics (`stats.py`)

#### `CacheStats`

A mutable dataclass capturing cache activity:

| Field | Description |
|---|---|
| `hits` | Successful `get` calls |
| `misses` | `get` calls that returned `None` |
| `sets` | Successful `set` calls |
| `deletes` | Successful `delete` calls |
| `evictions` | Entries evicted by LRU/byte limits |
| `errors` | Operations that raised (after incrementing counter) |
| `hit_rate` | `hits / (hits + misses)`, or `0.0` if no lookups yet |

```python
from rag_cache import CacheStats

stats = CacheStats()
# hit_rate defaults to 0.0 (no lookups yet)
assert stats.hit_rate == 0.0
```

#### `InstrumentedCache`

Wraps any `Cache` and records every operation into a `CacheStats`:

```python
from rag_cache import InstrumentedCache, MemoryCache, CacheStats

stats = CacheStats()
cache = InstrumentedCache(MemoryCache(), stats)

await cache.set("k", b"v")
assert await cache.get("k") == b"v"
assert stats.hits == 1
assert stats.sets == 1
assert stats.misses == 0
assert stats.hit_rate == 1.0

# Eviction counting is wired automatically for backends with an on_evict callback:
evictable = InstrumentedCache(MemoryCache(max_items=2), CacheStats())
await evictable.set("a", b"1")
await evictable.set("b", b"2")
await evictable.set("c", b"3")  # evicts "a"
assert evictable.stats.evictions == 1


# Errors are counted and propagated:
class BrokenCache:
    async def get(self, key):
        raise RuntimeError("boom")

    async def set(self, key, value, ttl=None): ...
    async def delete(self, key): ...


ic = InstrumentedCache(BrokenCache())
assert isinstance(ic, Cache)  # still satisfies the protocol
try:
    await ic.get("k")
except RuntimeError:
    pass
assert ic.stats.errors == 1
```

## Usage Guides

### Beginner: simple in-process cache

```python
import asyncio
from rag_cache import MemoryCache

cache = MemoryCache(max_items=1000, default_ttl=300)


async def main():
    await cache.set("hello", b"world")
    val = await cache.get("hello")
    print(val)  # b"world"
    await cache.delete("hello")
    assert await cache.get("hello") is None


asyncio.run(main())
```

### Beginner: cache with a factory pattern

```python
from rag_cache import NamespacedCache, MemoryCache, cache_key

cache = NamespacedCache(MemoryCache(), namespace="embeddings")


async def get_or_compute(doc_id: str) -> bytes:
    key = f"doc:{doc_id}"  # NamespacedCache hashes this via cache_key
    return await cache.get_or_set(key, lambda: expensive_embedding_call(doc_id))


# First call computes; second call returns cached
result = await get_or_compute("doc1")
result_cached = await get_or_compute("doc1")  # no recompute
```

### Intermediate: persistent caching across restarts

```python
from rag_cache import CacheConfig, create_cache, cache_key
from rag_core import to_json
import pickle

# Disk-persisted cache -- survives process restarts
disk_cache = create_cache(
    CacheConfig(
        backend="disk",
        directory="/var/cache/rag/embeddings",
        max_bytes=2_000_000_000,  # 2 GB cap, LRU by mtime
        default_ttl=86400,  # 24 hours
    )
)

# Cache key includes model + content hash so config changes invalidate automatically
key = cache_key("embeddings", model="fastembed-BAAI", text_hash="sha256:...")
await disk_cache.set(key, vector_bytes)

# Later, after restart:
cached = await disk_cache.get(key)
```

### Intermediate: SQLite cache for shared multi-process access

```python
from pathlib import Path
from rag_cache import SQLiteCache

# Multiple processes can safely share a SQLite cache (WAL mode)
cache = SQLiteCache(Path("/shared/rag-cache.db"), default_ttl=3600)

await cache.set("retrieval:result:q1", cached_result_bytes, ttl=300)
result = await cache.get("retrieval:result:q1")
```

### Advanced: instrumented multi-level cache

```python
import asyncio
from rag_cache import (
    MemoryCache,
    DiskCache,
    SQLiteCache,
    NamespacedCache,
    InstrumentedCache,
    CacheStats,
    cache_key,
)

# L1: fast memory, short TTL
l1 = InstrumentedCache(MemoryCache(max_items=5000, default_ttl=60), CacheStats())
# L2: disk persistence, longer TTL
l2 = InstrumentedCache(DiskCache(directory="/var/cache/rag", default_ttl=3600), CacheStats())

l1_ns = NamespacedCache(l1, "embeddings")
l2_ns = NamespacedCache(l2, "embeddings")


async def cached_compute(text: str) -> bytes:
    key = cache_key("embeddings", text=text, model="fastembed-BAAI")

    # Try L1 (memory)
    val = await l1.get(key)
    if val is not None:
        return val

    # Try L2 (disk)
    val = await l2.get(key)
    if val is not None:
        # Promote to L1
        await l1.set(key, val, ttl=60)
        return val

    # Compute and populate both levels
    val = expensive_embedding(text)
    await l1.set(key, val, ttl=60)
    await l2.set(key, val, ttl=3600)
    return val


# Inspect stats after a workload
async def report():
    print(f"L1 hit rate: {l1.stats.hit_rate:.1%}")
    print(f"L2 hit rate: {l2.stats.hit_rate:.1%}")
    print(f"L1 evictions: {l1.stats.evictions}")
```

### Advanced: caching with content-aware invalidation

```python
from rag_cache import cache_key
from rag_core import Document, config_hash

doc = Document(source_uri="s3://...", text="...")

# The cache key changes automatically when content or config changes:
# - doc.content_hash captures source_uri + text
# - config_hash captures the embedding model + chunker config
key = cache_key(
    "embeddings",
    doc_hash=doc.content_hash,
    model="fastembed-BAAI",
    chunker=config_hash({"strategy": "recursive", "chunk_size": 512}),
)
# Same content + same config -> same key (cache hit across runs)
# Different content or config -> different key (cache miss, recompute)
```

## Configuration

Cache configuration is expressed as a `CacheConfig` Pydantic model, which is
validated and consumed by the `create_cache` factory. See the [Public API](#public-api)
section for the full field table.

| Backend | Required fields | Optional fields |
|---|---|---|
| `memory` | none | `max_items`, `max_bytes`, `default_ttl`, `namespace` |
| `disk` | `directory` | `max_items`, `max_bytes`, `default_ttl`, `namespace` |
| `sqlite` | `path` | `default_ttl`, `namespace` |

Backend-specific limitations:

* `SQLiteCache` does not support `max_bytes` eviction (no in-process byte limit).
* `DiskCache` enforces `max_bytes` best-effort by evicting oldest files by mtime.
* `MemoryCache` byte accounting is approximate (uses `len(value)`).

## Testing

```bash
uv run pytest packages/rag-cache -q
```

All tests are fast and require no external services. They use the `unit` marker
(inherited from the workspace pytest configuration). Contract tests are
parametrized over all three backends using `tmp_path`.

| File | Focus |
|---|---|
| `tests/test_smoke.py` | Version string, basic import |
| `tests/test_keys.py` | Key format, determinism, order-insensitivity, namespace isolation, no-parts stability |
| `tests/test_backends.py` | Parametrized contract tests across all backends: Cache protocol conformance, set/get roundtrip, missing key returns None, TTL expiry, delete, overwrite, clear, size |
| `tests/test_memory.py` | LRU eviction with max_items, byte-limit eviction, default TTL, on_evict callback |
| `tests/test_namespaced.py` | Key prefixing (hashed and literal), namespace isolation, get_or_set with async/sync factories, failed factory not cached |
| `tests/test_stats.py` | hit_rate with no lookups, hits/misses/sets/deletes counting, eviction counting via on_evict, error propagation and counting, shared stats property, protocol conformance after instrumentation |

### Writing backend contract tests

The contract in `test_backends.py` can be reused for new backends:

```python
from rag_cache import MemoryCache, DiskCache, SQLiteCache
from rag_core.protocols import Cache
import pytest

BACKENDS = ["memory", "disk", "sqlite"]


@pytest.fixture(params=BACKENDS, ids=BACKENDS)
def cache(request, tmp_path):
    if request.param == "memory":
        return MemoryCache()
    if request.param == "disk":
        return DiskCache(directory=tmp_path / "disk")
    return SQLiteCache(path=tmp_path / "cache.db")


@pytest.mark.parametrize("cache", BACKENDS, indirect=True, ids=BACKENDS)
async def test_set_get_roundtrip(cache):
    await cache.set("k", b"v")
    assert await cache.get("k") == b"v"
```

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| `rag-core` | local | `Cache` protocol, `config_hash` for key construction, `ConfigError` for validation |

Optional extras:

| Extra | Packages | Status |
|---|---|---|
| `redis` | `redis>=5.0` | Declared in `pyproject.toml` but **no Redis backend ships yet**. A future adapter implementing `rag_core.protocols.Cache` can register behind this extra, selected via `CacheConfig(backend="redis", ...)`. |

## Cross-Package Relationships

```
rag-core ──► rag-cache
    │            │
    │            ├── implements protocols.Cache
    │            ├── uses ids.config_hash for cache keys
    │            ├── uses errors.ConfigError for validation
    │            └── uses serde.to_json/from_json (by callers) for value (de)serialization
    │
    └── consumed by: rag-embedder, rag-retrieval, rag-rerank,
                     rag-query, rag-context, rag-generation,
                     rag-orchestrator, rag-mass-inject
```

* `rag-cache` depends on `rag-core` only -- no other `rag-*` package.
* Each downstream package constructs its own `Cache` (via `create_cache` or a
  direct backend) and wraps it with `NamespacedCache` + `InstrumentedCache` as
  needed.
* The `Observer` protocol in `rag-core` is optionally used by consuming packages
  to observe cache hit/miss rates via `ObservabilityHub`.
* Cache keys embed `config_hash` / `content_hash` values computed by `rag-core`,
  so cache invalidation is automatic when models or content change -- no explicit
  cache-busting API is needed.
