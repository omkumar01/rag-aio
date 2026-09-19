# rag-cache

Multi-level caching for rag-aio stages (parsing, OCR, chunking, embeddings,
retrieval, reranking, generation). Cache keys embed a content/config hash via
`rag_core.ids.config_hash`, so stale artifacts are never reused. Backends are
async (file/sqlite I/O runs in `asyncio.to_thread`), TTL/LRU/size-bounded, and
share a single `rag_core.protocols.Cache` protocol.

## Public API

```python
from rag_cache import (
    CacheConfig,
    create_cache,  # typed config + factory
    MemoryCache,
    DiskCache,
    SQLiteCache,  # backends (async, Cache + clear()/size())
    NamespacedCache,  # namespace prefix + get_or_set(key, factory)
    CacheStats,
    InstrumentedCache,  # hit/miss/eviction/error counters
    cache_key,  # namespace:<sha256-of-parts>
)
```

## Backends

| Backend | Persistence | Eviction | TTL | Notes |
|---|---|---|---|---|
| `MemoryCache` | in-process | LRU (count + approximate bytes via `len(value)`) | lazy, on `get` | `on_evict` callback hook |
| `DiskCache` | filesystem | by size (oldest mtime) when `max_bytes` set | lazy, on `get` | sharded `<hash[:2]>/<sanitized-key>` |
| `SQLiteCache` | SQLite (WAL) | none | lazy, on `get` | per-operation connection in `to_thread` |

- `cache_key(namespace, **parts)` -> `"namespace:<sha256>"`, order-insensitive
  for dict parts (uses `rag_core.ids.config_hash`).
- `CacheConfig` + `create_cache(config)` validate backend-specific fields and
  raise `rag_core.errors.ConfigError` when a required field is missing
  (`directory` for disk, `path` for sqlite).
- `NamespacedCache(inner, namespace, key_hash=True)` prefixes keys (hashed by
  default) and `get_or_set` calls a sync or `async` factory once per miss.
- `InstrumentedCache(inner, stats=...)` records hits/misses/sets/deletes/
  evictions/errors and `stats.hit_rate`.

## Redis (not implemented in this pass)

`redis` is an optional extra in `pyproject.toml`. No Redis backend ships here;
a future adapter can implement `rag_core.protocols.Cache` and register it behind
the extra, selected via `CacheConfig(backend="redis", ...)`.

## Design choices

- **Expiry storage:** `MemoryCache` keeps an absolute monotonic deadline
  (`time.monotonic() + ttl`); `DiskCache` stores an 8-byte big-endian float64
  header (epoch via `time.time()`; `0.0` means "no expiry") so TTLs survive
  restarts; `SQLiteCache` stores `expires_at REAL` (`NULL` = no expiry).
  All backends delete expired entries lazily on `get`.
- **SQLite connection strategy:** a fresh `sqlite3.connect` is opened and closed
  per operation inside `asyncio.to_thread` (WAL mode, `busy_timeout`). Each
  connection lives on one worker thread, satisfying SQLite's cross-thread rule
  without `check_same_thread=False` or a process-wide lock -- the simplest,
  safe option for a single-writer cache.
- **Disk atomicity:** entries are written to a temp file then `os.replace`'d,
  so readers never observe a partial write.

## Testing

`uv run pytest packages/rag-cache -q`. Contract tests are parametrized over
all three backends using `tmp_path`; additional tests cover LRU eviction,
namespacing/`get_or_set`, and instrumentation counters. All tests are fast and
require no external services.
