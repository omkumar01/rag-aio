# rag-db-handler

Capability-specific persistence abstractions and store adapters for rag-aio.
Each capability is its own interface (ADR-0002) so backends swap via
configuration rather than code changes. Local (zero-infrastructure) operation is
the default; a production upgrade path is a config switch, not a code change.

## Public API

```python
from rag_db_handler import (
    # configs
    VectorStoreConfig,
    SQLStoreConfig,
    KeyValueStoreConfig,
    # factories
    create_vector_store,  # -> VectorStore
    create_sql_store,  # -> SQLDocumentStore (DocumentStore)
    create_kv_store,  # -> KeyValueStore
    resolve_env_secret,  # resolve an env-var secret reference
    # adapters
    QdrantVectorStore,  # VectorStore (qdrant)
    InMemoryVectorStore,  # VectorStore (dev/test fallback)
    SQLDocumentStore,  # DocumentStore (SQLAlchemy async)
    SQLKeyValueStore,  # KeyValueStore (SQLAlchemy async)
    InMemoryKeyValueStore,  # KeyValueStore (dev/test fallback)
    SQLStore,  # shared SQL engine/table base
    check_health,  # async aggregate health probe
    # serialization
    document_to_row_fields,
    row_to_document,
)
```

## Stores

| Capability     | Protocol (rag-core)        | Implementations                  |
| -------------- | -------------------------- | -------------------------------- |
| Vector store   | `VectorStore`              | `QdrantVectorStore`, `InMemoryVectorStore` |
| Document store | `DocumentStore`            | `SQLDocumentStore`               |
| Key/value      | `KeyValueStore`            | `SQLKeyValueStore`, `InMemoryKeyValueStore`          |

All `VectorStore` implementations expose the protocol methods
(`upsert`, `search`, `delete`, `health`) plus a `count(namespace)` helper and
an idempotent `ensure_collection()`. `SQLStore` subclasses add `init_db()`,
`close()`, and `count()`.

## Vector store: local vs server mode (ADR-0003)

`QdrantVectorStore` / `create_vector_store` default to **Qdrant local (embedded)
mode** -- a single-process persistent directory, no Docker or external service.

```python
# Local mode (default): persist vectors to a directory on disk.
cfg = VectorStoreConfig(mode="local", path="./data/vectors", collection="rag_chunks")
store = create_vector_store(cfg)

# Server mode: point at a running Qdrant server; the API key is resolved by
# reference from the environment (never stored on the instance/config).
cfg = VectorStoreConfig(
    mode="server",
    url="https://qdrant.example.com:6333",
    api_key_ref="RAG_QDRANT_API_KEY",  # env var name, e.g. export RAG_QDRANT_API_KEY=...
)
store = create_vector_store(cfg)
```

Caveats:
- Local mode is single-process; concurrent writers must funnel through one process.
- Re-upserting the same `(namespace, chunk_id)` is idempotent: point ids are a
  deterministic UUIDv5 of `f"{namespace}:{chunk_id}"`.
- `api_key_ref` is resolved at construction time from `os.environ`; the resolved
  value is forwarded to the `AsyncQdrantClient` only and never appears in any
  `repr`/`str` of the store.

## SQL stores (sqlite default)

```python
cfg = SQLStoreConfig(url="sqlite+aiosqlite:///./data/rag.db")
docs = create_sql_store(cfg)  # DocumentStore
kv = create_kv_store(KeyValueStoreConfig(backend="sqlite", path="./data/rag.kv"))

await docs.init_db()  # idempotent
await docs.put(document)
await docs.get(document_id)
await docs.find_by_hash(content_hash)
await kv.set("k", b"v")
```

Documents are persisted with a full-JSON `data` column (the source of truth for
lossless round-trips via `rag_core.serde.from_json`) plus denormalized query
columns (`content_hash`, `source_uri`, `namespace`, `title`, `mime_type`, ...).
The `kv` table uses a composite primary key `(namespace, key)` so identical keys
in different namespaces never collide.

## Secret handling

Secrets leave configuration objects as references only. `resolve_env_secret(ref)`
looks up `os.environ[ref]` and raises `rag_core.errors.ConfigError` if unset:

```python
key = resolve_env_secret("RAG_QDRANT_API_KEY")  # raises ConfigError if missing
```

## Health

```python
from rag_db_handler import check_health

status = await check_health({"vector": vec_store, "docs": doc_store, "kv": kv_store})
# {"vector": True, "docs": True, "kv": True}
```

Stores without a `health()` callable are assumed healthy; any exception during a
probe marks that backend `False` without aborting the whole probe.

## Testing

```sh
uv run pytest packages/rag-db-handler -q
```

Unit tests run always; `integration`-marked tests exercise real local backends
(Qdrant embedded local mode and sqlite) and are deselected by
`python scripts/check.py --fast`.

## Dependencies

`qdrant-client`, `SQLAlchemy>=2.0`, `aiosqlite`, and `rag-core`. Optional extras
`postgres` (asyncpg), `faiss`, and `pgvector` are declared in `pyproject.toml`
for alternate backends.
