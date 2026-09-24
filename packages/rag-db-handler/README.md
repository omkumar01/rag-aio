# rag-db-handler

Unified persistence for rag-aio: SQL-backed document store, SQL/in-memory key/value
stores, and a vector store adapter. Every backend is a thin adapter implementing one
of the runtime-checkable protocols defined by `rag-core` (ADR-0002), so downstream
modules consume an abstraction and backends swap via configuration — never through
code changes.

- **Vector store**: Qdrant in embedded local mode by default (zero-infrastructure,
  ADR-0003) with a server mode behind the same adapter; an in-memory fallback for
  tests.
- **Document store**: async SQLAlchemy (sqlite by default) with a faithful JSON
  `data` column plus denormalized query columns.
- **Key/value store**: async SQLAlchemy (`kv` table) or an in-process dict.
- **Secret handling**: secrets leave configs as *references* only and are resolved
  from the environment at construction / call time.

## Installation

```bash
uv add -e packages/rag-db-handler
```

Install a specific storage backend via an optional extra:

```bash
# PostgreSQL instead of sqlite (asyncpg)
uv add -e packages/rag-db-handler[postgres]

# FAISS vector store instead of Qdrant (faiss-cpu)
uv add -e packages/rag-db-handler[faiss]

# pgvector (PostgreSQL vector extension) instead of Qdrant
uv add -e packages/rag-db-handler[pgvector]
```

## Overview

`rag-db-handler` is the persistence boundary of the rag-aio data plane. It owns
*no* business logic: chunkers embed into vectors through it, retriever queries go
through it, ingestion writes documents and dedup keys through it. The contract is
expressed in three `rag-core` protocols — `VectorStore`, `DocumentStore`, and
`KeyValueStore` — so the rest of the platform never depends on Qdrant, SQLAlchemy,
or sqlite directly.

| Capability     | `rag-core` protocol   | Implementations                            |
| -------------- | ---------------------- | ------------------------------------------ |
| Vector store   | `VectorStore`          | `QdrantVectorStore`, `InMemoryVectorStore` |
| Document store | `DocumentStore`        | `SQLDocumentStore`                         |
| Key/value      | `KeyValueStore`        | `SQLKeyValueStore`, `InMemoryKeyValueStore`|

Every adapter additionally follows the `check_health` / `health()` and `close()`
conventions used by the workspace health endpoint (see [Health](#health)).

## Architecture / Design Principles

- **Protocols over vendors (ADR-0002).** Adapters implement
  `rag_core.protocols.{VectorStore,DocumentStore,KeyValueStore}`. No vendor type
  appears in the public API surface.
- **Local-first, server-upgradeable (ADR-0003).** `create_vector_store` defaults to
  Qdrant embedded local mode. Moving to a Qdrant server is a `mode="server"` switch.
- **Config as the only seam.** `VectorStoreConfig`, `SQLStoreConfig`, and
  `KeyValueStoreConfig` (Pydantic v2 `RagBaseModel`, `extra="forbid"`) are the
  sole inputs to the `create_*` factories.
- **Idempotent identity.** Vector point ids are a deterministic `UUIDv5` of
  `f"{namespace}:{chunk_id}"`, so re-upserting the same chunk is idempotent and
  namespace isolation is structural, not a payload convention.
- **JSON source of truth + query columns.** `Document` rows store the full JSON
  serialization in `data` (lossless round-trip via `rag_core.serde.from_json`);
  the other columns are denormalized query hints.
- **Upsert semantics.** `on_conflict` updates on the primary key (documents) or the
  composite `(namespace, key)` (KV), so `put`/`set` are idempotent upserts.
- **Secrets by reference.** `api_key_ref` names an env var resolved at construction
  time; the resolved value is forwarded to the backend client only and never appears
  in any `repr`/`str` of the store or config.

### Source tree

```
src/rag_db_handler/
├── __init__.py        # public API re-exports
├── config.py          # Pydantic configs + create_* factories + resolve_env_secret
├── health.py          # check_health aggregate probe
├── memory_store.py    # InMemoryVectorStore / InMemoryKeyValueStore
├── qdrant_store.py    # QdrantVectorStore (local + server)
├── serialization.py   # document_to_row_fields / row_to_document
└── sql_store.py       # SQLStore (engine/table base), SQLDocumentStore, SQLKeyValueStore
```

## Public API

```python
from rag_db_handler import (
    # configs
    VectorStoreConfig,
    SQLStoreConfig,
    KeyValueStoreConfig,
    # factories
    create_vector_store,   # -> VectorStore
    create_sql_store,      # -> SQLDocumentStore (DocumentStore)
    create_kv_store,       # -> KeyValueStore
    resolve_env_secret,    # resolve an env-var secret reference
    # vector adapters
    QdrantVectorStore,
    InMemoryVectorStore,
    # sql adapters
    SQLDocumentStore,
    SQLKeyValueStore,
    SQLStore,
    # in-memory kv
    InMemoryKeyValueStore,
    # serialization
    document_to_row_fields,
    row_to_document,
    # health
    check_health,
)
```

### Configs

```python
from rag_db_handler import VectorStoreConfig, SQLStoreConfig, KeyValueStoreConfig

# Vector store: local (embedded path) or server (url + optional api_key_ref)
VectorStoreConfig(
    backend="qdrant",            # Literal["qdrant"] — only backend today
    mode="local",                # Literal["local", "server"] (ADR-0003)
    path="./data/vectors",       # required in local mode
    url="https://qdrant.example.com:6333",  # required in server mode
    api_key_ref="RAG_QDRANT_API_KEY",  # env var name (never stored on instance)
    collection="rag_chunks",     # Qdrant collection name
    vector_size=384,             # ge=1
    distance="cosine",           # Literal["cosine", "euclid", "dot"]
)

# SQL store (sqlite by default)
SQLStoreConfig(
    url="sqlite+aiosqlite:///./data/rag.db",
    pool_size=5,        # ge=0
    max_overflow=10,    # ge=0
    echo=False,
)

# Key/value store
KeyValueStoreConfig(
    backend="sqlite",   # Literal["sqlite", "memory"]
    path=None,          # sqlite path override; falls back to default URL
    namespace="default",
)
```

`VectorStoreConfig.assert_valid()` raises `rag_core.errors.ConfigError` for invalid
mode-specific combinations (e.g. local mode without `path`, server mode without `url`).

### Factories

```python
from rag_db_handler import create_vector_store, create_sql_store, create_kv_store

vec: VectorStore = create_vector_store(VectorStoreConfig(...))
docs: SQLDocumentStore = create_sql_store(SQLStoreConfig(...))      # a DocumentStore
kv: KeyValueStore = create_kv_store(KeyValueStoreConfig(...))
```

`create_vector_store` resolves `api_key_ref` from the environment and fails fast with
`ConfigError` if the referenced variable is unset.

### Vector stores

```python
class QdrantVectorStore(VectorStore):
    def __init__(self, config: VectorStoreConfig) -> None: ...

    async def upsert(chunk_id, vector, payload, namespace=None) -> None: ...
    async def upsert_many(items, namespace=None) -> None: ...   # list[(chunk_id, vector, payload)]
    async def search(vector, top_k, filters=None, namespace=None) -> list[RetrievalHit]: ...
    async def delete(chunk_ids, namespace=None) -> None: ...
    async def count(namespace=None) -> int: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...
    async def ensure_collection(self) -> None: ...               # idempotent
    @property
    def collection(self) -> str: ...

class InMemoryVectorStore(VectorStore):
    def __init__(self, vector_size: int = 384) -> None: ...     # brute-force cosine sim
```

`upsert` derives point ids with `UUIDv5(NAMESPACE_DNS, f"{namespace}:{chunk_id}")`;
cosine scores are normalized to `[0, 1]`. Namespace isolation is a Qdrant `namespace`
filter (and an in-memory guard in the fallback store).

### SQL stores

```python
class SQLDocumentStore(SQLStore, DocumentStore):                 # documents table
    async def init_db(self) -> None: ...
    async def put(self, document: Document) -> None: ...        # on_conflict upsert by id
    async def get(self, document_id: str) -> Document | None: ...
    async def delete(self, document_id: str) -> None: ...
    async def find_by_hash(self, content_hash: str) -> Document | None: ...
    async def count(self) -> int: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...

class SQLKeyValueStore(SQLStore, KeyValueStore):                  # kv table
    def __init__(self, config: SQLStoreConfig, namespace: str = "default") -> None: ...
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes) -> None: ...    # upsert by (namespace, key)
    async def delete(self, key: str) -> None: ...
    async def health(self) -> bool: ...

class SQLStore:                                                  # shared engine/base
    async def init_db(self) -> None: ...                         # idempotent, cached
    async def close(self) -> None: ...
    @property
    def config(self) -> SQLStoreConfig: ...
```

Documents persist a full-JSON `data` column (source of truth for `rag_core.serde`) plus
denormalized query columns (`content_hash`, `source_uri`, `tenant`, `namespace`,
`title`, `mime_type`, `text`, ...). The `kv` table uses the composite primary key
`(namespace, key)` so identical keys in different namespaces never collide.

### In-memory stores

```python
InMemoryVectorStore(vector_size=384)            # cosine similarity, pure Python
InMemoryKeyValueStore(namespace="default")      # dict-backed
```

### Serialization helpers

```python
from rag_db_handler import document_to_row_fields, row_to_document
from rag_core.documents import Document

fields = document_to_row_fields(doc)            # Document -> {id, content_hash, data, ...}
doc = row_to_document({"data": json_str})       # {data: ...} -> Document (validates)
```

### Secrets

```python
from rag_db_handler import resolve_env_secret
from rag_core.errors import ConfigError

key = resolve_env_secret("RAG_QDRANT_API_KEY")  # -> os.environ["RAG_QDRANT_API_KEY"]
# raises ConfigError if unset
```

### Health

```python
from rag_db_handler import check_health

status = await check_health({"vector": vec, "docs": docs, "kv": kv})
# {"vector": True, "docs": True, "kv": True}
```

Stores without a `health()` callable are assumed healthy; any exception during a probe
marks that backend `False` without aborting the whole probe.

## Usage Guides

### Beginner — spin up a local vector store in seconds

Local mode needs nothing but a writable directory:

```python
import asyncio
from rag_db_handler import VectorStoreConfig, create_vector_store

cfg = VectorStoreConfig(mode="local", path="./data/vectors", collection="rag_chunks", vector_size=4)
store = create_vector_store(cfg)

async def main() -> None:
    await store.ensure_collection()
    await store.upsert("chunk-1", [1.0, 0.0, 0.0, 0.0],
                       {"document_id": "doc-1", "text": "alpha"}, namespace="ns")
    hits = await store.search([1.0, 0.0, 0.0, 0.0], top_k=5, namespace="ns")
    print(hits[0].chunk_id, hits[0].text, hits[0].normalized_score)
    print("count:", await store.count("ns"))
    await store.close()

asyncio.run(main())
```

### Beginner — use the in-memory stores

```python
import asyncio
from rag_db_handler import InMemoryVectorStore, InMemoryKeyValueStore

vec = InMemoryVectorStore(vector_size=4)
kv = InMemoryKeyValueStore(namespace="cache")

async def main() -> None:
    await vec.upsert("a", [1.0, 0.0, 0.0, 0.0], {"text": "x"}, "ns")
    await kv.set("token", b"abc123")
    assert await kv.get("token") == b"abc123"
asyncio.run(main())
```

### Intermediate — server-mode Qdrant with env-key resolution

```python
import os
from rag_db_handler import VectorStoreConfig, create_vector_store

os.environ["RAG_QDRANT_API_KEY"] = "..."   # secret lives in the environment only

cfg = VectorStoreConfig(
    mode="server",
    url="https://qdrant.example.com:6333",
    api_key_ref="RAG_QDRANT_API_KEY",     # env var name, resolved at construction
    collection="rag_chunks",
    vector_size=768,
)
store = create_vector_store(cfg)           # fails fast if env var is unset
print(repr(store))                       # QdrantVectorStore(collection='rag_chunks', mode='server')
```

Idempotency: re-upserting `(namespace, chunk_id)` overwrites the point because the
UUIDv5 id is deterministic. Namespace isolation is enforced as a Qdrant filter on
`namespace` and as an in-memory guard in the fallback store.

### Intermediate — SQL document + KV stores

```python
import asyncio
from rag_db_handler import SQLStoreConfig, KeyValueStoreConfig, create_sql_store, create_kv_store
from rag_core.documents import Document, DocumentMetadata

cfg = SQLStoreConfig(url="sqlite+aiosqlite:///./data/rag.db")
docs = create_sql_store(cfg)
kv = create_kv_store(KeyValueStoreConfig(backend="sqlite", namespace="cache"))

async def main() -> None:
    await docs.init_db()
    doc = Document(source_uri="file:///report.pdf", text="hello world",
                   metadata=DocumentMetadata(title="Report"))
    await docs.put(doc)
    assert (await docs.get(doc.id)).text == "hello world"
    assert await docs.find_by_hash(doc.content_hash) is not None
    await kv.set("k", b"v")
    assert await kv.get("k") == b"v"
    await docs.close()

asyncio.run(main())
```

### Advanced — aggregate health probing

```python
from rag_db_handler import check_health

results = await check_health({"vector": vec, "docs": docs, "kv": kv})
# Stores lacking health() are assumed healthy; an exception in one probe marks it
# False without aborting the others.
unhealthy = [name for name, ok in results.items() if not ok]
```

### Advanced — filtering vector search

Scalar and metadata filters translate to Qdrant `MatchValue`/`MatchAny` conditions.
Keys prefixed with `metadata.` traverse the nested payload metadata object; list
values produce a `MatchAny`:

```python
hits = await store.search(
    vector, top_k=10,
    filters={"metadata.lang": "en", "document_id": ["d1", "d2"]},
    namespace="ns",
)
```

## Configuration

### `VectorStoreConfig`

| Field           | Type                              | Default      | Notes                                  |
| --------------- | --------------------------------- | ------------ | -------------------------------------- |
| `backend`       | `Literal["qdrant"]`               | `"qdrant"`   | Only backend today.                    |
| `mode`          | `Literal["local","server"]`       | `"local"`    | ADR-0003.                              |
| `path`          | `str \| None`                     | `None`       | Required in local mode.                |
| `url`           | `str \| None`                     | `None`       | Required in server mode.               |
| `api_key_ref`   | `str \| None`                     | `None`       | Env var name resolved at construction. |
| `collection`    | `str`                             | `"rag_chunks"` | Qdrant collection name.              |
| `vector_size`   | `int` (`ge=1`)                    | `384`        | Must match your embedder dims.         |
| `distance`      | `Literal["cosine","euclid","dot"]` | `"cosine"`  | Maps to Qdrant distance.               |

### `SQLStoreConfig`

| Field          | Type  | Default                      | Notes          |
| -------------- | ----- | ---------------------------- | -------------- |
| `url`          | `str` | `"sqlite+aiosqlite:///./data/rag.db"` | SQLAlchemy async URL. |
| `pool_size`    | `int` | `5`                          | `ge=0`.        |
| `max_overflow` | `int` | `10`                         | `ge=0`.        |
| `echo`         | `bool`| `False`                      | SQL logging.   |

### `KeyValueStoreConfig`

| Field      | Type                              | Default       | Notes                         |
| ---------- | --------------------------------- | ------------- | ----------------------------- |
| `backend`  | `Literal["sqlite","memory"]`      | `"sqlite"`    |                               |
| `path`     | `str \| None`                     | `None`        | sqlite path (else default URL). |
| `namespace`| `str`                             | `"default"`   | KV namespace for `memory`.    |

## Testing

```bash
uv run pytest packages/rag-db-handler -q
```

- **Unit tests** (always run): config validation, serialization round-trips,
  in-memory store behavior.
- **Integration tests** (`pytest.mark.integration`): exercise real Qdrant local
  mode and sqlite through the async stores; deselected by
  `python scripts/check.py --fast`.

A shared test fixture builds a `VectorStore` / `DocumentStore` / `KeyValueStore` and
asserts `isinstance(store, rag_core.protocols.VectorStore)` etc. to verify protocol
conformance.

## Dependencies

`qdrant-client>=1.10`, `sqlalchemy>=2.0`, `aiosqlite>=0.20`, and `rag-core`.
Optional extras declared in `pyproject.toml`:

| Extra      | Packages          | Purpose                          |
| ---------- | ----------------- | -------------------------------- |
| `postgres` | `asyncpg>=0.29`   | PostgreSQL as the SQL backend.   |
| `faiss`    | `faiss-cpu>=1.8`  | Local FAISS vector store.        |
| `pgvector` | `pgvector>=0.3`   | pgvector via PostgreSQL.         |

## Cross-Package Relationships

- **`rag-core`** — owns the `VectorStore`, `DocumentStore`, and `KeyValueStore`
  protocols this package implements, plus `Document` (the row model), `RetrievalHit`
  (the search result), `errors.ConfigError`, and `serde.from_json` (used by
  `row_to_document`). Configs inherit `RagBaseModel` (`extra="forbid"`).
- **`rag-embedder`** — never imported, but its `VectorIndexer` writes into any
  duck-typed `VectorStore` this package provides and may be configured to use
  `create_vector_store`. The embedder treats the store as opaque.
- **`rag-retrieval`** — consumes `VectorStore.search` results (`RetrievalHit`s) for
  dense retrieval; never imports this package.
- **`rag-mass-inject`** — orchestrates ingestion across the document and KV stores
  for bulk jobs; dedup keys are written through `KeyValueStore`.
- **`rag-orchestrator`** — composes the health probe (`check_health`) and the store
  factories into a running pipeline via configuration.
- **`rag-cache` / `rag-observe`** — may wrap these stores for caching and telemetry
  without modifying them.

Service boundary: `rag-db-handler` runs in-process for latency; the
[architecture.md](../../architecture.md) notes that a FastAPI service wrapping these
adapters is the production scale path (ADR-0005), exposing `/health`, `/ready`,
`/metrics`.
