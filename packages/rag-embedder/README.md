# rag-embedder
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

End-to-end transformation from parsed documents to indexed representations: chunking
strategies, model-native tokenization, embedding selection (FastEmbed default),
dense + sparse embeddings, batching with backpressure, an explicit indexer abstraction
over duck-typed vector stores, and incremental reindexing driven by content/model/chunker
configuration hashes.

**Version:** 0.1.0 &mdash; **Python:** >=3.12 &mdash; **License:** MIT

---

## Overview

`rag-embedder` occupies the **ingestion-to-indexing spine** of the rag-aio pipeline.
It takes a canonical `rag_core.Document` (already parsed by `rag-doc-handler`) and
turns it into vectors persisted in a vector store. Concretely it owns four concerns:

1. **Tokenization** &mdash; model-aligned token counting and encoding, with a
   dependency-free fallback when a HuggingFace tokenizer cannot be fetched.
2. **Chunking** &mdash; six pluggable strategies (`recursive` is the default) that
   split documents into semantically meaningful, provenance-rich `Chunk` objects.
3. **Embedding** &mdash; dense and sparse embedding strategies backed by `fastembed`
   in production, with deterministic `MockEmbedder` alternatives for tests and CI.
4. **Indexing orchestration** &mdash; the `EmbeddingPipeline` wires chunking, embedding
   and indexing together and decides whether a document needs re-indexing.

The package depends only on `rag-core`, `fastembed`, `tokenizers`, and `numpy`. It does
**not** import `rag-db-handler`: vector stores are accepted via duck typing against the
`rag_core.protocols.VectorStore` protocol, so any store (Qdrant, FAISS, an in-memory
test double) can be dropped in without an import edge.

## Installation

```bash
uv add rag-embedder
```

The `sentence-transformers` extra is available for downstream consumers that want a
local cross-encoder reranker, but `rag-embedder` itself never requires it:

```bash
uv add 'rag-embedder[sentence-transformers]'
```

Within the monorepo, the package is installed as a workspace member alongside
`rag-core`; no extra steps are needed.

### Dependencies

| Dependency | Version | Scope | Notes |
|---|---|---|---|
| `rag-core` | workspace | required | Domain models and Protocol contracts |
| `fastembed` | >=0.3 | required | Production dense + sparse embedding |
| `tokenizers` | >=0.19 | required | HuggingFace tokenizer bindings |
| `numpy` | >=1.26 | required | Vector numerics |
| `sentence-transformers` | >=3.0 | optional | Reserved for reranker extras |

## Architecture / Design Principles

- **Zero new dependencies beyond the declared set.** No LangChain / LlamaIndex coupling;
  all vendor types are hidden behind `rag_core` protocols (see
  [ADR-0002](https://github.com/omkumar01/rag-aio/blob/main/adr/ADR-0002-protocol-based-contracts.md)).
- **No `rag-db-handler` import.** The package talks to vector stores through a
  duck-typed `rag_core.protocols.VectorStore` plus the local `SparseCapableStore`
  protocol. Any store implementing `upsert` / `search` works; sparse upserts are used
  when `isinstance(store, SparseCapableStore)` is true, otherwise silently skipped
  with a debug log.
- **Off the event loop.** Every I/O- or CPU-bound operation runs in a worker thread via
  `asyncio.to_thread`, so `async def` callers are never blocked.
- **Lazy model loading.** `fastembed` models are loaded on first use
  (`ensure_model()`), never at construction time &mdash; the constructor performs no
  download, which keeps tests and dry runs fast and offline-friendly.
- **Deterministic, stable IDs.** Chunks receive deterministic ids derived from
  `(document_id, chunker, chunker_version, index, text)` via `rag_core.stable_id`.
  The same input always yields the same chunk id, enabling incremental indexing.
- **Composable registry.** Chunkers are selected by a `strategy` string from the
  `CHUNKERS` registry; third-party chunkers can register an entry and gain
  `create_chunker` support for free.
- **Injectable factories.** `FastEmbedDense` and `CrossEncoderReranker` (via this
  package's downstream) accept a `model_factory` callable so tests can inject a fake
  model without touching the network.

### Module layout

```
src/rag_embedder/
├── __init__.py              # re-exports the public surface
├── tokenization.py          # HFTokenizer, SimpleTokenizer
├── chunking/
│   ├── __init__.py
│   ├── base.py              # ChunkerConfig, BaseChunker, build_page_map
│   ├── recursive.py         # RecursiveChunker (default)
│   ├── fixed.py             # FixedChunker
│   ├── sentence.py          # SentenceChunker
│   ├── token.py             # TokenChunker
│   ├── structural.py        # StructuralChunker
│   ├── parent_child.py      # ParentChildChunker
│   └── registry.py          # CHUNKERS, create_chunker
├── embedding.py             # FastEmbedDense/Sparse, Mock*, embed_many
├── indexing.py              # IndexerConfig, SparseCapableStore, VectorIndexer,
│                            # IngestionIndexer, IndexOutcome
└── pipeline.py              # EmbeddingPipeline, PipelineOutcome
```

## Public API

All public names are re-exported from the top-level `rag_embedder` package.

### Tokenization

```python
from rag_embedder import HFTokenizer, SimpleTokenizer
```

| Class | Description |
|---|---|
| `HFTokenizer` | Wraps a `tokenizers.Tokenizer`. Accepts `model_name_or_path` (tries `Tokenizer.from_pretrained`, falls back to `SimpleTokenizer` if the download fails) or an already-built `tokenizer` for offline use. Two class-methods: `from_identifier(model_name)` (may download) and `from_tokenizer(tokenizer, model_name)` (zero-download, preferred by tests). |
| `SimpleTokenizer` | Pure-Python whitespace + regex word splitter with a lazily-built vocabulary. `encode`/`decode` round-trip correctly for any text seen by the instance. |

Both implement the `rag_core.protocols.Tokenizer` protocol (`count_tokens`,
`encode`, `decode`).

### Chunking

```python
from rag_embedder import ChunkerConfig, create_chunker, build_page_map, BaseChunker
```

`create_chunker(config, tokenizer)` returns a chunker implementing the
`rag_core.protocols.Chunker` protocol. All chunkers offload work to threads and
produce `rag_core.Chunk` objects with deterministic ids.

| Strategy | Class | Notes |
|---|---|---|
| `recursive` (default) | `RecursiveChunker` | LangChain-style recursive separator hierarchy `["\n\n\n", "\n\n", "\n", ". ", " ", ""]`, token-budgeted merge with token-based overlap. |
| `fixed` | `FixedChunker` | Character-window chunking with configurable overlap. |
| `sentence` | `SentenceChunker` | Splits on `.!?` boundaries, packs sentences into token-budgeted chunks with sentence-level overlap. |
| `token` | `TokenChunker` | Token-aware sliding window: encode -> slide by `chunk_size - overlap` -> decode. |
| `structural` | `StructuralChunker` | Markdown (`#`/`##`/`###`) and HTML (`<h1>`-`<h6>`) heading-aware splitting with `section_path` metadata. Overflows a section into token-budgeted sub-spans. |
| `parent_child` | `ParentChildChunker` | Large parent chunks (4x `chunk_size`) + smaller child chunks with `parent_chunk_id` references. Parents materialised first. |

`ChunkerConfig` fields:

| Field | Default | Meaning |
|---|---|---|
| `strategy` | `"recursive"` | Selects the chunker via the `CHUNKERS` registry. |
| `chunk_size` | `512` | Target chunk token budget. |
| `overlap` | `64` | Tokens (or sentences, for `sentence`) to overlap between adjacent chunks. |
| `min_chunk_size` | `32` | Minimum token count to keep a span; smaller spans are merged into the previous one. |
| `respect_boundaries` | `True` | Reserved for downstream boundary-aware strategies; the recursive merger honours span edges. |

`build_page_map(document)` returns `(start_char, end_char, page_number)` ranges used
to infer `page_numbers` on each chunk.

Each chunk carries full `ChunkMetadata`: `chunker` name/version, `page_numbers`
(inferred from `build_page_map`), `char_start`/`char_end`, `token_count`,
`section_path`, `parent_chunk_id` (when applicable), `document_id`, `document_hash`,
and an extensible `custom` dict.

### Embedding

```python
from rag_embedder import (
    FastEmbedDense,
    FastEmbedSparse,
    MockEmbedder,
    MockSparseEmbedder,
    embed_many,
)
```

| Class | Description |
|---|---|
| `FastEmbedDense` | Dense embedder backed by `fastembed.TextEmbedding` (default model `BAAI/bge-small-en-v1.5`). Lazy model load via `ensure_model()`. Accepts a `model_factory` callable for testing. L2-normalizes output when `normalize=True` (default). |
| `FastEmbedSparse` | Sparse embedder backed by `fastembed.SparseTextEmbedding` (`Qdrant/bm25` default). Returns `SparseVector` objects with sorted index/value arrays. |
| `MockEmbedder` | Deterministic hash-based dense embedder (SHA-256 -> normalized vector). No downloads. |
| `MockSparseEmbedder` | Deterministic term-hash sparse embedder (term-frequency values). No downloads. |

```python
# Production dense embedding
dense = FastEmbedDense(model_name="BAAI/bge-small-en-v1.5", normalize=True)

# Mock embedder for offline tests / examples
mock = MockEmbedder(dim=384, normalize=True)

# Batched, semaphore-bounded embedding of many texts
vectors = await embed_many(texts, dense, batch_size=64, max_concurrency=4)
```

`embed_many(texts, embedder, batch_size=64, max_concurrency=4)` partitions `texts`
into batches and bounds concurrency with an `asyncio.Semaphore`.

### Indexing

```python
from rag_embedder import (
    IndexerConfig,
    SparseCapableStore,
    VectorIndexer,
    IngestionIndexer,
    IndexOutcome,
)
```

- **`VectorIndexer(store, config, metadata_provider)`** &mdash; implements the
  `rag_core.protocols.Indexer` protocol. Accepts any duck-typed `VectorStore`. Batches
  upserts with a semaphore. When the store satisfies `SparseCapableStore` (has
  `upsert_sparse`), sparse vectors are upserted alongside dense; otherwise they are
  skipped with a debug log.
- **`IngestionIndexer(config)`** &mdash; orchestrates chunking -> embedding ->
  indexing of one document's chunks. `index_document(document, chunks, embedder,
  sparse_embedder, store)` returns an `IndexOutcome` (`chunks_indexed`, `dims`,
  `model`, `sparse_model`, `took_ms`).
- **`IndexerConfig`** &mdash; `batch_size` (default 128), `max_concurrency`
  (default 4), `namespace` (default `None`).

### Pipeline

```python
from rag_embedder import EmbeddingPipeline, PipelineOutcome
```

`EmbeddingPipeline(chunker, embedder, sparse_embedder, indexer, config, store)`
exposes `process(document, store=None) -> PipelineOutcome` which runs the full
chunk -> embed -> sparse -> index flow and returns counts + timing.

```python
outcome = await pipeline.process(document, store=vector_store)
print(outcome.chunks, outcome.chunks_indexed, outcome.dims, outcome.took_ms)
```

`EmbeddingPipeline.should_reindex(document, config_hash_str, seen)` &mdash; a pure
static helper. It returns `False` when `document.id` already maps to
`"content_hash:config_hash"` in the `seen` dict, enabling incremental skips; otherwise
it records the key and returns `True`.

`PipelineOutcome` fields: `chunks`, `chunks_indexed`, `dims`, `model`,
`sparse_model`, `took_ms`, `reindexed`.

## Usage Guides

### Beginner: index a single document (mock embedder, in-memory store)

```python
import asyncio
from rag_core.documents import Document
from rag_embedder import (
    ChunkerConfig,
    HFTokenizer,
    MockEmbedder,
    MockSparseEmbedder,
    EmbeddingPipeline,
    create_chunker,
)


# A real store is any duck-typed VectorStore; here we use a trivial in-memory one.
class MemoryStore:
    def __init__(self):
        self.vectors = {}
        self.sparse = {}

    async def upsert(self, chunk_id, vector, payload, namespace=None):
        self.vectors[chunk_id] = vector
        self.payload = payload

    async def upsert_sparse(self, chunk_id, indices, values, namespace=None):
        self.sparse[chunk_id] = (indices, values)

    async def search(self, vector, top_k, filters=None, namespace=None):
        return []

    async def delete(self, chunk_ids, namespace=None):
        pass

    async def health(self):
        return True


async def main():
    doc = Document(source_uri="test://doc", text="Your document text here ...")
    pipeline = EmbeddingPipeline(
        chunker=create_chunker(ChunkerConfig(strategy="recursive"), HFTokenizer()),
        embedder=MockEmbedder(dim=384),
        sparse_embedder=MockSparseEmbedder(),
        store=MemoryStore(),
    )
    outcome = await pipeline.process(doc)
    print(outcome)


asyncio.run(main())
```

### Intermediate: choose a chunking strategy for Markdown

```python
from rag_core.documents import Document
from rag_embedder import ChunkerConfig, HFTokenizer, create_chunker

text = "# Title\n\n## Section A\nbody body body\n\n## Section B\nmore body"
doc = Document(source_uri="test://md", text=text)

# Structural chunker preserves heading hierarchy in section_path metadata.
cfg = ChunkerConfig(strategy="structural", chunk_size=128, overlap=16, min_chunk_size=8)
chunker = create_chunker(cfg, HFTokenizer())
chunks = await chunker.chunk(doc)
for c in chunks:
    print(c.metadata.section_path, len(c.text))
```

To swap strategies, change only the `strategy` string and (optionally) the config:

```python
# Token-aware sliding window — best when you must match a fixed model context.
cfg = ChunkerConfig(strategy="token", chunk_size=256, overlap=32)
# Parent/child — parents for rerank context, children for retrieval.
cfg = ChunkerConfig(strategy="parent_child", chunk_size=128)
```

### Advanced: incremental reindexing with the orchestrator pattern

```python
from rag_core.ids import config_hash
from rag_embedder import ChunkerConfig, EmbeddingPipeline, IndexerConfig

chunks_cfg = ChunkerConfig(strategy="recursive", chunk_size=512, overlap=64)
indexer_cfg = IndexerConfig(namespace="tenant-42")

pipeline = EmbeddingPipeline(chunker, embedder, sparse, store=store, config=indexer_cfg)
config_hash_str = config_hash(
    {"chunker": chunks_cfg.model_dump(), "indexer": indexer_cfg.model_dump()}
)

seen: dict[str, str] = load_seen_map()  # persisted across runs
for doc in documents:
    if not EmbeddingPipeline.should_reindex(doc, config_hash_str, seen):
        continue  # content + config unchanged -> skip embedding/ingesting
    outcome = await pipeline.process(doc)
    save_seen(seen)  # checkpoint
```

A change to `ChunkerConfig` or `IndexerConfig` alters `config_hash_str`, so every
document is automatically re-indexed only when configuration or content changes.

### Advanced: custom metadata payloads on index

```python
from rag_embedder import VectorIndexer, IndexerConfig

provider = lambda chunk_id: {"source": "api", "tenant": "acme"}
indexer = VectorIndexer(store, IndexerConfig(namespace="acme"), metadata_provider=provider)
await indexer.index(embeddings, sparse_embeddings)
```

## Configuration

All config objects are `rag_core.RagBaseModel` (Pydantic v2, `extra="forbid"`) and
serialize to JSON for the orchestrator's config-hash machinery.

| Object | Fields | Defaults |
|---|---|---|
| `ChunkerConfig` | `strategy`, `chunk_size`, `overlap`, `min_chunk_size`, `respect_boundaries` | see table above |
| `IndexerConfig` | `batch_size`, `max_concurrency`, `namespace` | 128, 4, `None` |
| `EmbeddingPipeline` | wraps a `ChunkerConfig`-driven chunker + `IndexerConfig` | constructed from parts |

`EmbeddingPipeline.config_hash()` hashes the pipeline's `IndexerConfig` (JSON mode)
and is intended for use with `should_reindex`.

## Testing

All tests use `MockEmbedder`/`MockSparseEmbedder` or injected `model_factory`
callables &mdash; **no network or model downloads**. Run the suite:

```bash
uv run pytest packages/rag-embedder -q
```

`FastEmbedDense` tests never hit the network because the constructor is lazy: they
inject a `_FakeDenseModel` via `model_factory`. The `conftest.py` provides shared
fixtures including a `FakeStore` that is both a `VectorStore` and a
`SparseCapableStore`, a locally-built `HFTokenizer` (no download), and `small_doc` /
`multi_page_doc` documents for page-map assertions.

## Dependencies

See `pyproject.toml`. Runtime: `rag-core`, `fastembed`, `tokenizers`, `numpy`.
Development (workspace dev group): `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`,
`mypy`, `httpx`. The package **does not** depend on `rag-db-handler`; any
duck-typed `VectorStore` satisfies the indexing contract.

## Cross-Package Relationships

```
rag-doc-handler ──produces──> Document  ──consumed by──> rag-embedder
                                          (chunking -> embedding -> indexing)
                               VectorStore (rag-db-handler)  <-  rag-embedder
                                     (duck-typed; Qdrant / in-memory)
rag-retrieval ──consumes──> VectorStore (search) + Embedder/SparseEmbedder
rag-rerank  ──consumes──> RetrievalHit (second-stage scoring)
rag-orchestrator ──wires──> EmbeddingPipeline + retrieve + rerank
```

- **rag-core** &mdash; owns `Document`, `Chunk`, `ChunkMetadata`, `Embedding`,
  `SparseEmbedding`, `SparseVector`, the `Tokenizer`, `Chunker`, `Embedder`,
  `SparseEmbedder`, `Indexer`, `VectorStore` protocols, and `stable_id` /
  `config_hash`. `rag-embedder` implements these protocols; it does not re-import
  `rag-db-handler`.
- **rag-db-handler** &mdash; provides the production `QdrantVectorStore` (and the
  `create_vector_store` factory) that satisfies the duck-typed `VectorStore` this
  package writes to. `QdrantVectorStore.upsert_sparse` additionally satisfies
  `SparseCapableStore`, so sparse vectors are persisted when the store supports them.
- **rag-orchestrator** &mdash; `services.load_local_services()` constructs an
  `EmbeddingPipeline` from a `RecursiveChunker` + `FastEmbedDense` + `FastEmbedSparse`
  and binds it to a local Qdrant store, then wires the same `store`/`embedder` into
  `DenseRetriever` (`rag-retrieval`) and a `HeuristicReranker` pipeline
  (`rag-rerank`). The chunker and embedders produced here are the same objects
  consumed downstream.
- **rag-retrieval / rag-rerank** &mdash; consume `EmbeddingPipeline`-produced chunks
  indirectly via the shared `VectorStore`, and the embedder used at query time is the
  same `FastEmbedDense` instance configured at ingestion.
