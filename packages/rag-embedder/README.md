# rag-embedder

End-to-end transformation from parsed documents to indexed representations:
chunking strategies, model-native tokenization, embedding selection (FastEmbed
default), dense + sparse embeddings, batching with backpressure, an explicit
indexer abstraction over duck-typed vector stores, and incremental reindexing
driven by content/model/chunker configuration hashes.

## Design

- **No new dependencies beyond `rag-core`, `fastembed`, `tokenizers`, and
  `numpy`** (as declared in `pyproject.toml`).  The code does not import
  `rag-db-handler` — vector stores are accepted via duck typing against the
  `rag_core.protocols.VectorStore` protocol.
- All I/O-bearing and CPU-bound operations run off the event loop via
  `asyncio.to_thread`.
- `fastembed` models are loaded lazily (constructor performs no download).

## API

### Tokenization

```python
from rag_embedder import HFTokenizer, SimpleTokenizer
```

| Class | Description |
|-------|-------------|
| `HFTokenizer` | Wraps a `tokenizers.Tokenizer`.  Accepts `model_name_or_path` (tries `from_pretrained`, falls back to `SimpleTokenizer` if the download fails) or an already-built `tokenizer` for offline use. |
| `SimpleTokenizer` | Pure-Python whitespace + regex word splitter with a lazily-built vocabulary. |

Both implement the `rag_core.protocols.Tokenizer` protocol (`count_tokens`,
`encode`, `decode`).

### Chunking

```python
from rag_embedder import ChunkerConfig, create_chunker, build_page_map
```

`create_chunker(config, tokenizer)` returns a chunker implementing
`rag_core.protocols.Chunker`.  All chunkers offload work to threads and produce
`rag_core.Chunk` objects with deterministic ids.

| Strategy | Class | Notes |
|----------|-------|-------|
| `recursive` (default) | `RecursiveChunker` | LangChain-style recursive separator hierarchy `["\n\n\n", "\n\n", "\n", ". ", " ", ""]`, token-budgeted merge with token-based overlap. |
| `fixed` | `FixedChunker` | Character-window chunking with configurable overlap. |
| `sentence` | `SentenceChunker` | Splits on `.!?` boundaries, packs sentences into token-budgeted chunks with sentence-level overlap. |
| `token` | `TokenChunker` | Token-aware sliding window: encode → slide by `chunk_size - overlap` → decode. |
| `structural` | `StructuralChunker` | Markdown (`#`/`##`/`###`) and HTML (`<h1>`–`<h6>`) heading-aware splitting with `section_path` metadata. |
| `parent_child` | `ParentChildChunker` | Large parent chunks (4× `chunk_size`) + smaller child chunks with `parent_chunk_id` references. Parents returned first. |

`ChunkerConfig` fields: `strategy`, `chunk_size` (default 512), `overlap`
(default 64), `min_chunk_size` (default 32), `respect_boundaries` (default
True).

Each chunk carries full `ChunkMetadata`: `chunker` name/version, `page_numbers`
(inferred from `build_page_map`), `char_start`/`char_end`, `token_count`,
`section_path`, and `parent_chunk_id` (when applicable).

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
|-------|-------------|
| `FastEmbedDense` | Dense embedder backed by `fastembed.TextEmbedding`. Lazy model load via `ensure_model()`. Accepts a `model_factory` callable for testing. |
| `FastEmbedSparse` | Sparse embedder backed by `fastembed.SparseTextEmbedding` (`Qdrant/bm25` default). Returns `SparseVector` objects. |
| `MockEmbedder` | Deterministic hash-based dense embedder (SHA-256 → normalized vector). No downloads. |
| `MockSparseEmbedder` | Deterministic term-hash sparse embedder. No downloads. |

`embed_many(texts, embedder, batch_size, max_concurrency)` — batched embedding
with an `asyncio.Semaphore` bound on concurrency.

### Indexing

```python
from rag_embedder import VectorIndexer, IngestionIndexer, IndexerConfig, SparseCapableStore
```

- `VectorIndexer(store, config, metadata_provider)` — implements the
  `rag_core.protocols.Indexer` protocol.  Accepts any duck-typed
  `VectorStore`.  Batches upserts with a semaphore.  When the store satisfies
  `SparseCapableStore` (has `upsert_sparse`), sparse vectors are upserted
  alongside dense; otherwise they are skipped with a debug log.
- `IngestionIndexer(config)` — orchestrates chunking → embedding → indexing.
  `index_document(document, chunks, embedder, sparse_embedder, store)` returns an
  `IndexOutcome` (`chunks_indexed`, `dims`, `model`, `took_ms`).
- `IndexerConfig` fields: `batch_size` (default 128), `max_concurrency`
  (default 4), `namespace`.

### Pipeline

```python
from rag_embedder import EmbeddingPipeline, PipelineOutcome
```

`EmbeddingPipeline(chunker, embedder, sparse_embedder, indexer, config, store)`
exposes `process(document, store=None) -> PipelineOutcome` which runs the full
chunk → embed → sparse → index flow and returns counts + timing.

`EmbeddingPipeline.should_reindex(document, config_hash, seen)` — pure helper
that returns `False` when `document.id` already maps to
`content_hash:config_hash` in the `seen` dict, enabling incremental skips.

## Testing

All tests use `MockEmbedder`/`MockSparseEmbedder` or injected `model_factory`
callables — no network or model downloads.  Run:

```bash
uv run pytest packages/rag-embedder -q
```
