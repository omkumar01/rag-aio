# rag-retrieval
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Dedicated retrieval engine for rag-aio: dense vector retrieval, sparse/BM25 retrieval
(`bm25s`), and hybrid retrieval with Reciprocal Rank Fusion and weighted-score fusion.
Designed to be modular, strictly typed, and fully async.

**Version:** 0.1.0 &mdash; **Python:** >=3.12 &mdash; **License:** MIT

---

## Overview

`rag-retrieval` is the **recall layer** of a RAG system. It answers a single question:
given a `rag_core.Query`, which `RetrievalHit` chunks from the vector store (and/or
in-process BM25 index) are most relevant?

The package implements the `Retriever` and `FusionStrategy` protocols defined in
`rag_core.protocols` and exposes four concrete strategies:

- **Dense** &mdash; embed the query, search a `VectorStore` by vector similarity.
- **Sparse** &mdash; embed the query sparsely, search a `SparseSearchStore`.
- **BM25** &mdash; in-process BM25 via `bm25s`, lazily indexed from a corpus provider.
- **Hybrid** &mdash; run one or more of the above concurrently and fuse the ranked
  tails with Reciprocal Rank Fusion or weighted-score fusion.

All retrievers return a `RetrievalResult` carrying per-hit provenance
(`strategy`, `model`, `score`, `normalized_score`, `strategy_scores`, `rank`,
`filters_applied`) and per-strategy timings, so downstream rerankers and explainability
tools have a full audit trail.

## Installation

```bash
uv add rag-retrieval
```

The optional `faiss` extra pulls in `faiss-cpu` for a local dense index when your
`VectorStore` does not already host one:

```bash
uv add 'rag-retrieval[faiss]'
```

### Dependencies

| Dependency | Version | Scope | Notes |
|---|---|---|---|
| `rag-core` | workspace | required | Protocols + boundary models |
| `rag-db-handler` | workspace | required | Provides `VectorStore` implementations used as retrieval backends |
| `numpy` | >=1.26 | required | Vector numerics |
| `bm25s` | >=0.2 | required | In-process BM25 index |
| `faiss-cpu` | >=1.8 | optional | Local dense index backend |

## Architecture / Design Principles

- **Strategies as retriever classes.** Each retriever implements the
  `rag_core.protocols.Retriever` protocol (`async retrieve(query: Query) ->
  RetrievalResult`). Retrievers are duck-typed on their backend: a `DenseRetriever`
  only needs a `VectorStore` with `search`, and the embedder only needs `async
  embed(texts)`.
- **Concurrency with isolation.** `HybridRetriever` runs every child retriever inside
  an `asyncio.TaskGroup`. A failing retriever is recorded (surfaced in `timings_ms`
  as `"<strategy>_failed"`) as long as at least one strategy succeeds; if *every*
  strategy fails, `RetrievalError` is raised. This means one flaky backend never
  collapses a hybrid run.
- **Fusion as a pluggable strategy.** `FusionStrategy` is a protocol with a `fuse`
  signature. Two implementations ship: `ReciprocalRankFusion` and
  `WeightedScoreFusion`. Either can be dropped into `HybridRetriever` without code
  changes.
- **Explainability by construction.** Every `RetrievalHit` carries `strategy_scores`
  (the per-strategy contribution, e.g. `{"dense": 0.8, "sparse": 0.1}`), `strategy`,
  `model`, and `timings_ms` on the result. `explain_result` serializes all of this for
  observability and evaluation.
- **Lazy BM25 corpus.** `BM25Retriever` builds its `bm25s` index lazily on the first
  `retrieve` call and rebuilds only when the corpus signature (the list of chunk ids)
  changes, so a stable corpus is indexed once.
- **Thread off the event loop.** `bm25s` indexing and searching run in a worker thread
  via `asyncio.to_thread`; vector-store searches are awaited directly (they are already
  async).

### Module layout

```
src/rag_retrieval/
├── __init__.py      # re-exports the public surface
├── config.py        # RetrievalConfig
├── dense.py         # DenseRetriever
├── sparse.py        # SparseRetriever, BM25Retriever, SparseSearchStore
├── fusion.py        # FusionStrategy, ReciprocalRankFusion, WeightedScoreFusion
├── hybrid.py        # HybridRetriever
└── explain.py       # explain_result, attach_text
```

## Public API

```python
from rag_retrieval import (
    RetrievalConfig,
    DenseRetriever,
    SparseRetriever,
    BM25Retriever,
    HybridRetriever,
    ReciprocalRankFusion,
    WeightedScoreFusion,
    SparseSearchStore,
    explain_result,
    attach_text,
)
```

### Retrieval strategies

| Class | Signature | Backend | Contract |
|---|---|---|---|
| `DenseRetriever` | `(store, embedder, config)` | `VectorStore.search` | Duck-typed embedder needs `async embed(texts)`; `model_name` attribute recorded as provenance. |
| `SparseRetriever` | `(store, sparse_embedder, config)` | `SparseSearchStore.search_sparse` | Duck-typed sparse embedder needs `async embed_sparse(texts)` returning `list[SparseVector]`. |
| `BM25Retriever` | `(corpus_provider, config)` | `bm25s` in a worker thread | `corpus_provider` returns `list[tuple[chunk_id, document_id, text]]`. |

Each retriever exposes a class-level `strategy` string (`"dense"`, `"sparse"`, or
`"bm25"`) used as the source label in fused results and as the failed-key prefix in
`timings_ms` (e.g. `"dense_failed"`).

### Fusion strategies

| Class | Formula | Notes |
|---|---|---|
| `ReciprocalRankFusion(k=60.0)` | `score(c) = Σ weight / (k + rank)` | Rank is 0-based within a list. Robust to score-scale differences. |
| `WeightedScoreFusion(normalize=True)` | `score(c) = Σ weight * normalized_score` | Optional min-max normalization per list before weighting. |

Both implement the `FusionStrategy` protocol (`fuse(results, top_k, weights, dedup)`).
Hits are deduplicated by `chunk_id`; per-strategy contributions are accumulated into
`strategy_scores`.

### Explainability

```python
from rag_retrieval import explain_result, attach_text

# explain_result: JSON-serializable per-hit summary with source strategy,
# raw and normalized scores, strategy_scores contributions, ranks, model
# provenance, applied filters, and timings.
audit = explain_result(result)

# attach_text: fill hit.text from a callable or mapping for any hit lacking
# text. Existing text is preserved.
attach_text(result, {hit.chunk_id: hit.text for hit in result.hits if hit.text})
```

### The `SparseSearchStore` protocol

```python
@runtime_checkable
class SparseSearchStore(Protocol):
    async def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]: ...
```

A `SparseSearchStore` is a `VectorStore` that *additionally* supports sparse vector
scoring. `QdrantVectorStore` (from `rag-db-handler`) implements this for learned-sparse
queries. `VectorIndexer` in `rag-embedder` uses the same structural type via
`SparseCapableStore` (with `upsert_sparse`) to write sparse vectors during ingestion.

## Usage Guides

### Beginner: dense retrieval from an in-memory store

```python
import asyncio
from rag_core.queries import Query
from rag_db_handler.memory_store import InMemoryVectorStore
from rag_retrieval import DenseRetriever, RetrievalConfig


async def main():
    store = InMemoryVectorStore(vector_size=384)
    # ... upsert chunks via an embedder ...

    # MockEmbedder is fine for a demo; swap in FastEmbedDense for real vectors.
    from rag_embedder import MockEmbedder

    embedder = MockEmbedder(dim=384)

    retriever = DenseRetriever(store, embedder, RetrievalConfig(top_k=5))
    result = await retriever.retrieve(Query(text="what is ai"))

    for hit in result.hits:
        print(hit.rank, hit.chunk_id, hit.score, hit.text)


asyncio.run(main())
```

### Intermediate: hybrid dense + BM25 with RRF

```python
import asyncio
from rag_core.queries import Query
from rag_db_handler.memory_store import InMemoryVectorStore
from rag_retrieval import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    ReciprocalRankFusion,
    RetrievalConfig,
    explain_result,
    attach_text,
)
from rag_embedder import MockEmbedder


async def main():
    store = InMemoryVectorStore(vector_size=384)
    dense_emb = MockEmbedder(dim=384)
    dense = DenseRetriever(store, dense_emb, RetrievalConfig())

    # corpus_provider returns (chunk_id, document_id, text) tuples.
    corpus = [("c1", "d1", "alpha"), ("c2", "d1", "beta"), ("c3", "d2", "gamma")]
    bm25 = BM25Retriever(lambda: corpus, RetrievalConfig())

    cfg = RetrievalConfig(
        strategies=["dense", "bm25"],
        fusion="rrf",
        top_k=5,
        candidate_k=10,
        dense_weight=1.0,
        sparse_weight=1.0,
    )
    hy = HybridRetriever([dense, bm25], ReciprocalRankFusion(), cfg)

    result = await hy.retrieve(Query(text="alpha gamma"))
    print(explain_result(result)["hits"][0]["strategy_scores"])
    # attach_text backfills hit.text from a chunk->text mapping/callable
    attach_text(result, {h.chunk_id: h.text for h in result.hits if h.text})


asyncio.run(main())
```

### Advanced: weighted fusion with per-source limits and score thresholding

```python
from rag_retrieval import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    WeightedScoreFusion,
    RetrievalConfig,
)

cfg = RetrievalConfig(
    strategies=["dense", "bm25"],
    fusion="weighted",
    top_k=10,
    candidate_k=50,
    dense_weight=0.7,
    sparse_weight=1.3,
    per_source_limits={"dense": 30, "bm25": 80},
    score_threshold=0.2,
    dedup=True,
)
hy = HybridRetriever(
    [DenseRetriever(store, embedder, cfg), BM25Retriever(corpus_provider, cfg)],
    WeightedScoreFusion(normalize=True),
    cfg,
)
result = await hy.retrieve(Query(text="neural search", filters={"metadata.category": "paper"}))
```

### Advanced: failure isolation and timings inspection

When a strategy fails, `HybridRetriever` does not abort; it records the failure:

```python
result = await hy.retrieve(query)
timings = result.timings_ms
# timings may contain entries like:
# {"dense_embed_ms": 1.2, "dense_search_ms": 4.3, "bm25": 12.0, "dense_failed": 1.0, "hybrid": 18.5}
if "bm25_failed" in timings:
    print("BM25 retriever was unavailable; fused from dense only")
```

## Configuration

`RetrievalConfig` (a `rag_core.RagBaseModel`, `extra="forbid"`):

| Field | Default | Meaning |
|---|---|---|
| `strategies` | `["dense"]` | Strategies to run. Recognized names: `"dense"`, `"sparse"`, `"bm25"`. |
| `top_k` | `10` | Final fused result budget. |
| `candidate_k` | `50` | Per-strategy fetch count before fusion. |
| `fusion` | `"rrf"` | `"rrf"` or `"weighted"` &mdash; selects the default fusion algorithm. |
| `dense_weight` | `1.0` | Weight applied to the dense contribution. |
| `sparse_weight` | `1.0` | Weight applied to the sparse / BM25 contribution. |
| `score_threshold` | `None` | Min raw score kept (dense/sparse), applied per-strategy before fusion. |
| `per_source_limits` | `{}` | Per-strategy candidate cap before fusion, e.g. `{"dense": 30, "bm25": 100}`. |
| `dedup` | `True` | Dedupe fused hits by `chunk_id`. |

The `fusion` string selects the default algorithm, but the concrete
`FusionStrategy` object passed to `HybridRetriever` is what actually runs &mdash;
use `"rrf"` with `ReciprocalRankFusion()` or `"weighted"` with
`WeightedScoreFusion(normalize=True)`. Two budgets are independent: `candidate_k`
gates how many each strategy returns before fusion, while `Query.top_k` (default 10)
bounds the final fused output. Per-strategy caps in `per_source_limits` are applied
to each `RetrievalResult` *before* fusion, trimming candidate tails early.

## Testing

```bash
uv run ruff format packages/rag-retrieval && uv run ruff check packages/rag-retrieval
uv run mypy packages/rag-retrieval/src
uv run pytest packages/rag-retrieval -q
```

Unit tests use small deterministic fakes (`_MockEmbedder`, in-memory stores, and
`FakeRetriever`/`_FailingRetriever` doubles for concurrency tests). No live HTTP
server is needed. The fusion tests assert exact RRF arithmetic
(`weight / (k + rank)`) and min-max math so regressions are caught numerically.
`DenseRetriever` tests populate an `InMemoryVectorStore` and assert top-k ordering,
filter passthrough, and namespace scoping.

## Dependencies

Runtime: `rag-core`, `rag-db-handler`, `numpy`, `bm25s`. Optional: `faiss-cpu`.
Development (workspace dev group): `pytest`, `pytest-asyncio`, `ruff`, `mypy`,
`httpx`.

## Cross-Package Relationships

```
rag-embedder ──produces──> chunks + embeddings ──written into──> VectorStore (rag-db-handler)
rag-retrieval ──consumes──> VectorStore (search) + Embedder/SparseEmbedder (rag-embedder)
rag-rerank  ──consumes──> RetrievalHit (second-stage scoring)
rag-orchestrator ──wires──> dense + hybrid retrievers into the query pipeline
```

- **rag-core** &mdash; owns the `Retriever`, `HybridRetriever`, `FusionStrategy`,
  `VectorStore`, `Embedder`, and `SparseEmbedder` protocols, plus the `Query`,
  `RetrievalHit`, and `RetrievalResult` boundary models. `rag-retrieval` implements
  these; it does not define its own data models.
- **rag-db-handler** &mdash; provides the `VectorStore` backends (`QdrantVectorStore`,
  `InMemoryVectorStore`) that `DenseRetriever` searches. The same store that received
  vectors from `rag-embedder` at ingestion time is searched here at query time.
  `SparseSearchStore` is a local protocol for stores that additionally implement
  `search_sparse`; `QdrantVectorStore` supports learned-sparse search.
- **rag-embedder** &mdash; the embedders used at query time (`FastEmbedDense`,
  `FastEmbedSparse`) are the same instances configured at ingestion, ensuring the
  query vector space matches the indexed one.
- **rag-rerank** &mdash; consumes `RetrievalHit` objects produced here and re-ranks the
  top candidates. `HybridRetriever` runs *before* reranking in the standard pipeline
  (retrieve -> rerank -> context -> generate).
- **rag-orchestrator** &mdash; `services.load_local_services()` wires a
  `HybridRetriever` (dense-only by default, with `ReciprocalRankFusion`) and exposes it
  alongside a `RerankPipeline` of the reranked candidates.
