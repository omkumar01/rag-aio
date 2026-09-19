# rag-retrieval

Dedicated retrieval engine for rag-aio: dense vector retrieval, sparse/BM25
retrieval (`bm25s`), and hybrid retrieval with Reciprocal Rank Fusion and
weighted-score fusion. Designed to be modular, strictly typed, and fully async.

## Design

- **Strategies as retriever classes** — each implements the `rag_core` `Retriever`
  protocol (`async retrieve(query: Query) -> RetrievalResult`):
  - `DenseRetriever` — embeds the query and searches a `VectorStore`.
  - `SparseRetriever` — embeds the query sparsely and searches a
    `SparseSearchStore` (searches via `search_sparse`).
  - `BM25Retriever` — in-process BM25 via `bm25s`; the corpus is built/refreshed
    lazily and all bm25s work runs in a worker thread (`asyncio.to_thread`).
- **Fusion** — `ReciprocalRankFusion` and `WeightedScoreFusion` implement the
  `FusionStrategy` protocol (`fuse(results, top_k, weights=None, dedup=True)`).
  Hits are deduplicated by `chunk_id` and their per-strategy contributions are
  accumulated into `strategy_scores`.
- **Hybrid orchestration** — `HybridRetriever` runs all retrievers concurrently
  via `asyncio.TaskGroup`. A retriever that raises is recorded (failure is
  surfaced in `timings_ms` as `"<strategy>_failed"`) as long as at least one
  strategy succeeds; if every strategy fails, `RetrievalError` is raised.

## Configuration

`RetrievalConfig` (a `rag_core` `RagBaseModel`):

| field               | default            | meaning                                              |
| ------------------- | ------------------ | ---------------------------------------------------- |
| `strategies`        | `["dense"]`        | strategies to run                                     |
| `top_k`             | `10`               | final fused result budget                            |
| `candidate_k`       | `50`               | per-strategy fetch before fusion                     |
| `fusion`            | `"rrf"`            | `"rrf"` or `"weighted"`                              |
| `dense_weight`      | `1.0`              | weight applied to dense contribution                 |
| `sparse_weight`     | `1.0`              | weight applied to sparse/BM25 contribution         |
| `score_threshold`   | `None`             | min raw score kept (dense/sparse)                    |
| `per_source_limits` | `{}`               | per-strategy candidate cap before fusion             |
| `dedup`             | `True`             | dedupe fused hits by `chunk_id`                      |

## Usage

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

config = RetrievalConfig(strategies=["dense", "bm25"], fusion="rrf", top_k=5)
dense = DenseRetriever(store, embedder, config)
bm25 = BM25Retriever(corpus_provider, config)
hy = HybridRetriever([dense, bm25], ReciprocalRankFusion(), config)

result = asyncio.run(hy.retrieve(Query(text="what is ai")))
print(explain_result(result))
attach_text(result, text_lookup_by_chunk_id)
```

## Explainability

`explain_result(result)` returns a JSON-serializable dict with per-hit source
strategy, raw and normalized scores, `strategy_scores` contributions, ranks,
model provenance, and timings. `attach_text(result, text_lookup)` fills
`hit.text` from a callable or mapping for any hit lacking text.

## Testing

```bash
uv run ruff format packages/rag-retrieval && uv run ruff check packages/rag-retrieval
uv run mypy packages/rag-retrieval/src
uv run pytest packages/rag-retrieval -q
```
