# rag-rerank

Second-stage relevance optimization for rag-aio, applied to small candidate sets
(reranking is more expensive than first-stage retrieval). Provides batching, score
thresholds, top-N selection, score normalization (`none` / `minmax` / `sigmoid`),
duplicate removal, and preservation of both original retrieval scores and reranker
scores for explainability.

**Version:** 0.1.0 &mdash; **Python:** >=3.12 &mdash; **License:** MIT

---

## Overview

`rag-rerank` sits in the **second stage** of the standard RAG funnel:
*ingest -> retrieve -> **rerank** -> context -> generate*. Where first-stage
retrieval optimizes for recall, reranking optimizes for precision over a handful of
candidates (typically 5-50) before the expensive context window is assembled.

The package offers three concrete rerankers built on a single shared scoring wrapper,
so batching, dedup, normalization, threshold, top-k, and output provenance are
implemented in exactly one place:

- **`HeuristicReranker`** &mdash; zero-dependency Jaccard term-overlap baseline.
  Useful for tests, local defaults, and as a documented fallback.
- **`CrossEncoderReranker`** &mdash; local cross-encoder via
  `sentence-transformers` (optional extra). Lazy model load with a double-checked
  `asyncio.Lock` for concurrent safety.
- **`RemoteReranker`** &mdash; HTTP reranker for OpenAI-compatible providers with
  runtime auto-detection of a native `/rerank` endpoint vs. a scoring-template
  fallback over `/chat/completions`.

Every reranker returns `RetrievalHit` objects tagged `strategy="reranked"` (via
`RerankPipeline`) with the original retrieval signal preserved in `metadata`
(`original_score`, `original_rank`, `original_model`), so evaluation can separate
retrieval quality from reranking quality.

## Installation

```bash
uv add rag-rerank
```

Install the optional `sentence-transformers` extra when you want a local
cross-encoder:

```bash
uv add 'rag-rerank[sentence-transformers]'
```

### Dependencies

| Dependency | Version | Scope | Notes |
|---|---|---|---|
| `rag-core` | workspace | required | `Query`, `RetrievalHit`, `RerankHit`, `RerankError`, `RagBaseModel` |
| `httpx` | >=0.27 | required | `RemoteReranker` HTTP transport |
| `sentence-transformers` | >=3.0 | optional | `CrossEncoderReranker` local model |

## Architecture / Design Principles

- **One scoring pipeline, many backends.** Every concrete reranker delegates to the
  shared `rerank_candidates(query_text, candidates, scorer, config)` wrapper, which
  enforces the `max_candidates` cap, dedup, batched scoring, normalization,
  threshold, sort, and top-k in a single implementation. Adding a new backend is
  just a matter of supplying a `Scorer` callable.
- **`max_candidates` is a hard cap.** Reranking is the most expensive RAG stage, so
  callers must limit the candidate set *before* asking a reranker to score it. Passing
  more than `max_candidates` candidates raises `RerankError`.
- **Original signal preserved.** Each output hit stores `original_score`,
  `original_rank`, and `original_model` in its `metadata`, while the new reranker
  score lives in `score`/`normalized_score`.
- **Secrets by reference, never by value.** `RemoteReranker.api_key_ref` names an
  environment variable resolved at request time; the token never appears in `repr`/
  logs/serialization.
- **Lazy / cached detection.** `RemoteReranker` probes `POST {base_url}/rerank` on the
  first call and caches the detected mode (`"rerank"` or `"chat"`) on the instance, so
  the probe runs exactly once.
- **Thread off the event loop.** `CrossEncoderReranker` runs `model.predict` in a
  worker thread via `asyncio.to_thread`; its model is loaded lazily under an
  `asyncio.Lock` so concurrent rerank calls share a single model instance.

### Module layout

```
src/rag_rerank/
├── __init__.py    # re-exports the public surface
├── base.py        # rerank_candidates wrapper + Scorer type alias
├── config.py      # RerankConfig, NormalizeMode
├── local.py       # HeuristicReranker, CrossEncoderReranker
├── remote.py      # RemoteReranker (endpoint auto-detection)
├── pipeline.py    # RerankPipeline orchestrator
└── explain.py     # to_rerank_hits, explain
```

## Public API

```python
from rag_rerank import (
    RerankConfig,
    NormalizeMode,
    Scorer,
    rerank_candidates,
    HeuristicReranker,
    CrossEncoderReranker,
    RemoteReranker,
    RerankPipeline,
    to_rerank_hits,
    explain,
)
```

### The `Scorer` contract

```python
Scorer = Callable[[str, list[str]], Awaitable[list[float]]]
```

A scorer takes the query text and a list of candidate texts and returns an aligned
list of relevance scores. `rerank_candidates` slices candidates by `batch_size` and
calls the scorer once per batch.

### Shared wrapper: `rerank_candidates`

```python
rerank_candidates(query_text, candidates, scorer, config) -> list[RetrievalHit]
```

Steps, in order:

1. Reject if `len(candidates) > config.max_candidates` (raises `RerankError`).
2. Dedup by `chunk_id` if `config.dedup` (first occurrence wins).
3. Score in batches of `config.batch_size`, validating count alignment.
4. Normalize raw scores via `config.normalize`.
5. Filter out hits below `config.score_threshold` (on the *normalized* score).
6. Sort descending by normalized score.
7. Truncate to `config.top_k`.
8. Reassign 0-based ranks, preserving original `score`/`rank`/`model` in `metadata`
   as `original_score`/`original_rank`/`original_model`.

### Rerankers

| Class | Signature | Backend | Key behavior |
|---|---|---|---|
| `HeuristicReranker` | `(config, keyword_overlap=True)` | none | Jaccard term-overlap baseline; `keyword_overlap=False` returns 0.0 for every candidate. |
| `CrossEncoderReranker` | `(model_name, config, model_factory)` | `sentence-transformers` | Lazy load under `asyncio.Lock`; inject `model_factory` for tests; inference in a worker thread. Missing optional dep raises a helpful `RerankError`. |
| `RemoteReranker` | `(base_url, model, api_key_ref, config, transport)` | `httpx` | Auto-detects `/rerank` vs `/chat/completions`; single probe cached on instance. |

All three rerankers expose a uniform async interface:
`async rerank(query, candidates, top_k=None) -> list[RetrievalHit]`.

### Endpoint auto-detection (`RemoteReranker`)

`RemoteReranker` resolves the open question in [DECISIONS.md](DECISIONS.md) &mdash;
the LM Studio / qwen3-reranker ambiguity. On the first rerank call it probes
`POST {base_url}/rerank` with a Jina/Cohere-style body (`{model, query, documents}`):

- **`200`** -> permanent **rerank mode**: real requests reuse `/rerank` and parse
  `results[i].score`, index-aligned to tolerate re-ordered responses.
- **`404` / `405` / `422`** -> permanent **chat mode**: fall back to
  `POST {base_url}/chat/completions` with a scoring template asking the model to emit
  a JSON array of 0-10 relevance scores; the first `[...]` block is extracted, mapped
  to 0-1, and validated for count. Any parse failure raises `RerankError`.

Detection is cached on the instance (`RemoteReranker.mode`); the probe runs once.

### Pipeline + explainability

- `RerankPipeline(reranker, config)` &mdash; thin orchestrator that delegates scoring
  to the wrapped reranker and tags every output hit with `strategy="reranked"`.
  Per-call `top_k` overrides the config default.
- `to_rerank_hits(before, after) -> list[RerankHit]` &mdash; converts pre/post-rerank
  hit lists into `RerankHit` audit records preserving `original_score`/`original_rank`.
- `explain(hits) -> dict[str, Any]` &mdash; summary with `n_results`, `score_min`,
  `score_max`, `ranks_changed`, and `reranker` model list.

```python
from rag_rerank import RerankPipeline, to_rerank_hits, explain

pipeline = RerankPipeline(HeuristicReranker(RerankConfig()), RerankConfig())
hits = await pipeline.rerank(query, candidates)
audit = to_rerank_hits(candidates, hits)  # list[RerankHit] for evaluation
summary = explain(audit)
print(summary["ranks_changed"], "hits moved relative to first-stage order")
```

## Usage Guides

### Beginner: heuristic reranking of retrieval hits

```python
import asyncio
from rag_core import Query, RetrievalHit
from rag_rerank import (
    RerankConfig, HeuristicReranker, RerankPipeline, to_rerank_hits, explain,
)

# Imagine these came back from DenseRetriever / HybridRetriever.
candidates = [
    RetrievalHit(chunk_id="a", document_id="d", score=0.2, normalized_score=0.2,
                 rank=0, strategy="dense", text="apple banana"),
    RetrievalHit(chunk_id="b", document_id="d", score=0.9, normalized_score=0.9,
                 rank=1, strategy="dense", text="completely unrelated topic"),
    RetrievalHit(chunk_id="c", document_id="d", score=0.5, normalized_score=0.5,
                 rank=2, strategy="dense", text="cherry apple tart"),
]

async def main():
    reranker = HeuristicReranker(RerankConfig(top_k=2, normalize="minmax"))
    pipeline = RerankPipeline(reranker, RerankConfig(max_candidates=100))
    hits = await pipeline.rerank(Query(text="apple pie recipe"), candidates)
    audit = to_rerank_hits(candidates, hits)
    print(explain(audit))
    for h in hits:
        print(h.rank, h.chunk_id, h.score, h.metadata["original_score"])

asyncio.run(main())
```

### Intermediate: local cross-encoder reranker

```bash
uv add 'rag-rerank[sentence-transformers]'
```

```python
from rag_rerank import RerankConfig, CrossEncoderReranker, RerankPipeline

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    config=RerankConfig(top_k=5, batch_size=8, normalize="none"),
)
pipeline = RerankPipeline(reranker, RerankConfig(max_candidates=50))
hits = await pipeline.rerank(query, first_stage_hits)
```

The model is loaded lazily on the first call and shared across concurrent rerank
invocations (thread-safe under an `asyncio.Lock`). For tests, inject a fake model:

```python
class FakeModel:
    def predict(self, pairs):
        return [0.9 if "apple" in text else 0.1 for _q, text in pairs]

reranker = CrossEncoderReranker(
    model_name="fake",
    config=RerankConfig(batch_size=2),
    model_factory=lambda: FakeModel(),
)
```

### Advanced: remote reranker with endpoint auto-detection

```python
from rag_rerank import RemoteReranker, RerankConfig

# api_key_ref names an env var; the secret is resolved at request time.
reranker = RemoteReranker(
    base_url="http://localhost:1234",
    model="qwen3-reranker-47b",
    api_key_ref="LM_STUDIO_API_KEY",
    config=RerankConfig(top_k=10, timeout_s=30.0, batch_size=16, max_candidates=100),
)
hits = await reranker.rerank(query, candidates)
print(reranker.mode)  # "rerank" | "chat" | None
await reranker.aclose()
```

`RemoteReranker` supports dependency injection of an `httpx.AsyncBaseTransport`
(e.g. `httpx.MockTransport`) for testing, so no live server is required in unit tests.

### Advanced: using `rerank_candidates` directly with a custom scorer

Because every reranker delegates to the shared wrapper, you can score with any
async callable that matches the `Scorer` signature:

```python
from rag_rerank import rerank_candidates, RerankConfig

async def my_scorer(query: str, texts: list[str]) -> list[float]:
    # ... call your own relevance model ...
    return [float(sim(query, t)) for t in texts]

hits = await rerank_candidates(
    query.text, candidates, my_scorer, RerankConfig(top_k=5, normalize="sigmoid")
)
```

## Configuration

`RerankConfig` (a `rag_core.RagBaseModel`, `extra="forbid"`):

| Field | Default | Constraints | Meaning |
|---|---|---|---|
| `top_k` | `None` | `ge=1` | Keep at most N candidates (0-based rank reassignment). |
| `score_threshold` | `None` | — | Min reranker score kept (applied to the *normalized* score). |
| `normalize` | `"none"` | — | `"none"`, `"minmax"`, or `"sigmoid"`. |
| `dedup` | `True` | — | Dedupe candidates by `chunk_id` (first occurrence wins). |
| `batch_size` | `16` | `ge=1` | Candidates scored per backend call. |
| `timeout_s` | `60.0` | `gt=0` | HTTP timeout for `RemoteReranker`. |
| `max_candidates` | `100` | `ge=1` | Hard cap; exceeding raises `RerankError`. |

`NormalizeMode = Literal["none", "minmax", "sigmoid"]`.

| Mode | Transform | Use when |
|---|---|---|
| `none` | Raw scores passed through. | Your scorer already returns 0-1. |
| `minmax` | `(s - min) / (max - min)`, collapsing to all-zero when all scores equal. | Scores are on an unknown/absolute scale and you want relative ranking. |
| `sigmoid` | `1 / (1 + exp(-s))`. | Raw scores are logits and you want a 0-1 probability interpretation. |

## Testing

```bash
uv run ruff format packages/rag-rerank && uv run ruff check packages/rag-rerank
uv run mypy packages/rag-rerank/src
uv run pytest packages/rag-rerank -q
```

Tests are network-free:

- `HeuristicReranker` is exercised directly (no deps).
- `CrossEncoderReranker` tests inject a `FakeModel` via `model_factory`; the lazy-load +
  concurrent-safety behavior is verified with `asyncio.gather`.
- `RemoteReranker` tests inject an `httpx.MockTransport` handler, covering rerank-mode
  parsing, the 404->chat-mode fallback, malformed JSON, score-count mismatch, batch
  slicing, and that the API key appears in the `Authorization` header but never in
  `repr`.
- `rerank_candidates` tests cover ordering, original-score preservation, dedup
  (on/off), score threshold, top-k, the `max_candidates` error, minmax/sigmoid
  normalization, and batch slicing.

## Dependencies

Runtime: `rag-core`, `httpx`. Optional: `sentence-transformers`. Development (workspace
dev group): `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `httpx`.

## Cross-Package Relationships

```
rag-retrieval ──produces──> RetrievalHit candidates ──consumed by──> rag-rerank
rag-rerank ──produces──> reranked RetrievalHit (strategy="reranked") ──> rag-context / rag-orchestrator
```

- **rag-core** &mdash; owns the `Reranker` protocol, `Query`, `RetrievalHit`,
  `RerankHit`, `RerankError`, and `RagBaseModel`. `rag-rerank` implements the
  `Reranker` contract and raises `rag_core.RerankError` for all user-facing failures
  (missing optional dep, bad candidate count, malformed HTTP response, missing env
  secret).
- **rag-retrieval** &mdash; the upstream producer of `RetrievalHit` candidates.
  `RerankPipeline` accepts any sequence of hits regardless of which strategy produced
  them, making reranking backend-agnostic.
- **rag-embedder** &mdash; indirectly coupled: the embedder used at retrieval time must
  match the one used at ingestion so the `score`/`normalized_score` fields on
  `RetrievalHit` are meaningful inputs to reranking.
- **rag-orchestrator** &mdash; `services.load_local_services()` wires a
  `HeuristicReranker` behind a `RerankPipeline` as the local-default reranker; switch
  to `CrossEncoderReranker` or `RemoteReranker` via configuration without touching
  the orchestrator.
