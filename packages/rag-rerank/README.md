# rag-rerank

Second-stage relevance optimization for rag-aio, applied to small candidate sets
(reranking is more expensive than first-stage retrieval). Provides batching, score
thresholds, top-N, score normalization (`none` / `minmax` / `sigmoid`), duplicate
removal, and preservation of both original retrieval scores and reranker scores for
explainability.

## Adapters

| Adapter | Dependency | Notes |
| --- | --- | --- |
| `HeuristicReranker` | none | Zero-dependency Jaccard term-overlap baseline; useful for tests and as a documented fallback. |
| `CrossEncoderReranker` | `sentence-transformers` (optional extra) | Local cross-encoder; lazy-loads on first use, inference via `asyncio.to_thread`. |
| `RemoteReranker` | `httpx` | OpenAI-compatible HTTP reranker with runtime endpoint auto-detection. |

Install the optional sentence-transformers extra when needed:

```bash
uv add 'rag-rerank[sentence-transformers]'
```

## Endpoint auto-detection (LM Studio / qwen3-reranker)

`RemoteReranker` resolves the open question in `DECISIONS.md`: it implements
**both** a native `/rerank` adapter **and** a scoring-template fallback, and
auto-detects which one the server exposes.

1. On the first rerank call it probes `POST {base_url}/rerank` with a Jina/Cohere-style
   body (`{model, query, documents}`).
2. `200` -> permanent **rerank mode**: real requests reuse `/rerank` and parse
   `results[i].score` (index-aligned to tolerate re-ordered responses).
3. `404` / `405` / `422` -> permanent **chat mode**: fall back to
   `POST {base_url}/chat/completions` with a scoring template that asks the model to
   emit a JSON array of 0-10 relevance scores; the first `[...]` block is extracted,
   mapped to 0-1, and validated for count. Any parse failure raises `RerankError`.

Detection is cached on the instance (`RemoteReranker.mode`); the probe runs once.

## Usage

```python
from rag_rerank import RerankConfig, HeuristicReranker, RerankPipeline, to_rerank_hits
from rag_core import Query

reranker = HeuristicReranker(RerankConfig(top_k=5, normalize="minmax"))
pipeline = RerankPipeline(reranker, RerankConfig())
hits = await pipeline.rerank(query, candidates)  # list[RetrievalHit], strategy="reranked"
audit = to_rerank_hits(candidates, hits)  # list[RerankHit] for explainability
```

Secrets are never stored by value: `api_key_ref` names an environment variable resolved at
request time, so the token never appears in `repr`/logs.
