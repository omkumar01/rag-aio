# rag-query

Query-intelligence and retrieval-execution layer for the rag-aio platform.

## Overview

`rag-query` turns a raw user query into one or more retrieval-ready query
variants and then fans those variants out across a pool of retrievers
concurrently. It is **async-first**, **modular**, and ships with a
**low-latency fast path**: with the default configuration (every transformation
flag off) the original query is sent straight to retrieval with no LLM call.

### Capabilities

- **Normalization** — unicode NFKC folding, control-character removal, whitespace
  collapse (`normalize_query`).
- **Intent classification** — lightweight heuristics bucket queries into
  `keyword`, `question`, `conversational`, or `navigational` (`classify_query`).
- **Pluggable strategies**, all implementing the `rag_core.protocols.QueryStrategy`
  contract (`async transform(query) -> list[QueryVariant]`):
  - `RewriteStrategy` — resolves conversational references (deterministic
    whitespace pass-through when no LLM is given).
  - `ExpansionStrategy` — deterministic keyword extraction or LLM-generated
    alternative phrasings.
  - `HyDEStrategy` — generates a hypothetical answer passage via an LLM
    (requires an LLM; construction without one raises `ConfigError`).
  - `DecomposeStrategy` — splits multi-part queries on conjunctions, or prompts
    an LLM for a JSON array of sub-questions with a robust fallback parser.
- **Engine** — `QueryEngine` normalizes, classifies, and runs strategies
  concurrently via `asyncio.TaskGroup`; always prepends an `original` variant,
  de-duplicates case-insensitively, and caps at `max_variants`.
- **Parallel retrieval** — `ParallelRetrievalExecutor` runs retrievers across
  the original query and each variant through a bounded `TaskGroup`
  (`asyncio.Semaphore`). One failing `(variant, retriever)` pair is logged and
  skipped; if *every* pair fails, `RetrievalError` is raised.

## LLM handling

LLM-dependent strategies accept a duck-typed generator exposing
`async def generate(request: GenerationRequest) -> GenerationResult`. A
`deterministic_only=True` config (the default) refuses LLM-only strategies such
as HyDE and selects deterministic fallbacks for the rest, so the engine never
makes a network call unless explicitly permitted.

## Usage

```python
from rag_core.queries import Query
from rag_query import QueryConfig, QueryEngine, ParallelRetrievalExecutor

config = QueryConfig(expand=True, decompose=True, deterministic_only=True, max_variants=5)
engine = QueryEngine(config)

result = await engine.process(Query(text="foo and bar baz"))
# result.variants -> [original, expansion, subquery, ...]

retriever = MyRetriever()  # async retrieve(query) -> RetrievalResult
executor = ParallelRetrievalExecutor([retriever], max_concurrency=4)
hits = await executor.execute(result.query, result.variants, top_k=10)
```

## Testing

Tests are offline and use in-process fakes (`FakeLLM`, fakes for retrievers and
strategies defined in `tests/conftest.py`).

```bash
uv run pytest packages/rag-query -q
```
