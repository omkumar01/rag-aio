# rag-query
> Part of the [rag-aio](../../README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Query-intelligence and retrieval-execution layer for the
[rag-aio](https://github.com/omkumar01/rag-aio) platform.

`rag-query` turns a raw user query into one or more retrieval-ready query
variants and then fans those variants out across a pool of retrievers
concurrently. It is **async-first**, **modular**, and ships with a
**low-latency fast path**: with the default configuration (every transformation
flag off) the original query is sent straight to retrieval with no LLM call.

## Overview

Given a single user query, `rag-query` produces a small ensemble of retrieval
queries — the original plus any rewrites, expansions, HyDE passages, and
decomposed subqueries — and runs those forms against a set of retrievers in
parallel. The output is a flat list of :class:`~rag_core.retrieval.RetrievalResult`
objects (one per retriever), each tagged with the query-form id that produced it,
ready for fusion or reranking upstream.

The package is split into two concerns:

1. **Query transformation** — normalization, intent classification, and a set
   of pluggable `QueryStrategy` implementations that each produce
   :class:`~rag_core.queries.QueryVariant` objects.
2. **Parallel retrieval execution** — `ParallelRetrievalExecutor` fans the
   original query and each variant across a pool of retrievers using a bounded
   `asyncio.TaskGroup`.

## Architecture / Design Principles

- **Protocol-based contracts (ADR-0002).** Every strategy implements the
  :class:`~rag_core.protocols.QueryStrategy` contract (`async transform(query)
  -> list[QueryVariant]`); the executor accepts anything implementing the
  :class:`~rag_core.protocols.Retriever` contract
  (`async retrieve(query) -> RetrievalResult`). `rag-core` defines the
  interfaces, this package provides concrete implementations.
- **Async-first / structured concurrency.** Strategies run concurrently inside an
  `asyncio.TaskGroup`; retrieval is fanned out the same way. `TaskGroup`
  semantics mean a failure in one branch cancels sibling coroutines cleanly
  rather than leaking tasks.
- **Bounded concurrency.** `ParallelRetrievalExecutor` wraps every retrieval in an
  `asyncio.Semaphore` (`max_concurrency`, default 4) so a large variant x
  retriever matrix cannot create thousands of simultaneous requests.
- **Low-latency fast path.** With a default `QueryConfig` (all transformation
  flags off), `QueryEngine.process` normalizes, classifies, and returns the
  original variant. No strategy runs and no LLM is ever contacted. Even with
  strategies enabled, the `deterministic_only=True` default guarantees a network
  call is never made unless the caller explicitly opts in.
- **Graceful partial failure.** In the executor, a single failing
  `(variant, retriever)` pair is logged and skipped so partial results still
  return. Only when *every* pair fails is :class:`~rag_core.errors.RetrievalError`
  raised.
- **Untrusted LLM output.** Model-generated text (HyDE answers, decomposition
  arrays, expansions) is always parsed into a structured `QueryVariant` and is
  never re-injected into a prompt, mitigating injection risk.
- **Deterministic fallbacks.** Every strategy that *can* operate without an LLM
  does: `RewriteStrategy` is a whitespace pass-through, `ExpansionStrategy`
  extracts frequent keywords, and `DecomposeStrategy` splits on conjunctions.
  Only `HyDEStrategy` is LLM-mandatory (and construction without one raises
  :class:`~rag_core.errors.ConfigError`).

## Public API

### `QueryConfig`

Tunable flags controlling query transformation and strategy selection.
`deterministic_only` gates every LLM-dependent strategy: when `True` (the
default) any strategy that *requires* an LLM is refused, so the engine never
makes a network call.

| field                  | default | meaning                                                                 |
| ---------------------- | ------- | ----------------------------------------------------------------------- |
| `normalize`            | `True`  | Apply `normalize_query` to fold unicode and collapse whitespace.        |
| `expand`               | `False` | Enable `ExpansionStrategy` (keyword extraction / LLM paraphrase).       |
| `rewrite`              | `False` | Enable `RewriteStrategy` (whitespace pass-through / LLM rewrite).        |
| `hyde`                 | `False` | Enable `HyDEStrategy` (LLM-generated hypothetical answer).             |
| `decompose`            | `False` | Enable `DecomposeStrategy` (conjunction split / LLM subquestions).     |
| `max_variants`         | `3`     | Hard cap on the number of variants returned (incl. `original`).         |
| `expansion_count`      | `2`     | Number of alternative phrasings / keywords for expansion.             |
| `deterministic_only`   | `True`  | Refuse LLM-only strategies (e.g. HyDE); use deterministic fallbacks.     |
| `min_query_length`     | `2`     | Minimum whitespace token count for strategies to run.                    |

### `QueryEngine`

Orchestrates normalization, classification, and concurrent strategy execution.
The result always includes a `kind="original"` variant; remaining variants are
de-duplicated (case-insensitive) and capped at `QueryConfig.max_variants`.

```python
from rag_core.queries import Query
from rag_query import QueryConfig, QueryEngine

config = QueryConfig(expand=True, decompose=True, deterministic_only=True, max_variants=5)
engine = QueryEngine(config)

result = await engine.process(Query(text="foo and bar baz"))
# result.variants -> [original, expansion, subquery, ...]
```

### `QueryResult`

| field          | type                              | meaning                                    |
| -------------- | --------------------------------- | ------------------------------------------ |
| `query`        | `Query`                           | Normalized query with `.variants` attached.|
| `variants`     | `list[QueryVariant]`              | Final ordered, de-duplicated variant list. |
| `query_class`  | `str`                             | `keyword` / `question` / `conversational` / `navigational`. |
| `timings_ms`   | `dict[str, float]`                | `normalize_ms`, `classify_ms`, `process_ms`.|

### `QueryClass`

StrEnum with values `keyword`, `question`, `conversational`, `navigational`.

### `classify_query(text: str) -> QueryClass`

Lightweight heuristic classifier evaluated in priority order:

1. Query ends with `?` **or** starts with a question word (`what`, `why`,
   `how`, `who`, `when`, `where`, `which`, `does`, `is`, `are`, `can`) ->
   `question`.
2. Query contains `show me` / `find me` -> `navigational`.
3. Query has at most three tokens -> `keyword`.
4. Otherwise -> `conversational`.

```python
from rag_query import classify_query

classify_query("what is RAG?")  # QueryClass.question
classify_query("show me quarterly sales")  # QueryClass.navigational
classify_query("rag")  # QueryClass.keyword
classify_query("explain retrieval")  # QueryClass.conversational
```

### `normalize_query(text, lowercase=False) -> str`

Applies unicode NFKC folding, removes C0/C1 control characters, collapses
whitespace runs to a single space, and strips. Optionally lower-cases.

```python
from rag_query import normalize_query

normalize_query("  Hello\x00  World  ")  # "Hello World"
normalize_query("Café", lowercase=True)  # "café"
```

### `ParallelRetrievalExecutor`

Runs one or more retrievers against the original query and each variant through
a bounded `asyncio.TaskGroup` (`asyncio.Semaphore`). One failing pair is logged
and skipped; if *every* pair fails, :class:`~rag_core.errors.RetrievalError` is
raised.

```python
from rag_query import ParallelRetrievalExecutor

executor = ParallelRetrievalExecutor([dense_retriever, bm25_retriever], max_concurrency=8)
results = await executor.execute(result.query, result.variants, top_k=10)
# each RetrievalResult.query_id -> the variant (form) id that produced it
```

### Strategies

All implement :class:`~rag_core.protocols.QueryStrategy`
(`async transform(query) -> list[QueryVariant]`).

| Strategy            | Deterministic (no LLM)             | LLM mode                              |
| ------------------- | ---------------------------------- | ------------------------------------- |
| `RewriteStrategy`   | Whitespace pass-through.           | Prompts for a standalone rewritten query. |
| `ExpansionStrategy` | Frequent-keyword extraction.       | Prompts for `count` alternative phrasings. |
| `DecomposeStrategy` | Splits on `and then` / `and` / `;`.  | Prompts for a JSON array of sub-questions with a line-based fallback parser. |
| `HyDEStrategy`      | N/A — **LLM required.**            | Prompts for a hypothetical answer passage. |

```python
from rag_query import HyDEStrategy
from my_llm import MyAsyncGenerator  # async generate(request) -> GenerationResult

hyde = HyDEStrategy(llm=MyAsyncGenerator())
variants = await hyde.transform(Query(text="why do we need embeddings?"))

from rag_query import ExpansionStrategy

expander = ExpansionStrategy(count=3)
variants = await expander.transform(Query(text="vector search performance"))
```

### `LLM` protocol

A duck-typed protocol (in `strategies/_base.py`) with a single required method:
`async def generate(request: GenerationRequest) -> GenerationResult`. Any object
exposing this shape is accepted — there is no hard dependency on a specific
provider client.

## Usage Guides

### Beginner — fast path (no LLM)

```python
import asyncio
from rag_core.queries import Query
from rag_query import QueryConfig, QueryEngine


async def main():
    engine = QueryEngine(QueryConfig())
    result = await engine.process(Query(text="rag vector search"))
    print(result.query_class)  # "keyword"
    print(result.variants)  # [original variant only]
    print(result.timings_ms)


asyncio.run(main())
```

### Intermediate — variants + parallel retrieval

```python
import asyncio
from rag_core.queries import Query
from rag_query import QueryConfig, QueryEngine, ParallelRetrievalExecutor


async def main():
    config = QueryConfig(expand=True, decompose=True, deterministic_only=True, max_variants=5)
    engine = QueryEngine(config)
    result = await engine.process(Query(text="compare dense and sparse retrieval"))

    executor = ParallelRetrievalExecutor([dense_retriever, bm25_retriever], max_concurrency=4)
    hits_per_form = await executor.execute(result.query, result.variants, top_k=10)

    # Merge all results (dedup / fuse upstream).
    all_hits = [hit for res in hits_per_form for hit in res.hits]
    print(f"{len(all_hits)} hits across {len(hits_per_form)} query forms")


asyncio.run(main())
```

### Advanced — HyDE with an LLM

```python
import asyncio
from rag_core.queries import Query
from rag_query import QueryConfig, QueryEngine, HyDEStrategy, ParallelRetrievalExecutor
from my_llm import MyAsyncLLM


async def main():
    # deterministic_only=False permits LLM strategies; HyDE requires one.
    config = QueryConfig(
        hyde=True,
        expand=True,
        decompose=True,
        deterministic_only=False,
        max_variants=6,
        expansion_count=3,
    )
    llm = MyAsyncLLM()
    engine = QueryEngine(config, llm=llm)
    result = await engine.process(
        Query(text="why is retrieval-augmented generation better than pure LLMs?")
    )

    executor = ParallelRetrievalExecutor([vector_retriever], max_concurrency=2)
    hits_per_form = await executor.execute(result.query, result.variants, top_k=15)


asyncio.run(main())
```

### Advanced — custom strategy

```python
from rag_core.protocols import QueryStrategy
from rag_core.queries import Query, QueryVariant


class EmojiStrategy(QueryStrategy):
    async def transform(self, query: Query) -> list[QueryVariant]:
        return [
            QueryVariant(
                query_id=query.id,
                text=f"{query.text} 😀",
                kind="expansion",
                strategy="emoji",
            )
        ]


engine = QueryEngine(QueryConfig(expand=True), strategies=[EmojiStrategy()])
```

## Configuration

All transformation is driven by `QueryConfig`. Construct it directly or load it
from a YAML/JSON dict (it is a `rag_core.base.RagBaseModel`, so `model_validate`
works):

```python
from rag_core.base import to_json, from_json  # rag-core serde helpers
from rag_query import QueryConfig

config = QueryConfig.model_validate(
    {
        "expand": True,
        "decompose": True,
        "max_variants": 6,
        "deterministic_only": False,
    }
)
```

| Consideration | Guidance |
| ------------- | -------- |
| `max_variants` | Keep small (3–8) — each variant is a separate retrieval pass. |
| `deterministic_only` | Leave `True` for zero-token, zero-network operation. Set `False` only when an LLM is wired in. |
| `min_query_length` | Queries shorter than this token count skip strategies entirely. |
| `expand`/`rewrite`/`decompose` | Safe to combine; `hyde` is LLM-only and is rejected under `deterministic_only`. |

The `ParallelRetrievalExecutor` takes `max_concurrency` (>= 1) directly; there is
no config model for it.

## Installation

```bash
uv pip install rag-query
```

In the monorepo, this pulls in the workspace siblings `rag-core` and
`rag-retrieval`.

## Testing

Tests are offline and use in-process fakes (`FakeLLM`, fake retrievers, and fake
strategies defined in `tests/conftest.py`).

```bash
uv run pytest packages/rag-query -q
```

## Dependencies

- `rag-core` — canonical models (`Query`, `QueryVariant`, `RetrievalResult`,
  `GenerationRequest`), the `QueryStrategy` / `Retriever` protocols, and the
  error taxonomy (`ConfigError`, `RetrievalError`).
- `rag-retrieval` — the default retrievers this package fans out against (this is
  a runtime dependency so `ParallelRetrievalExecutor` can accept the built-in
  `DenseRetriever` / `BM25Retriever` / `HybridRetriever` directly).

No transitive HTTP or LLM vendor SDKs are required.

## Cross-Package Relationships

- **rag-core** — defines the `Query`, `QueryVariant`, `RetrievalHit`,
  `RetrievalResult`, `GenerationRequest`, `GenerationResult` models, the
  `QueryStrategy` and `Retriever` protocols, and `RagError`/`ConfigError`/
  `RetrievalError`. This package is a pure implementation layer over those
  contracts (ADP-0002).
- **rag-retrieval** — provides the concrete `DenseRetriever`, `BM25Retriever`, and
  `HybridRetriever` retrievers that are typically passed into
  `ParallelRetrievalExecutor`. `rag-retrieval` itself depends on `rag-core`'s
  `Retriever` protocol and fusion strategies.
- **rag-llm-provider** — supplies model routing / `ProviderInfo` that a consumer's
  LLM implementation can resolve when building a generator to pass into an LLM
  strategy (e.g. `HyDEStrategy`). This package does not import `rag-llm-provider`
  directly; it only requires the duck-typed `LLM` shape.
- **rag-generation / rag-orchestrator** — the orchestrator typically wires
  `QueryEngine` + `ParallelRetrievalExecutor` together and attaches an LLM-backed
  generator before passing the result into `rag-context` for context assembly.

## Layout

```
src/rag_query/
  __init__.py        public API
  py.typed           PEP 561 marker
  classify.py        QueryClass + classify_query
  config.py          QueryConfig
  engine.py          QueryEngine + QueryResult
  normalize.py       normalize_query
  retrieval_exec.py  ParallelRetrievalExecutor
  strategies/
    __init__.py      re-exports
    _base.py         LLM protocol + is_short_query helper
    decompose.py     DecomposeStrategy
    expansion.py     ExpansionStrategy
    hyde.py          HyDEStrategy
    rewrite.py       RewriteStrategy
tests/
  conftest.py        FakeLLM, fake retrievers, VariantStrategy
  test_classify.py   intent classification
  test_normalize.py  normalization
  test_strategies.py rewrite/expand/decompose/hyde
  test_engine.py     pipeline, dedup, caps, concurrency
  test_executor.py   parallel execution, partial failure, all-fail
  test_smoke.py      import + version
```

## Verification

```bash
uv run ruff format packages/rag-query && uv run ruff check packages/rag-query
uv run mypy packages/rag-query/src
uv run pytest packages/rag-query -q
```
