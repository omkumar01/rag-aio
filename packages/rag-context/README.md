# rag-context
> Part of the [rag-aio](../../README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Context-engineering layer between retrieval and generation for the
[rag-aio](https://github.com/omkumar01/rag-aio) platform.

`rag-context` turns a list of :class:`~rag_core.retrieval.RetrievalHit` objects
into a token-budgeted, citation-preserving :class:`~rag_core.context.Context`
ready to inject into a generation prompt. It is modular and async-first: every
stage is a small, independently testable function, and the orchestrator calls
the async :class:`~rag_context.ContextBuilderImpl`.

## Overview

After retrieval and (optionally) reranking, the system holds a list of scored
chunks. `rag-context` is responsible for shaping those chunks into a prompt-sized,
non-redundant, citation-linked evidence block. It does this through a fixed,
ordered pipeline:

1. Score filtering
2. Deduplication (exact + near-duplicate)
3. Optional neighbor expansion
4. ContextItem conversion + token counting
5. Strategy-based ordering
6. Greedy token-budget selection
7. Per-document cap
8. Overlap merge
9. Citation assignment

The result is a :class:`~rag_core.context.Context` whose `truncated` flag is set
when any candidate was dropped for budget or by a per-document cap.

> **System instructions are assembled by the caller.** `render_context` only
> formats the selected evidence; it never injects prompt templates or
> instructions.

## Architecture / Design Principles

- **Protocol-based contracts (ADR-0002).** `ContextBuilderImpl` implements the
  :class:`~rag_core.protocols.ContextBuilder` contract (`async build(query, hits,
  token_budget) -> Context`). Callers depend on the abstract protocol, not the
  concrete class.
- **Single-pass, composable stages.** Each pipeline stage is a pure function
  (`dedupe_hits`, `order_by_strategy`, `_greedy_select`, `_merge_overlapping`,
  `build_citations`), making them independently testable and reusable.
- **Token-budget honesty.** A `ContextTokenizer` abstraction normalizes any
  backend (HuggingFace `tokenizers`, rag-core `Tokenizer`, or a
  `WhitespaceCounter` fallback) behind `count` / `truncate_to_tokens`. Budgeting
  is always in real tokens, never raw characters.
- **Greedy, not optimal.** Budget selection is greedy in final order: an item
  that does not fit is skipped, but smaller items further down are still tried —
  so a large chunk early never starves small, relevant chunks later.
- **Deduplication with score preservation.** Exact duplicates are removed by
  normalized-text hash; near-duplicates are merged via token-set Jaccard
  similarity (union-find clustering). Per-strategy scores accumulate (max per
  strategy) into the kept representative, so fusion provenance is not lost.
- **Citation stability.** Citations are assigned in the *final rendering order*
  via numeric ids (`[1]`, `[2]`, ...) matching `render_context`'s `numbered`
  output, with optional backfill from a caller-supplied `source_lookup`.
- **Async-first, zero network in the core path.** The `build` pipeline itself is
  synchronous in CPU terms; `async def` is used so a future neighbor provider
  can fetch from a remote chunker without breaking the protocol.
- **Defensive sanitization.** `sanitize_item_text` strips control characters,
  normalizes line endings, collapses runaway newlines, and caps length before
  any item is counted or rendered, keeping untrusted retrieval output safe.

## Pipeline

Given `hits` and `token_budget`, the builder runs:

1. **Score filter** — drops hits below `ContextConfig.min_score`.
2. **Dedup** — `dedupe_hits` removes exact duplicates (by normalized text hash)
   and, when `merge_overlapping` is set, merges near-duplicates via token-set
   Jaccard, accumulating `strategy_scores`.
3. **Neighbor expansion** — `expand_with_neighbors` inserts adjacent chunks from a
   caller-supplied provider, annotated as neighbors and never duplicating a
   selected `chunk_id`.
4. **ContextItem conversion** — each hit is sanitized
   (`sanitize_item_text`: control chars stripped, newline runs collapsed, length
   capped) and token-counted via the configured tokenizer.
5. **Ordering** — `ordering.order_by_strategy` applies one of five strategies:
   `relevance_first`, `chronological`, `section_aware`, `document_grouped`,
   `diversity`.
6. **Greedy budget selection** — the highest-priority items that fit are kept;
   non-fitting items are skipped but smaller ones further down are still tried.
7. **Per-document cap** — `max_per_document` enforces a hard limit per document.
8. **Overlap merge** — consecutive items whose trailing/leading text overlaps by
   more than 50% (longest suffix/prefix) collapse into a single union item.
9. **Citations** — `citation_style="numeric"` assigns `[1]`-style ids in final
   order (or `none` to suppress); `build_citations` produces
   :class:`~rag_core.context.Citation` objects, optionally backfilled by a
   `source_lookup`.

The resulting `Context.truncated` is `True` when any candidate was dropped for
budget or by a per-document cap.

## Public API

### `ContextConfig`

Tunables controlling context assembly, validated via `rag_core.base.RagBaseModel`.

| field                | type                              | default            | meaning                                                |
| -------------------- | --------------------------------- | ------------------ | ------------------------------------------------------ |
| `strategy`           | `Strategy` (Literal)              | `"relevance_first"`| Ordering strategy (see `STRATEGIES`).                  |
| `token_budget`       | `int`                             | `2048`             | Total token budget for the assembled context.          |
| `reserve_for_answer` | `int`                             | `0`                | Tokens subtracted from `token_budget` for the response.|
| `max_per_document`   | `int`                             | `0`                | Per-document item cap (`0` = unlimited).               |
| `dedup`              | `bool`                            | `True`             | Remove exact + near-duplicate hits.                    |
| `merge_overlapping`  | `bool`                            | `True`             | Merge near-duplicates (Jaccard) during dedup.          |
| `expand_neighbors`   | `bool`                            | `False`            | Insert neighbor chunks from `neighbor_provider`.       |
| `neighbor_window`    | `int`                             | `1`                | Max neighbors per hit.                               |
| `min_score`          | `float \| None`                   | `None`             | Drop hits below this score.                            |
| `citation_style`     | `CitationStyle` (Literal)         | `"numeric"`        | `"numeric"` (`[n]`) or `"none"`.                       |
| `max_item_chars`     | `int`                             | `8000`             | Per-item text length cap (post-sanitization).          |

### `ContextBuilderImpl`

The concrete :class:`~rag_core.protocols.ContextBuilder`. Construct it with a
`ContextTokenizer` and a `ContextConfig`; optionally pass a
`NeighborProvider` when `expand_neighbors=True`.

```python
import asyncio
from rag_context import (
    ContextBuilderImpl,
    ContextConfig,
    ContextTokenizer,
    WhitespaceCounter,
    render_context,
    build_citations,
)
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit

tokenizer = ContextTokenizer(WhitespaceCounter())
config = ContextConfig(strategy="relevance_first", token_budget=2048, reserve_for_answer=256)
builder = ContextBuilderImpl(tokenizer, config)

hits = [
    RetrievalHit(
        chunk_id="1",
        document_id="doc",
        score=0.95,
        normalized_score=0.95,
        rank=0,
        strategy="dense",
        text="...",
    )
]
ctx = asyncio.run(builder.build(Query(text="what is rag?"), hits, token_budget=config.token_budget))

print(render_context(ctx))
print(build_citations(ctx, source_lookup=lambda chunk_id: ("https://...", [1])))
```

### `ContextTokenizer` + `WhitespaceCounter`

`ContextTokenizer` accepts any backend with one of two duck-typed shapes:

- **HuggingFace `tokenizers`** — `encode(text)` returns an object with `.ids`;
  `decode(ids)` returns `str`.
- **rag-core `Tokenizer`** — `count_tokens(text) -> int`.

It exposes `count(text) -> int` and `truncate_to_tokens(text, max_tokens) -> str`.
A `WhitespaceCounter` fallback (`len(text.split())`) is provided for tests and
zero-dependency scenarios.

```python
from rag_context import ContextTokenizer, WhitespaceCounter

tok = ContextTokenizer(WhitespaceCounter())
tok.count("hello world foo")  # 3
tok.truncate_to_tokens("a b c d", 2)  # "a b"

# Wrap a HuggingFace tokenizer:
# tok = ContextTokenizer(hf_tokenizer)
```

### `dedupe_hits(hits, merge_overlapping, similarity_threshold)`

Two-pass deduplication:

1. **Exact dedup** by normalized-text hash (lower-cased, whitespace-collapsed),
   falling back to `chunk_id` when text is empty. The higher-scored hit is kept;
   per-strategy scores from dropped members accumulate via a max-merge into
   `strategy_scores`.
2. **Near-duplicate merge** (when `merge_overlapping=True`) via token-set Jaccard
   similarity over hit text, clustering with a union-find and collapsing each
   cluster to its highest-scored representative (`similarity_threshold=0.9`).

```python
from rag_context import dedupe_hits

unique = dedupe_hits(hits, merge_overlapping=True, similarity_threshold=0.85)
```

### `expand_with_neighbors(hits, neighbor_provider, window)`

Inserts supplementary chunks (siblings reported adjacent to a selected hit) into
the candidate stream. Inserted neighbors inherit the parent's score/strategy/
provenance and are annotated in metadata so they are never confused with
primary results. `chunk_id`s already present are never duplicated.

```python
from rag_context import expand_with_neighbors


def neighbor_provider(chunk_id: str):
    # Return [(neighbor_chunk_id, neighbor_text, neighbor_index), ...]
    ...


hits = expand_with_neighbors(hits, neighbor_provider, window=1)
```

`NeighborProvider = Callable[[str], tuple[str, str, int] | list[tuple[str, str, int]]]`

### Ordering strategies (`ordering.py`)

`order_by_strategy(items, strategy)` dispatches to one of five pure ordering
functions. All are deterministic and stable (ties keep input order).

```python
from rag_context import order_by_strategy, STRATEGIES

# Apply a named strategy:
ordered = order_by_strategy(items, "diversity")

# Call a strategy function directly:
from rag_context import relevance_first

ordered = relevance_first(items)
```

| strategy           | behavior                                                  |
| ------------------ | --------------------------------------------------------- |
| `relevance_first`  | Score descending; `None` scores sort last.                 |
| `chronological`    | Document id, then first page number.                     |
| `section_aware`    | Document id, then section path.                          |
| `document_grouped` | Documents by best score (desc), items within in order.   |
| `diversity`        | Round-robin across documents by best score (desc).       |

### `build_citations(context, source_lookup)` / `render_context`

`build_citations` emits one :class:`~rag_core.context.Citation` per cited item (in
final order), with optional backfill of `source_uri` and `page_numbers` from a
caller-supplied `SourceLookup = Callable[[str], tuple[str, Sequence[int]] | None]`.

```python
from rag_context import build_citations, render_context

citations = build_citations(ctx, source_lookup=lambda cid: ("https://doc", [2, 3]))
print(render_context(ctx))  # "[1] text...\n\n[2] text..."
```

### `sanitize_item_text(text, max_chars)`

Collapse whitespace, strip control characters (newline/tab preserved), cap
length. Exported for direct use by callers that preprocess hits.

```python
from rag_context import sanitize_item_text

sanitize_item_text("raw\r\n\ttext", max_chars=5)  # "raw\n\ttex"
```

## Usage Guides

### Beginner — minimal context with the whitespace tokenizer

```python
import asyncio
from rag_context import (
    ContextBuilderImpl, ContextConfig, ContextTokenizer, WhitespaceCounter,
    render_context,
)
from rag_core.retrieval import RetrievalHit

hits = [RetrievalHit(chunk_id="1", document_id="d1", score=0.9,
                    normalized_score=0.9, rank=0, strategy="dense",
                    text="RAG combines retrieval and generation")]]
builder = ContextBuilderImpl(ContextTokenizer(WhitespaceCounter()), ContextConfig())
ctx = asyncio.run(builder.build(Query(text="what is rag"), hits, token_budget=50))
print(render_context(ctx))
```

### Intermediate — real tokenizer + scoring + citations

```python
from tokenizers import Tokenizer
from rag_context import ContextBuilderImpl, ContextConfig, ContextTokenizer, build_citations

tok = ContextTokenizer(my_hf_tokenizer)
config = ContextConfig(
    strategy="relevance_first",
    token_budget=4096,
    reserve_for_answer=512,
    min_score=0.3,
    max_per_document=3,
    citation_style="numeric",
)
builder = ContextBuilderImpl(tok, config)
ctx = await builder.build(query, hits, token_budget=config.token_budget)
citations = build_citations(ctx, source_lookup=resolve_source)
```

### Advanced — neighbor expansion + diversity ordering

```python
from rag_context import ContextBuilderImpl, ContextConfig, ContextTokenizer, expand_with_neighbors

config = ContextConfig(
    strategy="diversity",
    token_budget=3000,
    reserve_for_answer=400,
    expand_neighbors=True,
    neighbor_window=2,
    max_per_document=2,
)
builder = ContextBuilderImpl(
    ContextTokenizer(WhitespaceCounter()),
    config,
    neighbor_provider=lambda cid: neighbor_index.get(cid, []),
)
ctx = await builder.build(query, hits, token_budget=config.token_budget)
```

### Advanced — custom ordering strategy

```python
from rag_context import ContextBuilderImpl, ContextConfig, ContextTokenizer, WhitespaceCounter
from rag_context.ordering import OrderFn, order_by_strategy, STRATEGIES
from rag_core.context import ContextItem


def my_strategy(items: list[ContextItem]) -> list[ContextItem]:
    return sorted(items, key=lambda i: i.token_count)


STRATEGIES["shortest_first"] = my_strategy  # register
ordered = order_by_strategy(items, "shortest_first")
```

## Installation

```bash
uv pip install rag-context
```

## Testing

Tests are offline and use `WhitespaceCounter` plus a small `wordlevel_tokenizer`
fixture (a `tokenizers` `Tokenizer` over a toy vocab) and a `make_hit` factory
defined in `tests/conftest.py`.

```bash
uv run pytest packages/rag-context -q
```

## Dependencies

- `rag-core` — canonical models (`Context`, `ContextItem`, `Citation`,
  `RetrievalHit`, `Query`), the `ContextBuilder` protocol, and error taxonomy.
- `tokenizers>=0.19` (optional at runtime) — only required when wrapping a
  HuggingFace `tokenizers.Tokenizer` backend. The `WhitespaceCounter` fallback
  makes the package fully functional without it.

## Cross-Package Relationships

- **rag-core** — defines `Context`, `ContextItem`, `Citation`, `RetrievalHit`,
  `Query`, the `ContextBuilder` / `Tokenizer` / `Retriever` protocols, and the
  error taxonomy. `rag-context` is a pure implementation layer over those
  contracts (ADR-0002); it re-exports nothing from `rag-core` but conforms to it.
- **rag-retrieval** — produces the `RetrievalResult` / `RetrievalHit` lists that
  feed the builder. Deduplication accumulates `strategy_scores` emitted by the
  retrieval fusion layer so provenance survives context assembly.
- **rag-rerank** — reranking happens upstream; the hits passed to `build` are
  assumed already ranked, which `relevance_first` ordering relies on.
- **rag-embedder / rag-llm-provider** — a real tokenizer backend is often obtained
  from `rag-embedder` (which owns model/tokenizer selection via
  `rag-llm-provider`'s `ProviderRegistry`); `ContextTokenizer` then wraps it. The
  package itself depends only on `rag-core`, keeping the token-counting path
  dependency-free.
- **rag-orchestrator** — typically constructs `ContextBuilderImpl`, invokes
  `build`, then calls `render_context` + `build_citations` to assemble the final
  prompt handed to `rag-generation`.

## Layout

```
src/rag_context/
  __init__.py     public API
  py.typed        PEP 561 marker
  tokenizer.py    ContextTokenizer + WhitespaceCounter
  config.py       ContextConfig, Strategy, CitationStyle
  dedup.py        dedupe_hits
  expansion.py    expand_with_neighbors + NeighborProvider
  ordering.py     per-strategy ordering functions + STRATEGIES
  builder.py      ContextBuilderImpl + sanitize_item_text
  citations.py    build_citations + SourceLookup
  render.py       render_context
tests/
  conftest.py    shared fixtures (tokenizer, make_hit)
  test_*.py      tokenizer, config, dedup, ordering, expansion, builder, citations
```

## Verification

```bash
uv run ruff format packages/rag-context && uv run ruff check packages/rag-context
uv run mypy packages/rag-context/src
uv run pytest packages/rag-context -q
```
