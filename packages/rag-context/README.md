# rag-context

Context-engineering layer between retrieval and generation for the
[rag-aio](https://github.com/rag-aio/rag-aio) platform.

`rag-context` turns a list of :class:`~rag_core.retrieval.RetrievalHit` objects
into a token-budgeted, citation-preserving :class:`~rag_core.context.Context`
ready to inject into a generation prompt. It is modular and async-first: every
stage is a small, independently testable function, and the orchestrator calls
the async :class:`~rag_context.ContextBuilderImpl`.

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

## Tokenizer abstraction

:class:`~rag_context.ContextTokenizer` accepts any backend with one of two
duck-typed shapes:

- HuggingFace `tokenizers` — `encode(text)` returns an object with `.ids`;
  `decode(ids)` returns `str`.
- rag-core `Tokenizer` — `count_tokens(text) -> int`.

It exposes `count(text) -> int` and
`truncate_to_tokens(text, max_tokens) -> str`. A `WhitespaceCounter` fallback
(`len(text.split())`) is provided for tests and zero-dependency scenarios.

## Usage

```python
import asyncio
from rag_context import (
    ContextBuilderImpl,
    ContextConfig,
    ContextTokenizer,
    WhitespaceCounter,
    build_citations,
    render_context,
)
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
ctx = asyncio.run(builder.build(query, hits, token_budget=config.token_budget))
print(render_context(ctx))
print(build_citations(ctx, source_lookup=lambda chunk_id: ("http://...", [1])))
```

> **System instructions are assembled by the caller.** `render_context` only
> formats the selected evidence; it never injects prompt templates or
> instructions.

## Layout

```
src/rag_context/
  __init__.py     public API
  tokenizer.py    ContextTokenizer + WhitespaceCounter
  config.py       ContextConfig
  dedup.py        dedupe_hits
  expansion.py    expand_with_neighbors
  ordering.py     per-strategy ordering functions
  builder.py      ContextBuilderImpl + sanitize_item_text
  citations.py    build_citations
  render.py       render_context
tests/
  conftest.py    shared fixtures (tokenizer, make_hit)
  test_*.py       tokenizer, config, dedup, ordering, expansion, builder, citations
```

## Verification

Package-scoped only:

```bash
uv run ruff format packages/rag-context && uv run ruff check packages/rag-context
uv run mypy packages/rag-context/src
uv run pytest packages/rag-context -q
```

## Dependencies

- `rag-core` (canonical domain models and the `ContextBuilder` protocol)
- `tokenizers>=0.19` (optional at runtime; only required when wrapping a
  HuggingFace tokenizer)
