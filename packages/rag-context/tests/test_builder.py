"""Tests for :mod:`rag_context.builder`."""

from __future__ import annotations

from typing import Any

from rag_context import (
    ContextBuilderImpl,
    ContextConfig,
    ContextTokenizer,
    WhitespaceCounter,
    build_citations,
    render_context,
)
from rag_core.queries import Query


def _wc() -> ContextTokenizer:
    return ContextTokenizer(WhitespaceCounter())


# --------------------------------------------------------------------------- #
# Pipeline: empty + budget
# --------------------------------------------------------------------------- #


async def test_empty_hits_returns_empty_context() -> None:
    cfg = ContextConfig()
    builder = ContextBuilderImpl(_wc(), cfg)
    ctx = await builder.build(Query(text="q"), [], token_budget=100)
    assert ctx.items == []
    assert ctx.truncated is False
    assert ctx.token_budget == 100


async def test_budget_respected_and_truncated_flag(make_hit: Any) -> None:
    cfg = ContextConfig(strategy="relevance_first", reserve_for_answer=0)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "one two three four"),  # 4 tokens
        make_hit("b", "d1", 0.8, "five six"),  # 2 tokens
        make_hit("c", "d1", 0.7, "seven eight nine ten"),  # 4 tokens
    ]
    ctx = await builder.build(Query(text="q"), hits, token_budget=6)
    # relevance order: a(4), b(2), c(4). a fits (4); b fits (6); c exceeds.
    assert ctx.total_tokens <= 6
    assert ctx.truncated is True
    assert [i.chunk_id for i in ctx.items] == ["a", "b"]


async def test_greedy_best_score_selection(make_hit: Any) -> None:
    cfg = ContextConfig(strategy="relevance_first", token_budget=3)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("big", "d1", 0.9, "one two three four five"),  # 5 > budget
        make_hit("small", "d1", 0.8, "a b"),  # 2
    ]
    ctx = await builder.build(Query(text="q"), hits, token_budget=3)
    # Big (5) doesn't fit budget 3; small (2) does. Greedy keeps smaller one.
    assert [i.chunk_id for i in ctx.items] == ["small"]
    assert ctx.total_tokens == 2
    assert ctx.truncated is True


# --------------------------------------------------------------------------- #
# Filters / caps
# --------------------------------------------------------------------------- #


async def test_min_score_filter(make_hit: Any) -> None:
    cfg = ContextConfig(min_score=0.8, token_budget=1000, dedup=False)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "x y"),
        make_hit("b", "d1", 0.5, "z w"),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    assert [i.chunk_id for i in ctx.items] == ["a"]


async def test_max_per_document_cap(make_hit: Any) -> None:
    cfg = ContextConfig(
        token_budget=1000, max_per_document=1, strategy="relevance_first", dedup=False
    )
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "x y z"),
        make_hit("b", "d1", 0.8, "p q r"),
        make_hit("c", "d2", 0.7, "s t u"),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    d1_items = [i for i in ctx.items if i.document_id == "d1"]
    assert len(d1_items) == 1
    assert len(ctx.items) == 2
    assert ctx.truncated is True  # one d1 item dropped by cap


async def test_reserve_for_answer_reduces_effective_budget(make_hit: Any) -> None:
    cfg = ContextConfig(token_budget=10, reserve_for_answer=4)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [make_hit("a", "d1", 0.9, "one two three four five six")]  # 6 tokens
    ctx = await builder.build(Query(text="q"), hits, token_budget=10)
    # effective = 10 - 4 = 6; item fits exactly.
    assert ctx.total_tokens == 6
    assert ctx.truncated is False


# --------------------------------------------------------------------------- #
# Sanitize
# --------------------------------------------------------------------------- #


async def test_sanitize_strips_control_chars_and_caps_length(make_hit: Any) -> None:
    cfg = ContextConfig(max_item_chars=10, citation_style="none", strategy="relevance_first")
    builder = ContextBuilderImpl(_wc(), cfg)
    raw = "hello\x00\x01world\n\n\n\nfoo"
    hits = [make_hit("a", "d1", 0.9, raw)]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    text = ctx.items[0].text
    assert "\x00" not in text
    assert "\x01" not in text
    assert text.count("\n") <= 2
    assert len(text) <= 10


async def test_sanitize_collapses_newline_runs(make_hit: Any) -> None:
    cfg = ContextConfig(max_item_chars=1000, citation_style="none")
    builder = ContextBuilderImpl(_wc(), cfg)
    raw = "para one\n\n\n\n\npara two"
    hits = [make_hit("a", "d1", 0.9, raw)]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    assert "\n\n\n" not in ctx.items[0].text


# --------------------------------------------------------------------------- #
# Citations + render
# --------------------------------------------------------------------------- #


async def test_citation_numbering_matches_render(make_hit: Any) -> None:
    cfg = ContextConfig(citation_style="numeric", strategy="relevance_first")
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "alpha beta"),
        make_hit("b", "d1", 0.8, "gamma delta"),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    cites = build_citations(ctx)
    assert [c.citation_id for c in cites] == ["1", "2"]
    rendered = render_context(ctx)
    blocks = rendered.split("\n\n")
    assert blocks[0].startswith("[1] alpha beta")
    assert blocks[1].startswith("[2] gamma delta")


async def test_citation_style_none_omits_ids(make_hit: Any) -> None:
    cfg = ContextConfig(citation_style="none", token_budget=1000)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [make_hit("a", "d1", 0.9, "alpha beta")]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    assert ctx.items[0].citation_id is None
    assert build_citations(ctx) == []


# --------------------------------------------------------------------------- #
# Strategies end-to-end
# --------------------------------------------------------------------------- #


async def test_strategy_chronological(make_hit: Any) -> None:
    cfg = ContextConfig(strategy="chronological", token_budget=1000, dedup=False)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "x y", metadata={"page_numbers": [2]}),
        make_hit("b", "d1", 0.5, "z w", metadata={"page_numbers": [1]}),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    assert [i.chunk_id for i in ctx.items] == ["b", "a"]


async def test_strategy_document_grouped(make_hit: Any) -> None:
    cfg = ContextConfig(strategy="document_grouped", token_budget=1000, dedup=False)
    builder = ContextBuilderImpl(_wc(), cfg)
    hits = [
        make_hit("a", "d1", 0.9, "x y"),
        make_hit("b", "d2", 0.8, "z w"),
        make_hit("c", "d1", 0.7, "p q"),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    # d1 best 0.9, d2 best 0.8 -> d1 group first (a, c), then d2 (b)
    assert [i.chunk_id for i in ctx.items] == ["a", "c", "b"]


# --------------------------------------------------------------------------- #
# Overlap merging + neighbor expansion through the builder
# --------------------------------------------------------------------------- #


async def test_overlap_merge_consecutive_items(make_hit: Any) -> None:
    cfg = ContextConfig(token_budget=1000, strategy="relevance_first", dedup=False)
    builder = ContextBuilderImpl(_wc(), cfg)
    # Two chunks sharing tokens "gamma delta" on the boundary (>50% overlap).
    hits = [
        make_hit("a", "d1", 0.9, "alpha beta gamma delta"),
        make_hit("b", "d1", 0.8, "gamma delta epsilon"),
    ]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    assert len(ctx.items) == 1
    merged_text = ctx.items[0].text
    assert "alpha beta gamma delta" in merged_text
    assert "epsilon" in merged_text
    # The duplicated overlap should not appear twice.
    assert merged_text.count("gamma delta") == 1


async def test_neighbor_expansion_budget_respected(make_hit: Any) -> None:
    cfg = ContextConfig(
        expand_neighbors=True,
        neighbor_window=1,
        token_budget=3,
        strategy="relevance_first",
        dedup=False,
    )

    def provider(cid: str) -> list[tuple[str, str, int]]:
        return [("n1", "neighbor one two", 0)] if cid == "a" else []

    builder = ContextBuilderImpl(_wc(), cfg, neighbor_provider=provider)
    hits = [make_hit("a", "d1", 0.9, "original text here")]  # 3 tokens
    ctx = await builder.build(Query(text="q"), hits, 3)
    # Original (3) fits; neighbor (3) would exceed budget -> skipped.
    assert [h.chunk_id for h in ctx.items] == ["a"]
    assert ctx.total_tokens == 3
    assert ctx.truncated is True


async def test_neighbor_no_duplicate_chunks(make_hit: Any) -> None:
    cfg = ContextConfig(
        expand_neighbors=True,
        neighbor_window=5,
        token_budget=1000,
        dedup=False,
        citation_style="none",
    )

    def provider(cid: str) -> list[tuple[str, str, int]]:
        return [("n1", "nb one", 0)] if cid == "a" else []

    builder = ContextBuilderImpl(_wc(), cfg, neighbor_provider=provider)
    hits = [make_hit("a", "d1", 0.9, "original"), make_hit("n1", "d1", 0.5, "dup")]
    ctx = await builder.build(Query(text="q"), hits, 1000)
    ids = [i.chunk_id for i in ctx.items]
    assert ids.count("n1") == 1
    assert "a" in ids
