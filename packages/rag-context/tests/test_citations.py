"""Tests for :mod:`rag_context.citations`."""

from __future__ import annotations

from rag_context import build_citations
from rag_core.context import Context, ContextItem


def _context(items: list[ContextItem]) -> Context:
    return Context(items=items, token_budget=100, strategy="relevance_first")


def test_citations_in_order_with_quote_prefix() -> None:
    context = _context(
        [
            ContextItem(
                chunk_id="a",
                document_id="d1",
                text="alpha beta gamma",
                token_count=3,
                citation_id="1",
            ),
            ContextItem(
                chunk_id="b",
                document_id="d2",
                text="delta epsilon",
                token_count=2,
                citation_id="2",
            ),
        ]
    )
    cites = build_citations(context)
    assert [c.citation_id for c in cites] == ["1", "2"]
    assert cites[0].quote == "alpha beta gamma"[:120]
    assert cites[0].document_id == "d1"
    assert cites[0].chunk_id == "a"


def test_source_lookup_fills_missing() -> None:
    def lookup(chunk_id: str) -> tuple[str, list[int]] | None:
        if chunk_id == "a":
            return ("http://example/doc1", [1, 2])
        return None

    context = _context(
        [
            ContextItem(
                chunk_id="a",
                document_id="d1",
                text="alpha beta",
                token_count=2,
                citation_id="1",
                source_uri=None,
                page_numbers=[],
            )
        ]
    )
    cites = build_citations(context, source_lookup=lookup)
    assert cites[0].source_uri == "http://example/doc1"
    assert cites[0].page_numbers == [1, 2]


def test_source_lookup_preserves_existing() -> None:
    def lookup(chunk_id: str) -> tuple[str, list[int]] | None:
        return ("http://should/not/be/used", [99])

    context = _context(
        [
            ContextItem(
                chunk_id="a",
                document_id="d1",
                text="alpha",
                token_count=1,
                citation_id="1",
                source_uri="existing",
                page_numbers=[3],
            )
        ]
    )
    cites = build_citations(context, source_lookup=lookup)
    assert cites[0].source_uri == "existing"
    assert cites[0].page_numbers == [3]


def test_none_citations_omitted() -> None:
    context = _context(
        [
            ContextItem(
                chunk_id="a",
                document_id="d1",
                text="alpha",
                token_count=1,
                citation_id=None,
            )
        ]
    )
    assert build_citations(context) == []


def test_quote_capped_to_120_chars() -> None:
    long_text = "word " * 60  # 300 chars
    context = _context(
        [
            ContextItem(
                chunk_id="a",
                document_id="d1",
                text=long_text,
                token_count=60,
                citation_id="1",
            )
        ]
    )
    cites = build_citations(context)
    assert len(cites[0].quote) == 120  # type: ignore[arg-type]
