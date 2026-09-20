"""Tests for :mod:`rag_context.ordering`."""

from __future__ import annotations

from rag_context.ordering import (
    chronological,
    diversity,
    document_grouped,
    order_by_strategy,
    relevance_first,
    section_aware,
)
from rag_core.context import ContextItem


def _items() -> list[ContextItem]:
    """Fixed fixture set: d1 score 0.9 page2 s2, d1 score 0.5 page1 s1, d2 score 0.8 page3 s3."""
    return [
        ContextItem(
            chunk_id="1",
            document_id="d1",
            text="a",
            token_count=1,
            score=0.9,
            page_numbers=[2],
            section_path=["s2"],
        ),
        ContextItem(
            chunk_id="2",
            document_id="d1",
            text="b",
            token_count=1,
            score=0.5,
            page_numbers=[1],
            section_path=["s1"],
        ),
        ContextItem(
            chunk_id="3",
            document_id="d2",
            text="c",
            token_count=1,
            score=0.8,
            page_numbers=[3],
            section_path=["s3"],
        ),
    ]


def test_relevance_first_orders_by_score_desc() -> None:
    out = relevance_first(_items())
    assert [i.chunk_id for i in out] == ["1", "3", "2"]


def test_relevance_first_none_scores_last() -> None:
    items = [
        ContextItem(chunk_id="1", document_id="d1", text="a", token_count=1, score=0.5),
        ContextItem(chunk_id="2", document_id="d1", text="b", token_count=1, score=None),
        ContextItem(chunk_id="3", document_id="d1", text="c", token_count=1, score=0.9),
    ]
    out = relevance_first(items)
    assert [i.chunk_id for i in out] == ["3", "1", "2"]


def test_chronological_by_document_then_page() -> None:
    out = chronological(_items())
    assert [i.chunk_id for i in out] == ["2", "1", "3"]


def test_section_aware() -> None:
    out = section_aware(_items())
    assert [i.chunk_id for i in out] == ["2", "1", "3"]


def test_document_grouped_groups_by_document() -> None:
    out = document_grouped(_items())
    # d1 (best 0.9) first preserving input order, then d2.
    assert [i.chunk_id for i in out] == ["1", "2", "3"]
    # Both d1 items are contiguous.
    docs = [i.document_id for i in out]
    assert docs == ["d1", "d1", "d2"]


def test_diversity_round_robins_documents() -> None:
    out = diversity(_items())
    # d1=[1,2], d2=[3]; round-robin: 1, 3, 2.
    assert [i.chunk_id for i in out] == ["1", "3", "2"]


def test_order_by_strategy_dispatch() -> None:
    items = _items()
    assert order_by_strategy(items, "relevance_first") == relevance_first(items)
