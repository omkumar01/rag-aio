"""Tests for :mod:`rag_context.dedup`."""

from __future__ import annotations

from rag_context import dedupe_hits
from rag_core.retrieval import RetrievalHit


def _hit(chunk_id: str, text: str, score: float, **scores: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc",
        score=score,
        normalized_score=score,
        rank=0,
        strategy="dense",
        text=text,
        strategy_scores=dict(scores),
    )


def test_exact_duplicates_removed() -> None:
    hits = [
        _hit("1", "hello world", 0.9, dense=0.9),
        _hit("2", "hello world", 0.5, dense=0.5),
    ]
    out = dedupe_hits(hits, merge_overlapping=False)
    assert len(out) == 1
    assert out[0].chunk_id == "1"  # higher score kept
    assert out[0].score == 0.9
    # Strategy scores accumulated via max-merge.
    assert out[0].strategy_scores == {"dense": 0.9}


def test_preserves_order_of_first_appearance() -> None:
    hits = [
        _hit("1", "alpha beta", 0.5),
        _hit("2", "gamma delta", 0.9),
        _hit("3", "alpha beta", 0.7),
    ]
    out = dedupe_hits(hits, merge_overlapping=False)
    # "alpha beta" group first appears at chunk 1; its higher-scorer is chunk 3.
    # "gamma delta" group (chunk 2) follows. Order follows group first appearance.
    assert [h.chunk_id for h in out] == ["3", "2"]
    assert out[0].score == 0.7


def test_no_merge_without_merge_overlapping() -> None:
    hits = [
        _hit("1", "alpha beta gamma", 0.9),
        _hit("2", "alpha beta gamma delta", 0.5),
    ]
    out = dedupe_hits(hits, merge_overlapping=False)
    assert len(out) == 2


def test_jaccard_near_duplicates_merged() -> None:
    hits = [
        _hit("1", "alpha beta gamma delta", 0.9, dense=0.9),
        _hit("2", "alpha beta gamma delta epsilon zeta", 0.5, sparse=0.5),
    ]
    # Jaccard of token sets: 4/6 = 0.667 >= 0.5 threshold.
    out = dedupe_hits(hits, merge_overlapping=True, similarity_threshold=0.5)
    assert len(out) == 1
    assert out[0].score == 0.9
    assert out[0].chunk_id == "1"
    assert "sparse" in out[0].strategy_scores
    assert out[0].strategy_scores["sparse"] == 0.5


def test_jaccard_below_threshold_keeps_both() -> None:
    hits = [
        _hit("1", "alpha beta gamma delta", 0.9),
        _hit("2", "alpha beta epsilon zeta eta theta", 0.5),
    ]
    # intersection {alpha, beta} = 2, union 7 -> 0.286 < 0.5 -> kept.
    out = dedupe_hits(hits, merge_overlapping=True, similarity_threshold=0.5)
    assert len(out) == 2


def test_none_text_deduped_by_chunk_id() -> None:
    hits = [
        RetrievalHit(
            chunk_id="1",
            document_id="doc",
            score=0.9,
            normalized_score=0.9,
            rank=0,
            strategy="dense",
            text=None,
        ),
        RetrievalHit(
            chunk_id="1",
            document_id="doc",
            score=0.5,
            normalized_score=0.5,
            rank=1,
            strategy="dense",
            text=None,
        ),
    ]
    out = dedupe_hits(hits, merge_overlapping=False)
    assert len(out) == 1
    assert out[0].score == 0.9
