"""Tests for fusion strategies (RRF and weighted)."""

from __future__ import annotations

import math

import pytest
from rag_core.errors import FusionError
from rag_core.retrieval import RetrievalHit, RetrievalResult
from rag_retrieval import ReciprocalRankFusion, WeightedScoreFusion


def _hit(
    chunk_id: str,
    score: float,
    strategy: str = "dense",
    rank: int = 0,
    document_id: str = "doc",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        score=score,
        normalized_score=score,
        rank=rank,
        strategy=strategy,
        strategy_scores={strategy: score},
    )


def test_rrf_exact_math() -> None:
    """RRF score = sum of weight/(k+rank) per source, merged by chunk_id."""
    rrf = ReciprocalRankFusion(k=60.0)
    r1 = RetrievalResult(
        query_id="q",
        hits=[_hit("a", 0.9, strategy="dense", rank=0), _hit("b", 0.5, rank=1)],
        strategies=["dense"],
    )
    r2 = RetrievalResult(
        query_id="q",
        hits=[_hit("a", 0.8, strategy="sparse", rank=0)],
        strategies=["sparse"],
    )
    fused = rrf.fuse([r1, r2], top_k=10)

    a_exp = 1 / 60.0 + 1 / 60.0  # dense rank0 + sparse rank0
    b_exp = 1 / 61.0  # dense rank1

    assert len(fused.hits) == 2
    assert fused.hits[0].chunk_id == "a"
    assert fused.hits[0].strategy == "hybrid"
    assert math.isclose(fused.hits[0].score, a_exp)
    assert math.isclose(fused.hits[0].strategy_scores["dense"], 1 / 60.0)
    assert math.isclose(fused.hits[0].strategy_scores["sparse"], 1 / 60.0)
    assert math.isclose(fused.hits[0].normalized_score, 1.0)
    assert math.isclose(fused.hits[1].score, b_exp)
    assert math.isclose(fused.hits[1].normalized_score, b_exp / a_exp)


def test_rrf_dedup_by_chunk_id() -> None:
    """A chunk appearing in two lists is merged into one hit."""
    rrf = ReciprocalRankFusion(k=60.0)
    r1 = RetrievalResult(
        query_id="q",
        hits=[_hit("x", 0.9, strategy="dense", rank=0), _hit("nope", 0.1, rank=1)],
        strategies=["dense"],
    )
    r2 = RetrievalResult(
        query_id="q",
        hits=[_hit("x", 0.8, strategy="sparse", rank=0)],
        strategies=["sparse"],
    )
    fused = rrf.fuse([r1, r2], top_k=10)
    assert [h.chunk_id for h in fused.hits] == ["x", "nope"]
    assert len(fused.hits) == 2


def test_rrf_weighted() -> None:
    """Per-list weights scale RRF contributions."""
    rrf = ReciprocalRankFusion(k=60.0)
    r1 = RetrievalResult(
        query_id="q", hits=[_hit("a", 0.9, strategy="dense", rank=0)], strategies=["dense"]
    )
    r2 = RetrievalResult(
        query_id="q", hits=[_hit("a", 0.8, strategy="sparse", rank=0)], strategies=["sparse"]
    )
    fused = rrf.fuse([r1, r2], top_k=10, weights=[2.0, 1.0])
    a = fused.hits[0]
    assert math.isclose(a.score, 2.0 / 60.0 + 1.0 / 60.0)
    assert math.isclose(a.strategy_scores["dense"], 2.0 / 60.0)
    assert math.isclose(a.strategy_scores["sparse"], 1.0 / 60.0)


def test_rrf_empty_lists() -> None:
    rrf = ReciprocalRankFusion()
    r1 = RetrievalResult(query_id="q", hits=[], strategies=["dense"])
    r2 = RetrievalResult(query_id="q", hits=[], strategies=["sparse"])
    fused = rrf.fuse([r1, r2], top_k=10)
    assert fused.hits == []
    assert fused.strategies == ["hybrid"]
    assert fused.query_id == "q"


def test_rrf_empty_result_list() -> None:
    rrf = ReciprocalRankFusion()
    fused = rrf.fuse([], top_k=5)
    assert fused.hits == []
    assert fused.strategies == ["hybrid"]


def test_rrf_weights_mismatch_raises() -> None:
    rrf = ReciprocalRankFusion()
    r1 = RetrievalResult(query_id="q", hits=[_hit("a", 0.9)], strategies=["dense"])
    with pytest.raises(FusionError):
        rrf.fuse([r1], top_k=10, weights=[1.0, 2.0])


def test_weighted_minmax() -> None:
    wf = WeightedScoreFusion(normalize=True)
    # dense scores [0.8, 0.4] -> minmax [1.0, 0.0]
    r1 = RetrievalResult(
        query_id="q",
        hits=[
            _hit("a", 0.8, strategy="dense", rank=0),
            _hit("b", 0.4, rank=1),
        ],
        strategies=["dense"],
    )
    # sparse single score 0.6 -> minmax [1.0]
    r2 = RetrievalResult(
        query_id="q",
        hits=[_hit("a", 0.6, strategy="sparse", rank=0)],
        strategies=["sparse"],
    )
    fused = wf.fuse([r1, r2], top_k=10)
    a = next(h for h in fused.hits if h.chunk_id == "a")
    b = next(h for h in fused.hits if h.chunk_id == "b")
    assert fused.hits[0].chunk_id == "a"
    assert math.isclose(a.score, 2.0)
    assert math.isclose(a.strategy_scores["dense"], 1.0)
    assert math.isclose(a.strategy_scores["sparse"], 1.0)
    assert math.isclose(b.score, 0.0)
    assert math.isclose(b.strategy_scores["dense"], 0.0)
    assert math.isclose(a.normalized_score, 1.0)
    assert math.isclose(b.normalized_score, 0.0)


def test_weighted_raw_scores() -> None:
    """When normalize=False raw scores are weighted and summed."""
    wf = WeightedScoreFusion(normalize=False)
    r1 = RetrievalResult(
        query_id="q",
        hits=[_hit("a", 0.8, strategy="dense", rank=0)],
        strategies=["dense"],
    )
    r2 = RetrievalResult(
        query_id="q",
        hits=[_hit("a", 0.2, strategy="sparse", rank=0)],
        strategies=["sparse"],
    )
    fused = wf.fuse([r1, r2], top_k=10, weights=[1.0, 1.0])
    a = fused.hits[0]
    assert math.isclose(a.score, 0.8 + 0.2)
    assert math.isclose(a.strategy_scores["dense"], 0.8)
    assert math.isclose(a.strategy_scores["sparse"], 0.2)


def test_weighted_empty_lists() -> None:
    wf = WeightedScoreFusion()
    fused = wf.fuse(
        [
            RetrievalResult(query_id="q", hits=[], strategies=["dense"]),
            RetrievalResult(query_id="q", hits=[], strategies=["sparse"]),
        ],
        top_k=10,
    )
    assert fused.hits == []
    assert fused.strategies == ["hybrid"]
