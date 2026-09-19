"""Tests for HybridRetriever (concurrent execution + fusion)."""

from __future__ import annotations

import asyncio

import pytest
from rag_core.errors import RetrievalError
from rag_core.protocols import HybridRetriever as HybridRetrieverProtocol
from rag_core.protocols import Retriever
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit, RetrievalResult
from rag_retrieval import (
    HybridRetriever,
    ReciprocalRankFusion,
    RetrievalConfig,
    WeightedScoreFusion,
)


def _hit(chunk_id: str, score: float, strategy: str = "dense", rank: int = 0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="d",
        score=score,
        normalized_score=score,
        rank=rank,
        strategy=strategy,
        strategy_scores={strategy: score},
    )


def _result(strategy: str, hits: list[RetrievalHit], query_id: str = "q") -> RetrievalResult:
    return RetrievalResult(
        query_id=query_id,
        hits=hits,
        strategies=[strategy],
        timings_ms={strategy: 1.0},
    )


class _FakeRetriever:
    def __init__(self, strategy: str, hits: list[RetrievalHit]) -> None:
        self.strategy = strategy
        self._hits = hits

    async def retrieve(self, query: Query) -> RetrievalResult:
        await asyncio.sleep(0)
        return _result(self.strategy, list(self._hits), query_id=query.id)


class _FailingRetriever:
    strategy: str = "dense"

    async def retrieve(self, query: Query) -> RetrievalResult:
        await asyncio.sleep(0)
        raise RuntimeError("boom")


async def test_hybrid_fused_ordering_rrf() -> None:
    r1 = _FakeRetriever("dense", [_hit("a", 0.9, "dense", rank=0), _hit("b", 0.5, rank=1)])
    r2 = _FakeRetriever("sparse", [_hit("a", 0.8, "sparse", rank=0)])
    hy = HybridRetriever([r1, r2], ReciprocalRankFusion(), RetrievalConfig())

    result = await hy.retrieve(Query(text="x"))
    assert isinstance(result, RetrievalResult)
    assert result.strategies == ["hybrid"]
    # a appears in both lists (ranks 0 and 0) -> highest RRF score.
    assert result.hits[0].chunk_id == "a"
    assert result.hits[1].chunk_id == "b"
    assert result.hits[0].strategy == "hybrid"
    assert "hybrid" in result.timings_ms


async def test_hybrid_partial_failure_tolerated() -> None:
    failing = _FailingRetriever()  # strategy = "dense"
    surviving = _FakeRetriever("sparse", [_hit("z", 0.9, "sparse", rank=0)])
    hy = HybridRetriever([failing, surviving], ReciprocalRankFusion(), RetrievalConfig())

    result = await hy.retrieve(Query(text="x"))
    assert result.hits[0].chunk_id == "z"
    assert "dense_failed" in result.timings_ms
    assert "sparse" in result.timings_ms


async def test_hybrid_all_fail_raises() -> None:
    hy = HybridRetriever(
        [_FailingRetriever(), _FailingRetriever()], ReciprocalRankFusion(), RetrievalConfig()
    )
    with pytest.raises(RetrievalError):
        await hy.retrieve(Query(text="x"))


async def test_hybrid_weighted_fusion() -> None:
    r1 = _FakeRetriever("dense", [_hit("a", 0.9, "dense", rank=0), _hit("b", 0.4, rank=1)])
    r2 = _FakeRetriever("sparse", [_hit("a", 0.6, "sparse", rank=0)])
    cfg = RetrievalConfig(fusion="weighted", dense_weight=1.0, sparse_weight=1.0)
    hy = HybridRetriever([r1, r2], WeightedScoreFusion(normalize=True), cfg)

    result = await hy.retrieve(Query(text="x"))
    assert result.hits[0].chunk_id == "a"


async def test_hybrid_is_protocol() -> None:
    r = _FakeRetriever("dense", [_hit("a", 0.9, "dense", rank=0)])
    hy = HybridRetriever([r], ReciprocalRankFusion(), RetrievalConfig())
    assert isinstance(hy, HybridRetrieverProtocol)
    assert isinstance(r, Retriever)
