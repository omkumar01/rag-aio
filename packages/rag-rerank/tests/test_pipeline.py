"""Tests for :class:`RerankPipeline` and explainability helpers."""

from __future__ import annotations

import pytest
from rag_core import Query, RetrievalHit
from rag_rerank import (
    HeuristicReranker,
    RerankConfig,
    RerankPipeline,
    explain,
    to_rerank_hits,
)


def make_query(text: str = "apple pie recipe") -> Query:
    return Query(text=text)


def make_hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id="a",
            document_id="d",
            score=0.2,
            normalized_score=0.2,
            rank=0,
            strategy="dense",
            text="apple banana",
            strategy_scores={"dense": 0.2, "sparse": 0.1},
        ),
        RetrievalHit(
            chunk_id="b",
            document_id="d",
            score=0.9,
            normalized_score=0.9,
            rank=1,
            strategy="dense",
            text="completely unrelated topic",
            strategy_scores={"dense": 0.9, "sparse": 0.05},
        ),
        RetrievalHit(
            chunk_id="c",
            document_id="d",
            score=0.5,
            normalized_score=0.5,
            rank=2,
            strategy="dense",
            text="cherry apple tart",
            strategy_scores={"dense": 0.5, "sparse": 0.4},
        ),
    ]


@pytest.mark.asyncio
async def test_pipeline_end_to_end_preserves_types_and_strategy() -> None:
    reranker = HeuristicReranker()
    pipeline = RerankPipeline(reranker, RerankConfig())
    out = await pipeline.rerank(make_query(), make_hits())
    assert isinstance(out, list)
    assert all(isinstance(h, RetrievalHit) for h in out)
    assert all(h.strategy == "reranked" for h in out)
    assert all(h.model == "heuristic" for h in out)
    # "apple" appears in a and c -> ranked above b.
    assert out[-1].chunk_id == "b"
    assert {out[0].chunk_id, out[1].chunk_id} == {"a", "c"}
    # strategy_scores preserved from the original dense retrieval.
    for h in out:
        assert "dense" in h.strategy_scores
        assert h.metadata["original_rank"] is not None


@pytest.mark.asyncio
async def test_pipeline_top_k_propagated() -> None:
    reranker = HeuristicReranker()
    pipeline = RerankPipeline(reranker, RerankConfig())
    out = await pipeline.rerank(make_query(), make_hits(), top_k=1)
    assert len(out) == 1
    assert out[0].rank == 0


@pytest.mark.asyncio
async def test_to_rerank_hits_maps_original_ranks() -> None:
    reranker = HeuristicReranker()
    pipeline = RerankPipeline(reranker, RerankConfig())
    before = make_hits()
    after = await pipeline.rerank(make_query(), before)
    records = to_rerank_hits(before, after)
    assert len(records) == len(after)
    # original ranks come from the before list.
    by_chunk = {r.chunk_id: r for r in records}
    assert by_chunk["a"].original_rank == 0
    assert by_chunk["b"].original_rank == 1
    assert by_chunk["c"].original_rank == 2
    # original scores preserved.
    assert by_chunk["a"].original_score == pytest.approx(0.2)
    # New rank reflects reranked order.
    assert records[0].rank == 0
    assert records[0].chunk_id == after[0].chunk_id
    # model propagated.
    assert all(r.model == "heuristic" for r in records)


@pytest.mark.asyncio
async def test_explain_summary() -> None:
    reranker = HeuristicReranker()
    pipeline = RerankPipeline(reranker, RerankConfig())
    before = make_hits()
    after = await pipeline.rerank(make_query(), before)
    records = to_rerank_hits(before, after)
    summary = explain(records)
    assert summary["n_results"] == 3
    assert summary["score_min"] is not None
    assert summary["score_max"] is not None
    assert summary["reranker"] == ["heuristic"]
