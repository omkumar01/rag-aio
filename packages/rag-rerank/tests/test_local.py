"""Tests for :class:`CrossEncoderReranker` and :class:`HeuristicReranker`."""

from __future__ import annotations

import asyncio

import pytest
from rag_core import Query, RetrievalHit
from rag_rerank import CrossEncoderReranker, HeuristicReranker
from rag_rerank.config import RerankConfig


def make_query(text: str = "query apple pie") -> Query:
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
        ),
        RetrievalHit(
            chunk_id="b",
            document_id="d",
            score=0.9,
            normalized_score=0.9,
            rank=1,
            strategy="dense",
            text="completely unrelated topic",
        ),
        RetrievalHit(
            chunk_id="c",
            document_id="d",
            score=0.5,
            normalized_score=0.5,
            rank=2,
            strategy="dense",
            text="cherry apple tart",
        ),
    ]


class FakeModel:
    """Stand-in for sentence-transformers CrossEncoder with call recording."""

    def __init__(self) -> None:
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        # Score by whether the query term appears in the doc text.
        out: list[float] = []
        for _query, text in pairs:
            if "apple" in text:
                out.append(0.9)
            else:
                out.append(0.1)
        return out


@pytest.mark.asyncio
async def test_cross_encoder_batch_slicing() -> None:
    fake = FakeModel()
    cfg = RerankConfig(batch_size=2)
    reranker = CrossEncoderReranker(model_name="fake", config=cfg, model_factory=lambda: fake)
    out = await reranker.rerank(make_query(), make_hits())
    assert len(fake.calls) == 2  # 3 hits -> 2 batches with batch_size=2
    assert len(fake.calls[0]) == 2
    assert len(fake.calls[1]) == 1
    assert [h.chunk_id for h in out] == ["a", "c", "b"]
    assert all(h.model == "fake" for h in out)


@pytest.mark.asyncio
async def test_cross_encoder_concurrent_safety() -> None:
    fake = FakeModel()
    reranker = CrossEncoderReranker(model_name="fake", model_factory=lambda: fake)
    results = await asyncio.gather(
        reranker.rerank(make_query(), make_hits()),
        reranker.rerank(make_query(), make_hits()),
    )
    for out in results:
        assert [h.chunk_id for h in out] == ["a", "c", "b"]
    # Both calls should share the single lazily-loaded model instance.
    assert fake is reranker._model


@pytest.mark.asyncio
async def test_heuristic_relevant_ranks_above_irrelevant() -> None:
    reranker = HeuristicReranker()
    out = await reranker.rerank(make_query(), make_hits())
    # "apple" appears in a and c; query terms are {query,apple,pie}.
    assert out[0].chunk_id in {"a", "c"}
    assert out[-1].chunk_id == "b"
    assert all(h.model == "heuristic" for h in out)


@pytest.mark.asyncio
async def test_heuristic_keyword_overlap_disabled_scores_zero() -> None:
    reranker = HeuristicReranker(keyword_overlap=False)
    out = await reranker.rerank(make_query(), make_hits())
    assert len(out) == 3
    # All scores 0.0; order preserved from input.
    assert all(h.score == 0.0 for h in out)


@pytest.mark.asyncio
async def test_heuristic_empty_candidates() -> None:
    reranker = HeuristicReranker()
    out = await reranker.rerank(make_query(), [])
    assert out == []
