"""Tests for the shared :func:`rerank_candidates` wrapper logic."""

from __future__ import annotations

import pytest
from rag_core import RerankError, RetrievalHit
from rag_rerank import rerank_candidates
from rag_rerank.config import RerankConfig as RConfig


def make_hit(
    chunk_id: str, score: float, rank: int, text: str = "", strategy: str = "dense"
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        score=score,
        normalized_score=score,
        rank=rank,
        strategy=strategy,
        text=text,
        strategy_scores={"dense": score},
    )


def scorer_from_map(scores: dict[str, float]):
    """Return a scorer whose score for a text equals scores[text]."""

    async def scorer(query: str, texts: list[str]) -> list[float]:
        return [scores.get(t, 0.0) for t in texts]

    return scorer


@pytest.mark.asyncio
async def test_orders_by_score_desc() -> None:
    cfg = RConfig()
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=0.9, rank=1, text="beta"),
        make_hit("c", score=0.5, rank=2, text="gamma"),
    ]
    scorer = scorer_from_map({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert [h.chunk_id for h in out] == ["b", "c", "a"]
    assert [h.score for h in out] == [0.9, 0.5, 0.2]
    assert [h.rank for h in out] == [0, 1, 2]


@pytest.mark.asyncio
async def test_preserves_original_score_and_rank() -> None:
    cfg = RConfig()
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=0.9, rank=1, text="beta"),
    ]
    scorer = scorer_from_map({"alpha": 0.9, "beta": 0.2})
    out = await rerank_candidates("q", hits, scorer, cfg)
    # New order is a (0.9) then b (0.2); original signal preserved in metadata.
    top = out[0]
    assert top.chunk_id == "a"
    assert top.metadata["original_score"] == 0.2
    assert top.metadata["original_rank"] == 0
    second = out[1]
    assert second.chunk_id == "b"
    assert second.metadata["original_score"] == 0.9
    assert second.metadata["original_rank"] == 1


@pytest.mark.asyncio
async def test_dedup_by_chunk_id() -> None:
    cfg = RConfig()
    hits = [
        make_hit("a", score=0.1, rank=0, text="x"),
        make_hit("a", score=0.2, rank=1, text="x"),  # duplicate
        make_hit("b", score=0.9, rank=2, text="y"),
    ]
    scorer = scorer_from_map({"x": 0.5, "y": 0.9})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert [h.chunk_id for h in out] == ["b", "a"]
    # First occurrence kept (original score 0.1).
    x_hit = next(h for h in out if h.chunk_id == "a")
    assert x_hit.metadata["original_score"] == 0.1


@pytest.mark.asyncio
async def test_dedup_disabled_keeps_duplicates() -> None:
    cfg = RConfig(dedup=False)
    hits = [
        make_hit("a", score=0.1, rank=0, text="x"),
        make_hit("a", score=0.2, rank=1, text="x"),
    ]
    scorer = scorer_from_map({"x": 0.5})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_score_threshold() -> None:
    cfg = RConfig(score_threshold=0.5)
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=0.9, rank=1, text="beta"),
        make_hit("c", score=0.5, rank=2, text="gamma"),
    ]
    scorer = scorer_from_map({"alpha": 0.1, "beta": 0.9, "gamma": 0.4})
    out = await rerank_candidates("q", hits, scorer, cfg)
    chunks = [h.chunk_id for h in out]
    assert chunks == ["b"]


@pytest.mark.asyncio
async def test_top_k() -> None:
    cfg = RConfig(top_k=2)
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=0.9, rank=1, text="beta"),
        make_hit("c", score=0.5, rank=2, text="gamma"),
    ]
    scorer = scorer_from_map({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert [h.chunk_id for h in out] == ["b", "c"]


@pytest.mark.asyncio
async def test_max_candidates_raises() -> None:
    cfg = RConfig(max_candidates=2)
    hits = [
        make_hit("a", score=0.1, rank=0, text="a"),
        make_hit("b", score=0.2, rank=1, text="b"),
        make_hit("c", score=0.3, rank=2, text="c"),
    ]
    scorer = scorer_from_map({"a": 0.1, "b": 0.2, "c": 0.3})
    with pytest.raises(RerankError):
        await rerank_candidates("q", hits, scorer, cfg)


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_list() -> None:
    cfg = RConfig()
    scorer = scorer_from_map({})
    out = await rerank_candidates("q", [], scorer, cfg)
    assert out == []


@pytest.mark.asyncio
async def test_normalize_minmax_bounds() -> None:
    cfg = RConfig(normalize="minmax")
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=0.9, rank=1, text="beta"),
        make_hit("c", score=0.5, rank=2, text="gamma"),
    ]
    scorer = scorer_from_map({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert all(0.0 <= h.score <= 1.0 for h in out)
    by_id = {h.chunk_id: h.score for h in out}
    assert by_id["b"] == pytest.approx(1.0)
    assert by_id["a"] == pytest.approx(0.0)
    assert by_id["c"] == pytest.approx((0.5 - 0.2) / (0.9 - 0.2))


@pytest.mark.asyncio
async def test_normalize_sigmoid_in_range() -> None:
    cfg = RConfig(normalize="sigmoid")
    hits = [
        make_hit("a", score=0.2, rank=0, text="alpha"),
        make_hit("b", score=5.0, rank=1, text="beta"),
        make_hit("c", score=-3.0, rank=2, text="gamma"),
    ]
    scorer = scorer_from_map({"alpha": 0.2, "beta": 5.0, "gamma": -3.0})
    out = await rerank_candidates("q", hits, scorer, cfg)
    assert all(0.0 < h.score < 1.0 for h in out)
    # Higher raw score -> higher sigmoid score.
    by_id = {h.chunk_id: h.score for h in out}
    assert by_id["b"] > by_id["a"] > by_id["c"]


@pytest.mark.asyncio
async def test_batch_slicing_calls_scorer_per_batch() -> None:
    cfg = RConfig(batch_size=2)
    hits = [
        make_hit(c, score=0.1, rank=0, text=txt)
        for c, txt in zip("abcd", ["a", "b", "c", "d"], strict=True)
    ]
    calls: list[list[str]] = []

    async def scorer(query: str, texts: list[str]) -> list[float]:
        calls.append(list(texts))
        return [0.1 for _ in texts]

    await rerank_candidates("q", hits, scorer, cfg)
    assert calls == [["a", "b"], ["c", "d"]]
