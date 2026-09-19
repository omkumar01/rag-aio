"""Tests for explainability helpers."""

from __future__ import annotations

from rag_core.retrieval import RetrievalHit, RetrievalResult
from rag_retrieval import attach_text, explain_result


def _hit(chunk_id: str = "a", text: str | None = None) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="d",
        score=0.9,
        normalized_score=1.0,
        rank=0,
        strategy="hybrid",
        model="m",
        strategy_scores={"dense": 0.8, "sparse": 0.1},
        text=text,
        metadata={"x": 1},
    )


def test_explain_result_shape() -> None:
    result = RetrievalResult(
        query_id="q",
        hits=[_hit("a"), _hit("b")],
        strategies=["hybrid"],
        timings_ms={"hybrid": 1.2},
    )
    exp = explain_result(result)
    assert exp["query_id"] == "q"
    assert exp["strategies"] == ["hybrid"]
    assert exp["timings_ms"]["hybrid"] == 1.2
    assert exp["hits"][0]["chunk_id"] == "a"
    assert exp["hits"][0]["strategy_scores"] == {"dense": 0.8, "sparse": 0.1}
    assert exp["hits"][0]["model"] == "m"
    assert exp["hits"][0]["rank"] == 0
    assert "filters_applied" in exp["hits"][0]


def test_attach_text_mapping() -> None:
    result = RetrievalResult(query_id="q", hits=[_hit("a"), _hit("b")], strategies=["hybrid"])
    attach_text(result, {"a": "content-a", "b": "content-b"})
    assert result.hits[0].text == "content-a"
    assert result.hits[1].text == "content-b"


def test_attach_text_callable() -> None:
    result = RetrievalResult(query_id="q", hits=[_hit("a")], strategies=["hybrid"])
    attach_text(result, lambda cid: f"text-for-{cid}")
    assert result.hits[0].text == "text-for-a"


def test_attach_text_preserves_existing() -> None:
    result = RetrievalResult(
        query_id="q", hits=[_hit("a", text="already-here")], strategies=["hybrid"]
    )
    attach_text(result, {"a": "ignored"})
    assert result.hits[0].text == "already-here"


def test_attach_text_none_is_noop() -> None:
    result = RetrievalResult(query_id="q", hits=[_hit("a")], strategies=["hybrid"])
    attach_text(result, None)
    assert result.hits[0].text is None
