"""Tests for :mod:`rag_context.expansion`."""

from __future__ import annotations

from collections.abc import Sequence

from rag_context import expand_with_neighbors
from rag_core.retrieval import RetrievalHit


def _hit(chunk_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc",
        score=0.9,
        normalized_score=0.9,
        rank=0,
        strategy="dense",
        text="original text",
    )


def test_inserts_neighbors_up_to_window() -> None:
    def provider(chunk_id: str) -> Sequence[tuple[str, str, int]]:
        if chunk_id == "a":
            return [("n1", "neighbor one", 0), ("n2", "neighbor two", 1)]
        return []

    out = expand_with_neighbors([_hit("a")], provider, window=2)
    ids = [h.chunk_id for h in out]
    assert ids == ["a", "n1", "n2"]
    # Neighbors are annotated in metadata with their in-document index.
    n1 = out[1]
    assert n1.metadata.get("is_neighbor") is True
    assert n1.metadata.get("neighbor_index") == 0
    n2 = out[2]
    assert n2.metadata.get("neighbor_index") == 1
    assert n1.parent_chunk_id == "a"
    # Inherited document and score.
    assert n1.document_id == "doc"
    assert n1.score == 0.9


def test_no_duplicate_chunks() -> None:
    def provider(chunk_id: str) -> Sequence[tuple[str, str, int]]:
        return [("n1", "neighbor text", 0)]

    hits = [_hit("a"), _hit("n1")]  # n1 already selected
    out = expand_with_neighbors(hits, provider, window=2)
    ids = [h.chunk_id for h in out]
    assert ids.count("n1") == 1
    # n1 is after a, and not re-added as a neighbor of a.
    assert ids == ["a", "n1"]


def test_window_limits_neighbor_count() -> None:
    def provider(chunk_id: str) -> Sequence[tuple[str, str, int]]:
        if chunk_id == "a":
            return [("n1", "one", 0), ("n2", "two", 1), ("n3", "three", 2)]
        return []

    out = expand_with_neighbors([_hit("a")], provider, window=1)
    assert [h.chunk_id for h in out] == ["a", "n1"]


def test_empty_text_neighbor_skipped() -> None:
    def provider(chunk_id: str) -> Sequence[tuple[str, str, int]]:
        return [("n1", "   ", 0), ("n2", "real neighbor", 1)]

    out = expand_with_neighbors([_hit("a")], provider, window=5)
    assert [h.chunk_id for h in out] == ["a", "n2"]
