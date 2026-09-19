"""Tests for DenseRetriever using the in-memory vector store."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from rag_core.protocols import Retriever
from rag_core.queries import Query
from rag_db_handler.memory_store import InMemoryVectorStore
from rag_retrieval import DenseRetriever, RetrievalConfig

_VEC_A = [3.0, 0.0, 0.0]
_VEC_B = [2.0, 1.0, 0.0]
_VEC_C = [0.0, 0.0, 3.0]


class _MockEmbedder:
    """Deterministic dense embedder mapping query text to a vector."""

    model_name = "test-embedder"

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._mapping[text] for text in texts]


async def _populate(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _VEC_A, {"chunk_id": "a", "document_id": "d1", "text": "alpha"})
    await store.upsert("b", _VEC_B, {"chunk_id": "b", "document_id": "d1", "text": "beta"})
    await store.upsert("c", _VEC_C, {"chunk_id": "c", "document_id": "d2", "text": "gamma"})


async def test_dense_topk_ordering() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await _populate(store)
    embedder = _MockEmbedder({"what is ai": _VEC_A})
    config = RetrievalConfig()
    retriever = DenseRetriever(store, embedder, config)

    query = Query(text="what is ai")
    result = await retriever.retrieve(query)

    assert [h.chunk_id for h in result.hits] == ["a", "b", "c"]
    assert result.strategies == ["dense"]
    assert result.hits[0].score == pytest.approx(1.0)
    assert result.hits[0].model == "test-embedder"
    assert result.hits[0].strategy == "dense"
    assert "dense" in result.timings_ms
    assert result.timings_ms["dense"] >= 0.0


async def test_dense_filters_passthrough() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await store.upsert(
        "a",
        _VEC_A,
        {"chunk_id": "a", "document_id": "d1", "text": "alpha", "metadata": {"category": "news"}},
    )
    await store.upsert(
        "b",
        _VEC_B,
        {"chunk_id": "b", "document_id": "d1", "text": "beta", "metadata": {"category": "sports"}},
    )
    embedder = _MockEmbedder({"q": _VEC_A})
    retriever = DenseRetriever(store, embedder, RetrievalConfig())

    query = Query(text="q", filters={"metadata.category": "news"})
    result = await retriever.retrieve(query)
    assert [h.chunk_id for h in result.hits] == ["a"]


async def test_dense_namespace_passthrough() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await store.upsert(
        "a", _VEC_A, {"chunk_id": "a", "document_id": "d1", "text": "alpha"}, namespace="ns1"
    )
    await store.upsert("b", _VEC_B, {"chunk_id": "b", "document_id": "d1", "text": "beta"})
    embedder = _MockEmbedder({"q": _VEC_A})
    retriever = DenseRetriever(store, embedder, RetrievalConfig())

    query = Query(text="q", namespace="ns1")
    result = await retriever.retrieve(query)
    assert [h.chunk_id for h in result.hits] == ["a"]


async def test_dense_score_threshold() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await _populate(store)
    embedder = _MockEmbedder({"q": _VEC_A})
    retriever = DenseRetriever(store, embedder, RetrievalConfig(score_threshold=0.9))

    query = Query(text="q")
    result = await retriever.retrieve(query)
    # a cosine == 1.0 qualifies; b (~0.894) and c (0.0) do not.
    assert [h.chunk_id for h in result.hits] == ["a"]


async def test_dense_is_retriever() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await _populate(store)
    embedder = _MockEmbedder({"q": _VEC_A})
    retriever = DenseRetriever(store, embedder, RetrievalConfig())
    assert isinstance(retriever, Retriever)
