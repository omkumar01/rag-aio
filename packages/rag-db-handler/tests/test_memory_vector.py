"""Tests for the in-memory vector store (pure-Python, numpy-free)."""

from __future__ import annotations

import math

import pytest
from rag_core.protocols import VectorStore
from rag_db_handler import InMemoryVectorStore
from rag_db_handler.memory_store import InMemoryKeyValueStore


def _v(*xs: float) -> list[float]:
    return [float(x) for x in xs]


@pytest.fixture()
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore(vector_size=4)


@pytest.mark.unit
async def test_protocol_conformance(store: InMemoryVectorStore) -> None:
    assert isinstance(store, VectorStore)


@pytest.mark.unit
async def test_upsert_and_search_top_k_ordering(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "alpha"}, "ns")
    await store.upsert("b", _v(0, 1, 0, 0), {"document_id": "d2", "text": "beta"}, "ns")
    await store.upsert("c", _v(0.9, 0.1, 0, 0), {"document_id": "d3", "text": "near"}, "ns")

    hits = await store.search(_v(1, 0, 0, 0), top_k=2, namespace="ns")
    assert len(hits) == 2
    # nearest (a, cosine 1.0) then (c, cosine ~0.949), never (b)
    assert hits[0].chunk_id == "a"
    assert hits[1].chunk_id == "c"
    assert hits[0].score > hits[1].score


@pytest.mark.unit
async def test_namespace_isolation(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns1")
    await store.upsert("b", _v(1, 0, 0, 0), {"document_id": "d2", "text": "y"}, "ns2")

    hits_ns1 = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns1")
    hits_ns2 = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns2")
    assert [h.chunk_id for h in hits_ns1] == ["a"]
    assert [h.chunk_id for h in hits_ns2] == ["b"]


@pytest.mark.unit
async def test_filters_scalars(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    await store.upsert("b", _v(1, 0, 0, 0), {"document_id": "d2", "text": "y"}, "ns")

    hits = await store.search(
        _v(1, 0, 0, 0), top_k=5, filters={"document_id": "d2"}, namespace="ns"
    )
    assert [h.chunk_id for h in hits] == ["b"]


@pytest.mark.unit
async def test_filters_metadata_key(store: InMemoryVectorStore) -> None:
    await store.upsert(
        "a",
        _v(1, 0, 0, 0),
        {"document_id": "d1", "text": "x", "metadata": {"lang": "en", "ver": 1}},
        "ns",
    )
    await store.upsert(
        "b",
        _v(1, 0, 0, 0),
        {"document_id": "d2", "text": "y", "metadata": {"lang": "fr"}},
        "ns",
    )
    hits = await store.search(
        _v(1, 0, 0, 0), top_k=5, filters={"metadata.lang": "en"}, namespace="ns"
    )
    assert [h.chunk_id for h in hits] == ["a"]
    assert hits[0].metadata.get("lang") == "en"


@pytest.mark.unit
async def test_rank_by_position(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    await store.upsert("b", _v(0.9, 0.1, 0, 0), {"document_id": "d2", "text": "y"}, "ns")
    hits = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns")
    assert [h.rank for h in hits] == [0, 1]


@pytest.mark.unit
async def test_delete(store: InMemoryVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    await store.upsert("b", _v(0, 1, 0, 0), {"document_id": "d2", "text": "y"}, "ns")
    await store.delete(["a"], namespace="ns")
    hits = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns")
    assert [h.chunk_id for h in hits] == ["b"]


@pytest.mark.unit
async def test_count_and_health(store: InMemoryVectorStore) -> None:
    assert await store.health() is True
    assert store.count("ns") == 0
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    assert store.count("ns") == 1
    assert store.count("other") == 0


@pytest.mark.unit
async def test_cosine_score_range_and_normalized() -> None:
    store = InMemoryVectorStore(vector_size=3)
    await store.upsert("a", _v(1, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    # orthogonal vector -> cosine 0
    hits = await store.search(_v(0, 1, 0), top_k=1, namespace="ns")
    assert math.isclose(hits[0].score, 0.0, abs_tol=1e-9)
    assert 0.0 <= hits[0].normalized_score <= 1.0


@pytest.mark.unit
async def test_in_memory_kv_roundtrip() -> None:
    store = InMemoryKeyValueStore(namespace="kv")
    assert await store.get("k") is None
    await store.set("k", b"value")
    assert await store.get("k") == b"value"
    await store.delete("k")
    assert await store.get("k") is None
    assert await store.health() is True
