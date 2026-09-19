"""Integration tests for Qdrant local (embedded) mode."""

from __future__ import annotations

import pytest
from rag_core.protocols import VectorStore
from rag_db_handler import QdrantVectorStore, VectorStoreConfig

pytestmark = pytest.mark.integration


def _v(*xs: float) -> list[float]:
    return [float(x) for x in xs]


@pytest.fixture()
async def store(tmp_path) -> QdrantVectorStore:  # type: ignore[no-untyped-def]
    path = tmp_path / "qdrant"
    cfg = VectorStoreConfig(mode="local", path=str(path), vector_size=4)
    s = QdrantVectorStore(cfg)
    # ensure collection is created for the test's lifetime
    await s.ensure_collection()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture()
def vector_store(store: QdrantVectorStore) -> VectorStore:
    return store


async def test_protocol_conformance(store: QdrantVectorStore) -> None:
    assert isinstance(store, VectorStore)


async def test_health_true(store: QdrantVectorStore) -> None:
    assert await store.health() is True


async def test_upsert_and_search_nearest_first(
    store: QdrantVectorStore, vector_store: VectorStore
) -> None:
    await vector_store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "alpha"}, "ns")
    await vector_store.upsert(
        "b", _v(0.95, 0.05, 0, 0), {"document_id": "d2", "text": "near-a"}, "ns"
    )
    await vector_store.upsert(
        "c", _v(0, 0, 1, 0), {"document_id": "d3", "text": "orthogonal"}, "ns"
    )

    hits = await vector_store.search(_v(1, 0, 0, 0), top_k=3, namespace="ns")
    assert [h.chunk_id for h in hits] == ["a", "b", "c"]
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert hits[0].document_id == "d1"
    assert hits[0].text == "alpha"
    assert hits[0].strategy == "dense"
    assert [h.rank for h in hits] == [0, 1, 2]


async def test_namespace_isolation(store: QdrantVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns1")
    await store.upsert("b", _v(1, 0, 0, 0), {"document_id": "d2", "text": "y"}, "ns2")
    hits_ns1 = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns1")
    hits_ns2 = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns2")
    assert [h.chunk_id for h in hits_ns1] == ["a"]
    assert [h.chunk_id for h in hits_ns2] == ["b"]


async def test_filters_metadata(store: QdrantVectorStore) -> None:
    await store.upsert(
        "a",
        _v(1, 0, 0, 0),
        {"document_id": "d1", "text": "x", "metadata": {"lang": "en"}},
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


async def test_delete(store: QdrantVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}, "ns")
    await store.upsert("b", _v(0, 1, 0, 0), {"document_id": "d2", "text": "y"}, "ns")
    assert await store.count("ns") == 2
    await store.delete(["a"], namespace="ns")
    assert await store.count("ns") == 1
    hits = await store.search(_v(1, 0, 0, 0), top_k=5, namespace="ns")
    assert [h.chunk_id for h in hits] == ["b"]


async def test_count_helper(store: QdrantVectorStore) -> None:
    await store.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1"}, "ns1")
    await store.upsert("b", _v(1, 0, 0, 0), {"document_id": "d2"}, "ns2")
    assert await store.count("ns1") == 1
    assert await store.count("ns2") == 1
    assert await store.count() == 2


async def test_persistence_across_instances(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "qdrant"
    cfg = VectorStoreConfig(mode="local", path=str(path), vector_size=4)
    s1 = QdrantVectorStore(cfg)
    await s1.upsert("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "hi"}, "ns")
    await s1.close()

    s2 = QdrantVectorStore(cfg)
    try:
        assert await s2.health() is True
        hits = await s2.search(_v(1, 0, 0, 0), top_k=5, namespace="ns")
        assert [h.chunk_id for h in hits] == ["a"]
    finally:
        await s2.close()


async def test_upsert_many(store: QdrantVectorStore) -> None:
    items = [
        ("a", _v(1, 0, 0, 0), {"document_id": "d1", "text": "x"}),
        ("b", _v(0, 1, 0, 0), {"document_id": "d2", "text": "y"}),
    ]
    await store.upsert_many(items, namespace="ns")
    assert await store.count("ns") == 2
