"""Unit tests for CompositeVectorStore: fan-out search, write routing, counts."""

from __future__ import annotations

from typing import Any

import pytest
from rag_db_handler.memory_store import InMemoryVectorStore

from web_example.services.collections import CompositeVectorStore


@pytest.fixture
def base() -> InMemoryVectorStore:
    return InMemoryVectorStore(vector_size=4)


@pytest.fixture
def composite(base: InMemoryVectorStore) -> CompositeVectorStore:
    return CompositeVectorStore(base, primary_ns="user_1", search_ns=("user_1", "global"))


async def test_upsert_routes_to_primary_namespace(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await composite.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "c1", "document_id": "d1"})
    assert base.count("user_1") == 1
    assert base.count("global") == 0


async def test_delete_routes_to_primary_namespace(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await base.upsert("c1", [1.0, 0.0, 0.0, 0.0], {}, namespace="user_1")
    await base.upsert("c1", [0.0, 1.0, 0.0, 0.0], {}, namespace="global")
    await composite.delete(["c1"])
    assert base.count("user_1") == 0
    assert base.count("global") == 1


async def test_search_merges_namespaces_and_tags_pool(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await base.upsert(
        "own", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "own", "document_id": "d1"}, namespace="user_1"
    )
    await base.upsert(
        "shared",
        [1.0, 0.0, 0.0, 0.0],
        {"chunk_id": "shared", "document_id": "d2"},
        namespace="global",
    )
    # Lives in a namespace outside the view: must never come back.
    await base.upsert(
        "other",
        [1.0, 0.0, 0.0, 0.0],
        {"chunk_id": "other", "document_id": "d3"},
        namespace="user_2",
    )
    hits = await composite.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert {hit.chunk_id for hit in hits} == {"own", "shared"}
    assert {hit.chunk_id: hit.metadata.get("pool") for hit in hits} == {
        "own": "user_1",
        "shared": "global",
    }


async def test_search_dedupes_preferring_higher_score(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await base.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "c1"}, namespace="user_1")
    # Same chunk id in the shared pool with a lower score: user copy must win.
    await base.upsert("c1", [0.5, 0.5, 0.0, 0.0], {"chunk_id": "c1"}, namespace="global")
    hits = await composite.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(hits) == 1
    assert hits[0].metadata.get("pool") == "user_1"


async def test_search_orders_by_score_and_truncates_to_top_k(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await base.upsert("low", [0.6, 0.8, 0.0, 0.0], {"chunk_id": "low"}, namespace="user_1")
    await base.upsert("high", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "high"}, namespace="global")
    await base.upsert("mid", [0.8, 0.6, 0.0, 0.0], {"chunk_id": "mid"}, namespace="user_1")
    hits = await composite.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert [hit.chunk_id for hit in hits] == ["high", "mid"]
    assert [hit.rank for hit in hits] == [0, 1]


async def test_search_respects_filters(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    await base.upsert(
        "a", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "a", "document_id": "d1"}, namespace="user_1"
    )
    await base.upsert(
        "b", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "b", "document_id": "d2"}, namespace="user_1"
    )
    hits = await composite.search([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"document_id": "d2"})
    assert [hit.chunk_id for hit in hits] == ["b"]


async def test_search_zero_top_k_returns_empty(composite: CompositeVectorStore) -> None:
    assert await composite.search([1.0, 0.0, 0.0, 0.0], top_k=0) == []


async def test_health_delegates_to_base(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    assert await composite.health() is True


async def test_count_sums_search_namespaces(
    base: InMemoryVectorStore, composite: CompositeVectorStore
) -> None:
    payload: dict[str, Any] = {}
    await base.upsert("a", [1.0, 0.0, 0.0, 0.0], payload, namespace="user_1")
    await base.upsert("b", [0.0, 1.0, 0.0, 0.0], payload, namespace="user_1")
    await base.upsert("c", [0.0, 0.0, 1.0, 0.0], payload, namespace="global")
    await base.upsert("d", [0.0, 0.0, 0.0, 1.0], payload, namespace="user_2")
    assert await composite.count() == 3
    assert await composite.count("global") == 1
