"""Tests for :func:`rag_retrieval.supports_sparse` feature detection."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from rag_db_handler.memory_store import InMemoryVectorStore
from rag_retrieval import SparseSearchStore, supports_sparse

pytestmark = pytest.mark.unit


class _SparseCapableStore:
    """Minimal structural store: only ``search_sparse`` is required."""

    async def search_sparse(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        namespace: str | None = None,
    ) -> list[object]:
        return []


class _DenseOnlyStore:
    """Structural store without any sparse capability."""

    async def search(self, vector: Sequence[float], top_k: int) -> list[object]:
        return []


def test_sparse_capable_store_is_detected() -> None:
    store = _SparseCapableStore()
    assert supports_sparse(store)
    assert isinstance(store, SparseSearchStore)


def test_dense_only_store_is_not_detected() -> None:
    assert not supports_sparse(_DenseOnlyStore())
    assert not supports_sparse(InMemoryVectorStore())


def test_arbitrary_object_is_not_detected() -> None:
    assert not supports_sparse(object())
