"""Tests for ``load_local_services`` parameter resolution (no heavy backends).

The heavy lazily-imported backends (fastembed embedders, the local Qdrant
vector store, the SQL document store) are patched with fakes so these tests
exercise *only* parameter resolution and warning behavior.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
from rag_orchestrator.services import load_local_services


def _load(**kwargs: object) -> MagicMock:
    """Call load_local_services with heavy backends patched; return the SQL mock."""
    with (
        patch("rag_embedder.embedding.FastEmbedDense", MagicMock()),
        patch("rag_embedder.embedding.FastEmbedSparse", MagicMock()),
        patch("rag_db_handler.config.create_vector_store", MagicMock(return_value=MagicMock())),
        patch(
            "rag_db_handler.config.create_sql_store", MagicMock(return_value=MagicMock())
        ) as sql_mock,
    ):
        load_local_services(**kwargs)  # type: ignore[arg-type]
    return sql_mock


def _sql_url(sql_mock: MagicMock) -> str:
    (config,), _kwargs = sql_mock.call_args
    return str(config.url)


def test_document_db_url_wins_over_db_url() -> None:
    sql_mock = _load(
        db_url="sqlite+aiosqlite:///./old.db",
        document_db_url="sqlite+aiosqlite:///./new.db",
    )
    assert _sql_url(sql_mock) == "sqlite+aiosqlite:///./new.db"


def test_document_db_url_alone_is_used_and_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sql_mock = _load(document_db_url="sqlite+aiosqlite:///./doc.db")
    assert _sql_url(sql_mock) == "sqlite+aiosqlite:///./doc.db"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_db_url_only_emits_deprecation_warning_and_still_works() -> None:
    with pytest.warns(DeprecationWarning, match="db_url is deprecated"):
        sql_mock = _load(db_url="sqlite+aiosqlite:///./legacy.db")
    assert _sql_url(sql_mock) == "sqlite+aiosqlite:///./legacy.db"


def test_default_db_url_emits_deprecation_warning() -> None:
    """Relying on the implicit ``db_url`` default is deprecated too."""
    with pytest.warns(DeprecationWarning, match="db_url is deprecated"):
        sql_mock = _load()
    assert _sql_url(sql_mock) == "sqlite+aiosqlite:///./data/rag.db"


class _SparseCapableStore:
    """Fake store implementing the ``SparseSearchStore`` protocol shape."""

    async def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        namespace: str | None = None,
    ) -> list[object]:
        return []


class _DenseOnlyStore:
    """Fake store without any sparse capability."""

    async def search(self, vector: list[float], top_k: int) -> list[object]:
        return []


def _load_with_store(store: object) -> object:
    """Call load_local_services with a specific vector store fake."""
    with pytest.warns(DeprecationWarning, match="db_url is deprecated"):
        return _load_returning_services(store)


def _load_returning_services(store: object) -> object:
    with (
        patch("rag_embedder.embedding.FastEmbedDense", MagicMock()),
        patch("rag_embedder.embedding.FastEmbedSparse", MagicMock()),
        patch("rag_db_handler.config.create_vector_store", MagicMock(return_value=store)),
        patch("rag_db_handler.config.create_sql_store", MagicMock(return_value=MagicMock())),
    ):
        return load_local_services()  # type: ignore[arg-type]


def test_sparse_capable_store_wires_sparse_into_hybrid() -> None:
    services = _load_with_store(_SparseCapableStore())
    (hybrid,) = services.retrievers  # type: ignore[attr-defined]
    assert len(hybrid._retrievers) == 2
    strategies = [getattr(r, "strategy", "?") for r in hybrid._retrievers]
    assert strategies == ["dense", "sparse"]


def test_dense_only_store_keeps_hybrid_dense_only() -> None:
    services = _load_with_store(_DenseOnlyStore())
    (hybrid,) = services.retrievers  # type: ignore[attr-defined]
    assert len(hybrid._retrievers) == 1
    assert getattr(hybrid._retrievers[0], "strategy", "?") == "dense"
