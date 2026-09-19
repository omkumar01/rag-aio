"""Tests for rag-db-handler config models and factory functions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_core.errors import ConfigError
from rag_db_handler import (
    KeyValueStoreConfig,
    SQLStoreConfig,
    VectorStoreConfig,
    create_kv_store,
    create_sql_store,
    create_vector_store,
)
from rag_db_handler.memory_store import InMemoryKeyValueStore
from rag_db_handler.sql_store import SQLDocumentStore, SQLKeyValueStore

pytestmark = pytest.mark.unit

# --- VectorStoreConfig --------------------------------------------------------


def test_vector_store_config_defaults() -> None:
    cfg = VectorStoreConfig(path="/tmp/rag-vec")
    assert cfg.backend == "qdrant"
    assert cfg.mode == "local"
    assert cfg.collection == "rag_chunks"
    assert cfg.vector_size == 384
    assert cfg.distance == "cosine"
    assert cfg.api_key_ref is None


def test_vector_store_config_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        VectorStoreConfig(path="/tmp/rag-vec", bogus="nope")  # type: ignore[call-arg]


def test_create_vector_store_invalid_mode_combo() -> None:
    cfg = VectorStoreConfig(mode="local", path=None)  # local without path
    with pytest.raises(ConfigError):
        create_vector_store(cfg)


def test_create_vector_store_server_without_url() -> None:
    cfg = VectorStoreConfig(mode="server", url=None)
    with pytest.raises(ConfigError):
        create_vector_store(cfg)


def test_create_vector_store_resolves_api_key_ref_no_env() -> None:
    cfg = VectorStoreConfig(
        mode="server", url="http://localhost:6333", api_key_ref="RAG_QDRANT_KEY_NOT_SET"
    )
    with pytest.raises(ConfigError):
        create_vector_store(cfg)


def test_create_vector_store_api_key_value_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_KEY", "super-secret-value-123")
    cfg = VectorStoreConfig(
        mode="server", url="https://localhost:6333", api_key_ref="RAG_QDRANT_KEY"
    )
    store = create_vector_store(cfg)
    # The resolved secret value must never appear in the store's repr.
    assert "super-secret-value-123" not in repr(store)
    assert "super-secret-value-123" not in str(store)


def test_create_vector_store_local_returns_qdrant(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = VectorStoreConfig(mode="local", path=str(tmp_path / "qdrant"))
    store = create_vector_store(cfg)
    assert store.__class__.__name__ == "QdrantVectorStore"


# --- SQLStoreConfig -----------------------------------------------------------


def test_sql_store_config_defaults() -> None:
    cfg = SQLStoreConfig()
    assert cfg.url.startswith("sqlite+aiosqlite")
    assert cfg.pool_size == 5
    assert cfg.max_overflow == 10
    assert cfg.echo is False


def test_create_sql_store_requires_url() -> None:
    cfg = SQLStoreConfig(url="")
    with pytest.raises(ConfigError):
        create_sql_store(cfg)


def test_create_sql_store_returns_sql_document_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = SQLStoreConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'd.db'}")
    store = create_sql_store(cfg)
    assert isinstance(store, SQLDocumentStore)


# --- KeyValueStoreConfig / create_kv_store ------------------------------------


def test_create_kv_store_memory() -> None:
    cfg = KeyValueStoreConfig(backend="memory", namespace="kvtest")
    store = create_kv_store(cfg)
    assert isinstance(store, InMemoryKeyValueStore)


def test_create_kv_store_sqlite_uses_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = tmp_path / "kv.sqlite3"
    cfg = KeyValueStoreConfig(backend="sqlite", path=str(db), namespace="kvtest")
    store = create_kv_store(cfg)
    assert isinstance(store, SQLKeyValueStore)
