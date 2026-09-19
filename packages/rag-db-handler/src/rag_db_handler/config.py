"""Pydantic configuration models and factory functions for rag-db-handler.

Configs inherit :class:`rag_core.base.RagBaseModel` (strict, ``extra="forbid"``)
so contract drift fails loudly. Factory functions perform semantic validation
of field combinations and raise :class:`rag_core.errors.ConfigError` on invalid
configurations.

Secret references (e.g. ``api_key_ref``) are resolved *by reference* from the
environment at construction time; the resolved value is forwarded to a backend
client only and is never stored on a config object or written to logs.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from rag_core.base import RagBaseModel
from rag_core.errors import ConfigError
from rag_core.protocols import KeyValueStore, VectorStore

from .memory_store import InMemoryKeyValueStore
from .qdrant_store import QdrantVectorStore
from .sql_store import SQLDocumentStore, SQLKeyValueStore


def resolve_env_secret(ref: str) -> str:
    """Resolve a secret *by reference* from the process environment.

    The reference is an environment variable name. The resolved value is
    returned to the caller only insofar as it is forwarded to a backend client;
    it is never persisted on a configuration object or written to logs.
    """
    value = os.environ.get(ref)
    if value is None:
        raise ConfigError(f"api_key_ref '{ref}' is not set in the environment")
    return value


class VectorStoreConfig(RagBaseModel):
    """Configuration for a vector store backend.

    Local mode (default, ADR-0003) persists to ``path``; server mode connects to
    ``url`` with an optional API key looked up by reference.
    """

    backend: Literal["qdrant"] = "qdrant"
    mode: Literal["local", "server"] = "local"
    path: str | None = None
    url: str | None = None
    api_key_ref: str | None = None
    collection: str = "rag_chunks"
    vector_size: int = Field(default=384, ge=1)
    distance: Literal["cosine", "euclid", "dot"] = "cosine"

    def assert_valid(self) -> None:
        """Validate mode-specific combinations, raising :class:`ConfigError`."""
        if self.mode == "local" and not self.path:
            raise ConfigError("VectorStoreConfig in local mode requires 'path'")
        if self.mode == "server" and not self.url:
            raise ConfigError("VectorStoreConfig in server mode requires 'url'")


class SQLStoreConfig(RagBaseModel):
    """Configuration for the SQLAlchemy async store (sqlite by default)."""

    url: str = "sqlite+aiosqlite:///./data/rag.db"
    pool_size: int = Field(default=5, ge=0)
    max_overflow: int = Field(default=10, ge=0)
    echo: bool = False


class KeyValueStoreConfig(RagBaseModel):
    """Configuration for a key/value store backend."""

    backend: Literal["sqlite", "memory"] = "sqlite"
    path: str | None = None
    namespace: str = "default"


# --- factories ----------------------------------------------------------------


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Build a :class:`VectorStore` from configuration.

    Only the ``qdrant`` backend is supported. Local mode is the default
    (ADR-0003); server mode is opt-in via ``mode="server"``. The constructed
    adapter resolves ``api_key_ref`` from the environment itself, so a missing
    secret fails fast with :class:`ConfigError`.
    """
    if config.backend != "qdrant":
        raise ConfigError(f"unsupported vector store backend: {config.backend}")
    config.assert_valid()
    return QdrantVectorStore(config)


def create_sql_store(config: SQLStoreConfig) -> SQLDocumentStore:
    """Build the SQL :class:`DocumentStore` from configuration."""
    if not config.url:
        raise ConfigError("SQLStoreConfig requires a non-empty 'url'")
    return SQLDocumentStore(config)


def create_kv_store(config: KeyValueStoreConfig) -> KeyValueStore:
    """Build a :class:`KeyValueStore` from configuration.

    ``memory`` returns an in-process dict-backed store; ``sqlite`` builds a
    :class:`SQLKeyValueStore`. When ``path`` is omitted for sqlite the default
    repository-local database URL is used.
    """
    if config.backend == "memory":
        return InMemoryKeyValueStore(namespace=config.namespace)
    if config.backend == "sqlite":
        url = f"sqlite+aiosqlite:///{config.path}" if config.path else SQLStoreConfig().url
        return SQLKeyValueStore(SQLStoreConfig(url=url), namespace=config.namespace)
    raise ConfigError(f"unsupported key-value store backend: {config.backend}")


__all__ = [
    "KeyValueStoreConfig",
    "SQLStoreConfig",
    "VectorStoreConfig",
    "create_kv_store",
    "create_sql_store",
    "create_vector_store",
    "resolve_env_secret",
]
