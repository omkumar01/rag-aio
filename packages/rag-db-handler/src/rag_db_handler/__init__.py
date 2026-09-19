"""rag-db-handler: unified persistence abstractions and store adapters.

Capability-specific persistence for rag-aio: a Qdrant vector store adapter
(embedded local mode by default, server mode via config -- ADR-0003), async SQL
backed document and key/value stores (sqlite by default, PostgreSQL behind the
``postgres`` extra), and in-memory stores used for unit tests and dev fallback.
"""

from __future__ import annotations

from .config import (
    KeyValueStoreConfig,
    SQLStoreConfig,
    VectorStoreConfig,
    create_kv_store,
    create_sql_store,
    create_vector_store,
    resolve_env_secret,
)
from .health import check_health
from .memory_store import InMemoryKeyValueStore, InMemoryVectorStore
from .qdrant_store import QdrantVectorStore
from .serialization import document_to_row_fields, row_to_document
from .sql_store import SQLDocumentStore, SQLKeyValueStore, SQLStore

__version__ = "0.1.0"

__all__ = [
    "InMemoryKeyValueStore",
    "InMemoryVectorStore",
    "KeyValueStoreConfig",
    "QdrantVectorStore",
    "SQLDocumentStore",
    "SQLKeyValueStore",
    "SQLStore",
    "SQLStoreConfig",
    "VectorStoreConfig",
    "check_health",
    "create_kv_store",
    "create_sql_store",
    "create_vector_store",
    "document_to_row_fields",
    "resolve_env_secret",
    "row_to_document",
]
