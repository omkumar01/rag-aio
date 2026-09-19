# rag-db-handler

Capability-specific persistence abstractions for rag-aio: `SQLStore`, `DocumentStore`,
`KeyValueStore`, `VectorStore`, `SparseStore`, `CacheStore`, `SearchStore`. Ships a Qdrant
adapter (embedded local mode by default, server mode via config — ADR-0003) and async
SQL via SQLAlchemy 2.x (sqlite default, PostgreSQL behind the `postgres` extra). FAISS and
pgvector adapters behind extras.
