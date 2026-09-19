# ADR-0003: Qdrant local mode as default vector store

**Status:** Accepted | **Date:** 2026-09-19

## Context
The default deployment target is a high-performance local workstation without Docker or external services, while production deployments want a real Qdrant server.

## Decision
`qdrant-client` in embedded local mode (persistent path) is the default `VectorStore` implementation. The identical adapter points to a Qdrant server via configuration (`mode = "local" | "server"`). `docker-compose.qdrant.yml` is provided for server mode. FAISS and pgvector adapters exist behind extras.

## Consequences
- Zero-infrastructure default with a production upgrade path that is a config change, not a code change.
- Local mode is single-process: concurrent writers must funnel through one process; the adapter documents this limitation.
