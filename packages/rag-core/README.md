# rag-core

Canonical domain models (`Document`, `Chunk`, `RetrievalHit`, `Context`, `GenerationResult`,
...) and runtime-checkable Protocol contracts (`Embedder`, `VectorStore`, `Retriever`,
`Generator`, ...) for the rag-aio platform.

Dependency-light by design: installing `rag-core` pulls in only `pydantic`. See
`docs/modules/` and `adr/ADR-0002-protocol-based-contracts.md`.
