"""Indexer abstractions for writing chunks + embeddings into vector stores.

* :class:`IndexerConfig` — batching / concurrency / namespace tuning.
* :class:`SparseCapableStore` — protocol for stores that accept sparse vectors.
* :class:`VectorIndexer` — implements the rag-core ``Indexer`` protocol;
  accepts any duck-typed :class:`~rag_core.protocols.VectorStore`.
* :class:`IngestionIndexer` — orchestrates embedding + indexing of a document's
  chunks and reports an :class:`IndexOutcome`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rag_core.base import RagBaseModel
from rag_core.chunks import Chunk
from rag_core.documents import Document
from rag_core.embeddings import Embedding, SparseEmbedding
from rag_core.protocols import Embedder, Indexer, SparseEmbedder, VectorStore

__all__ = [
    "IndexOutcome",
    "IndexerConfig",
    "IngestionIndexer",
    "SparseCapableStore",
    "VectorIndexer",
]

log = logging.getLogger(__name__)


class IndexerConfig(RagBaseModel):
    """Configuration for the indexer."""

    batch_size: int = 128
    max_concurrency: int = 4
    namespace: str | None = None


@runtime_checkable
class SparseCapableStore(Protocol):
    """Protocol extension for stores that accept sparse vectors.

    A regular :class:`~rag_core.protocols.VectorStore` is *not* required to
    support sparse vectors.  Stores that do should implement
    :meth:`upsert_sparse`.
    """

    async def upsert_sparse(
        self,
        chunk_id: str,
        indices: list[int],
        values: list[float],
        namespace: str | None = None,
    ) -> None: ...


@dataclass
class IndexOutcome:
    """Result of indexing one document."""

    chunks_indexed: int
    dims: int | None
    model: str
    sparse_model: str | None
    took_ms: float


_MetadataProvider = Callable[[str], dict[str, Any]]


def _default_metadata_provider(
    chunks: Sequence[Chunk],
    document: Document | None = None,
) -> _MetadataProvider:
    """Build a payload provider that enriches sparse/dense vectors with chunk metadata.

    The payload carries the chunk ``text`` and a nested ``metadata`` dict (with
    document provenance) so that vector stores can rebuild fully-populated
    :class:`~rag_core.retrieval.RetrievalHit` instances at query time.
    """
    chunk_map = {c.id: c for c in chunks}
    doc_meta = {
        "source_uri": document.source_uri if document is not None else None,
        "document_title": document.metadata.title if document is not None else None,
    }

    def provider(chunk_id: str) -> dict[str, Any]:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            return {"chunk_id": chunk_id}
        m = chunk.metadata
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_hash": m.document_hash,
            "chunker": m.chunker,
            "chunker_version": m.chunker_version,
            "page_numbers": m.page_numbers,
            "section_path": m.section_path,
            "char_start": m.char_start,
            "char_end": m.char_end,
            "token_count": m.token_count,
            "parent_chunk_id": m.parent_chunk_id,
            "language": m.language,
            "text": chunk.text,
            "metadata": {
                "page_numbers": m.page_numbers,
                "section_path": m.section_path,
                "language": m.language,
                **doc_meta,
            },
        }

    return provider


class VectorIndexer(Indexer):
    """Bulk-writes embeddings into any duck-typed :class:`VectorStore`.

    Parameters
    ----------
    store:
        Any object satisfying the :class:`~rag_core.protocols.VectorStore`
        protocol (duck-typed).
    config:
        Batching and concurrency settings.
    metadata_provider:
        Callable mapping a ``chunk_id`` to its payload dict.  Defaults to a
        minimal ``{"chunk_id": ...}`` payload.
    """

    def __init__(
        self,
        store: VectorStore,
        config: IndexerConfig | None = None,
        metadata_provider: _MetadataProvider | None = None,
    ) -> None:
        self._store = store
        self.config: IndexerConfig = config or IndexerConfig()
        self._metadata_provider = metadata_provider

    def _payload(self, chunk_id: str) -> dict[str, Any]:
        if self._metadata_provider is not None:
            return self._metadata_provider(chunk_id)
        return {"chunk_id": chunk_id}

    async def index(
        self,
        embeddings: Sequence[Embedding],
        sparse: Sequence[SparseEmbedding],
    ) -> None:
        if not embeddings:
            return

        sparse_map: dict[str, SparseEmbedding] = {s.chunk_id: s for s in sparse}
        batch_size = max(self.config.batch_size, 1)
        concurrency = max(self.config.max_concurrency, 1)
        batches = [
            list(embeddings[i : i + batch_size]) for i in range(0, len(embeddings), batch_size)
        ]
        sem = asyncio.Semaphore(concurrency)
        namespace = self.config.namespace

        async def _process(batch: Sequence[Embedding]) -> None:
            async with sem:
                for emb in batch:
                    payload = self._payload(emb.chunk_id)
                    ns = namespace or emb.namespace
                    await self._store.upsert(
                        chunk_id=emb.chunk_id,
                        vector=emb.vector,
                        payload=payload,
                        namespace=ns,
                    )
                    sp = sparse_map.get(emb.chunk_id)
                    if sp is not None and isinstance(self._store, SparseCapableStore):
                        await self._store.upsert_sparse(
                            chunk_id=sp.chunk_id,
                            indices=sp.indices,
                            values=sp.values,
                            namespace=ns or sp.namespace,
                        )

        await asyncio.gather(*[_process(b) for b in batches])


class IngestionIndexer:
    """Orchestrates embedding generation and indexing for a document's chunks."""

    def __init__(self, config: IndexerConfig | None = None) -> None:
        self.config: IndexerConfig = config or IndexerConfig()

    async def index_document(
        self,
        document: Document,
        chunks: list[Chunk],
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None,
        store: VectorStore,
        metadata_provider: _MetadataProvider | None = None,
    ) -> IndexOutcome:
        start = time.perf_counter()
        texts = [c.text for c in chunks]

        dense_vectors = await embedder.embed(texts) if texts else []
        dense_model = getattr(embedder, "model_name", "")
        dense_dim = len(dense_vectors[0]) if dense_vectors else None
        dense_normalize = getattr(embedder, "normalize", False)

        embeddings: list[Embedding] = []
        for chunk, vec in zip(chunks, dense_vectors, strict=False):
            embeddings.append(
                Embedding(
                    chunk_id=chunk.id,
                    vector=vec,
                    model=dense_model,
                    dimension=len(vec),
                    normalized=bool(dense_normalize),
                    namespace=self.config.namespace,
                )
            )

        sparse_embeddings: list[SparseEmbedding] = []
        sparse_model: str | None = None
        if sparse_embedder is not None and texts:
            sparse_vectors = await sparse_embedder.embed_sparse(texts)
            sparse_model = getattr(sparse_embedder, "model_name", None)
            for chunk, sp in zip(chunks, sparse_vectors, strict=False):
                sparse_embeddings.append(
                    SparseEmbedding(
                        chunk_id=chunk.id,
                        indices=sp.indices,
                        values=sp.values,
                        model=sparse_model or "",
                        namespace=self.config.namespace,
                    )
                )

        provider = metadata_provider or _default_metadata_provider(chunks, document)
        indexer = VectorIndexer(store, self.config, provider)
        await indexer.index(embeddings, sparse_embeddings)

        took_ms = (time.perf_counter() - start) * 1000.0
        return IndexOutcome(
            chunks_indexed=len(chunks),
            dims=dense_dim,
            model=dense_model,
            sparse_model=sparse_model,
            took_ms=took_ms,
        )
