"""End-to-end embedding + indexing pipeline.

:class:`EmbeddingPipeline` wires together a chunker, embedder(s), and indexer,
providing incremental reindexing via :meth:`should_reindex`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rag_core import config_hash
from rag_core.documents import Document
from rag_core.protocols import Chunker, Embedder, SparseEmbedder, VectorStore

from .indexing import IndexerConfig, IndexOutcome, IngestionIndexer

__all__ = ["EmbeddingPipeline", "PipelineOutcome"]


@dataclass
class PipelineOutcome:
    """Result of running :meth:`EmbeddingPipeline.process`."""

    chunks: int
    chunks_indexed: int
    dims: int | None
    model: str
    sparse_model: str | None
    took_ms: float
    reindexed: bool


class EmbeddingPipeline:
    """Orchestrates chunking → embedding → sparse → indexing.

    Parameters
    ----------
    chunker:
        Any object implementing the rag-core ``Chunker`` protocol.
    embedder:
        Dense embedder (``Embedder`` protocol).
    sparse_embedder:
        Optional sparse embedder (``SparseEmbedder`` protocol).
    indexer:
        Optional ``IngestionIndexer`` instance; one is created from *config*
        if omitted.
    config:
        Indexer configuration (batching, concurrency, namespace).
    store:
        Vector store used at index time.  May also be passed per-call to
        :meth:`process`.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        indexer: IngestionIndexer | None = None,
        config: IndexerConfig | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.config: IndexerConfig = config or IndexerConfig()
        self.indexer = indexer or IngestionIndexer(self.config)
        self.store = store

    # -- incremental reindexing --------------------------------------------

    @staticmethod
    def should_reindex(document: Document, config_hash_str: str, seen: dict[str, str]) -> bool:
        """Return *True* if *document* should be re-indexed.

        The ``seen`` dict maps ``document.id`` → ``"content_hash:config_hash"``.
        If the stored key matches the current document + config, the document
        is skipped (``False``).
        """
        key = f"{document.content_hash}:{config_hash_str}"
        if seen.get(document.id) == key:
            return False
        seen[document.id] = key
        return True

    def config_hash(self) -> str:
        """Hash of the pipeline's indexer configuration for incremental checks."""
        return config_hash(self.config.model_dump(mode="json"))

    # -- main entry point --------------------------------------------------

    async def process(
        self,
        document: Document,
        store: VectorStore | None = None,
    ) -> PipelineOutcome:
        """Run the full pipeline on *document* and return the outcome."""
        store = store or self.store
        if store is None:
            raise ValueError("a vector store is required to run the pipeline")

        start = time.perf_counter()
        chunks = await self.chunker.chunk(document)

        outcome: IndexOutcome = await self.indexer.index_document(
            document,
            chunks,
            self.embedder,
            self.sparse_embedder,
            store,
        )
        took_ms = (time.perf_counter() - start) * 1000.0
        return PipelineOutcome(
            chunks=len(chunks),
            chunks_indexed=outcome.chunks_indexed,
            dims=outcome.dims,
            model=outcome.model,
            sparse_model=outcome.sparse_model,
            took_ms=took_ms,
            reindexed=True,
        )
