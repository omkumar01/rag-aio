"""Dense vector retrieval."""

from __future__ import annotations

import time

from rag_core.protocols import Embedder, VectorStore
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit, RetrievalResult

from .config import RetrievalConfig


class DenseRetriever:
    """Retrieves chunks by dense vector similarity.

    The embedder is duck-typed: any object with an ``async embed(texts)``
    method returning ``list[list[float]]`` works. If it exposes a
    ``model_name`` attribute, that is recorded as provenance on each hit.
    """

    strategy: str = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder, config: RetrievalConfig) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config
        self._model: str = getattr(embedder, "model_name", "unknown")

    async def retrieve(self, query: Query) -> RetrievalResult:
        """Embed the query, search the store, and return ranked hits."""
        t0 = time.perf_counter()
        vectors = await self._embedder.embed([query.text])
        t_embed = time.perf_counter()
        vector = vectors[0]
        raw = await self._store.search(
            vector,
            self._config.candidate_k,
            filters=query.filters,
            namespace=query.namespace,
        )
        t_search = time.perf_counter()

        ordered = sorted(raw, key=lambda h: (-h.score, h.chunk_id))
        if self._config.score_threshold is not None:
            threshold = self._config.score_threshold
            ordered = [h for h in ordered if h.score >= threshold]

        hits: list[RetrievalHit] = []
        for rank, h in enumerate(ordered):
            hits.append(
                RetrievalHit(
                    chunk_id=h.chunk_id,
                    document_id=h.document_id,
                    score=h.score,
                    normalized_score=h.normalized_score,
                    rank=rank,
                    strategy=self.strategy,
                    model=self._model,
                    strategy_scores={self.strategy: h.score},
                    filters_applied=dict(query.filters),
                    parent_chunk_id=h.parent_chunk_id,
                    text=h.text,
                    metadata=dict(h.metadata),
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            query_id=query.id,
            hits=hits,
            strategies=[self.strategy],
            timings_ms={
                self.strategy: elapsed,
                "dense_embed_ms": (t_embed - t0) * 1000.0,
                "dense_search_ms": (t_search - t_embed) * 1000.0,
            },
            total_candidates=len(hits),
        )
