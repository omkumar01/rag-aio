"""Sparse and BM25 retrieval."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import bm25s
from rag_core.protocols import Embedder as _Embedder  # noqa: F401  (kept for symmetry)
from rag_core.protocols import SparseEmbedder
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit, RetrievalResult

from .config import RetrievalConfig


@runtime_checkable
class SparseSearchStore(Protocol):
    """A vector store that can score sparse (learned-sparse/BM25) queries."""

    async def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]:
        """Return the top-``k`` sparse-matching hits."""


class SparseRetriever:
    """Retrieves chunks using a learned-sparse/embedding model.

    The sparse embedder is duck-typed: any object with an
    ``async embed_sparse(texts)`` method returning ``list[SparseVector]``
    works.
    """

    strategy: str = "sparse"

    def __init__(
        self,
        store: SparseSearchStore,
        sparse_embedder: SparseEmbedder,
        config: RetrievalConfig,
    ) -> None:
        self._store = store
        self._embedder = sparse_embedder
        self._config = config
        self._model: str = getattr(sparse_embedder, "model_name", "unknown")

    async def retrieve(self, query: Query) -> RetrievalResult:
        """Embed sparsely, search, and return ranked hits."""
        t0 = time.perf_counter()
        vectors = await self._embedder.embed_sparse([query.text])
        t_embed = time.perf_counter()
        sv = vectors[0]
        raw = await self._store.search_sparse(
            sv.indices,
            sv.values,
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
                "sparse_embed_ms": (t_embed - t0) * 1000.0,
                "sparse_search_ms": (t_search - t_embed) * 1000.0,
            },
            total_candidates=len(hits),
        )


class BM25Retriever:
    """In-process BM25 retriever backed by ``bm25s``.

    ``corpus_provider`` returns ``(chunk_id, document_id, text)`` records. The
    BM25 index is built lazily and refreshed whenever the corpus signature
    changes. All bm25s work runs in a worker thread.
    """

    strategy: str = "bm25"

    def __init__(
        self,
        corpus_provider: Callable[[], list[tuple[str, str, str]]],
        config: RetrievalConfig,
    ) -> None:
        self._corpus_provider = corpus_provider
        self._config = config
        self._model: Any = None
        self._corpus: list[tuple[str, str, str]] = []
        self._sig: tuple[str, ...] = ()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lower tokenization."""
        return text.lower().split()

    def _signature(self, corpus: list[tuple[str, str, str]]) -> tuple[str, ...]:
        return tuple(chunk_id for chunk_id, _, _ in corpus)

    def _build(self, corpus: list[tuple[str, str, str]]) -> None:
        if not corpus:
            self._model = None
            self._corpus = []
            return
        texts = [self._tokenize(text) for _, _, text in corpus]
        model = bm25s.BM25()
        model.index(texts, show_progress=False)
        self._model = model
        self._corpus = corpus

    def _search(self, query_tokens: list[list[str]], k: int) -> tuple[list[int], list[float]]:
        res = self._model.retrieve(query_tokens, k=k, return_as="tuple", show_progress=False)
        indices, scores = res
        idx_row = indices[0] if _ndim2(indices) else indices
        sc_row = scores[0] if _ndim2(scores) else scores
        doc_indices = [int(i) for i in idx_row]
        doc_scores = [float(s) for s in sc_row]
        return doc_indices, doc_scores

    async def retrieve(self, query: Query) -> RetrievalResult:
        t0 = time.perf_counter()
        corpus = await asyncio.to_thread(self._corpus_provider)
        sig = self._signature(corpus)
        if sig != self._sig or self._model is None:
            await asyncio.to_thread(self._build, corpus)
            self._sig = sig

        if not corpus or self._model is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query_id=query.id,
                hits=[],
                strategies=[self.strategy],
                timings_ms={self.strategy: elapsed},
                total_candidates=0,
            )

        k = min(self._config.candidate_k, len(corpus))
        query_tokens = [self._tokenize(query.text)]
        doc_indices, doc_scores = await asyncio.to_thread(self._search, query_tokens, k)

        scored = [
            (idx, score) for idx, score in zip(doc_indices, doc_scores, strict=True) if idx != -1
        ]
        max_score = max((s for _, s in scored), default=0.0)
        scored.sort(key=lambda p: (-p[1], self._corpus[p[0]][0]))

        hits: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(scored):
            chunk_id, document_id, text = self._corpus[idx]
            normalized = score / max_score if max_score > 0 else 0.0
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    score=score,
                    normalized_score=normalized,
                    rank=rank,
                    strategy=self.strategy,
                    model="bm25s",
                    strategy_scores={self.strategy: score},
                    filters_applied=dict(query.filters),
                    text=text,
                    metadata={},
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            query_id=query.id,
            hits=hits,
            strategies=[self.strategy],
            timings_ms={self.strategy: elapsed},
            total_candidates=len(hits),
        )


def _ndim2(arr: Any) -> bool:
    """Return True if ``arr`` is a 2D array-like requiring row extraction."""
    ndim = getattr(arr, "ndim", 1)
    return bool(ndim == 2)


__all__ = ["BM25Retriever", "SparseRetriever", "SparseSearchStore"]
