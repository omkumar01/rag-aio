"""High-level reranking pipeline wrapping a single reranker."""

from __future__ import annotations

from collections.abc import Sequence

from rag_core import Query, RetrievalHit
from rag_core.protocols import Reranker

from .config import RerankConfig

__all__ = ["RerankPipeline"]


class RerankPipeline:
    """Thin orchestrator over a :class:`Reranker`.

    Delegates scoring to the wrapped reranker, then tags each output hit with
    ``strategy="reranked"``. The underlying reranker applies
    ``score_threshold`` / ``normalize`` / dedup via the shared
    :func:`~rag_rerank.rerank_candidates` wrapper; per-call ``top_k`` (or the
    config default) is forwarded to the reranker.
    """

    def __init__(self, reranker: Reranker, config: RerankConfig) -> None:
        self._reranker = reranker
        self._config = config

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    async def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        effective_top_k = top_k if top_k is not None else self._config.top_k
        hits = await self._reranker.rerank(query, candidates, top_k=effective_top_k)
        for hit in hits:
            hit.strategy = "reranked"
        return hits
