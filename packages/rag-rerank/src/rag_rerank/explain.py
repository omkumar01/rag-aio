"""Explainability helpers mapping pre/post-rerank hits to audit records."""

from __future__ import annotations

from typing import Any

from rag_core import RerankHit, RetrievalHit

__all__ = ["explain", "to_rerank_hits"]


def to_rerank_hits(before: list[RetrievalHit], after: list[RetrievalHit]) -> list[RerankHit]:
    """Convert pre/post-rerank hit lists into :class:`RerankHit` audit records.

    ``before`` is the pre-rerank candidate set (with original scores/ranks);
    ``after`` is the reranker's output. Each ``RerankHit`` carries the new
    score/rank alongside the original retrieval score/rank so evaluation can
    separate retrieval quality from reranking quality.
    """
    before_map = {hit.chunk_id: hit for hit in before}
    out: list[RerankHit] = []
    for new_rank, hit in enumerate(after):
        original = before_map.get(hit.chunk_id)
        if original is not None:
            original_score = original.score
            original_rank = original.rank
        else:
            # Fall back to metadata written by the shared rerank wrapper.
            original_score = float(hit.metadata.get("original_score", hit.score))
            original_rank = int(hit.metadata.get("original_rank", hit.rank))
        out.append(
            RerankHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=hit.score,
                rank=new_rank,
                original_score=original_score,
                original_rank=original_rank,
                model=hit.model,
                text=hit.text,
                metadata=hit.metadata,
            )
        )
    return out


def explain(hits: list[RerankHit]) -> dict[str, Any]:
    """Summarize a set of :class:`RerankHit` records for observability."""
    scores = [hit.score for hit in hits]
    models = sorted({hit.model for hit in hits if hit.model})
    return {
        "n_results": len(hits),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "ranks_changed": sum(1 for hit in hits if hit.original_rank != hit.rank),
        "reranker": models,
    }
