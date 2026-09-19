"""Shared reranking wrapper logic.

This module implements :func:`rerank_candidates`, the helper that every
concrete reranker (remote, local, heuristic) delegates to. Centralizing the
wrapper means scoring, dedup, normalization, threshold, top-k, and
explainability-preserving output construction happen in exactly one place.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

from rag_core import RerankError, RetrievalHit

from .config import RerankConfig

# Scoring callable signature: (query_text, candidate_texts) -> per-candidate
# relevance scores, aligned 1:1 with the input texts.
Scorer = Callable[[str, list[str]], Awaitable[list[float]]]

__all__ = ["Scorer", "rerank_candidates"]


def _normalize(scores: list[float], mode: str) -> list[float]:
    """Apply the configured normalization to a list of raw scores."""
    if mode == "none":
        return list(scores)
    if mode == "minmax":
        lo = min(scores)
        hi = max(scores)
        if hi == lo:
            return [0.0 for _ in scores]
        span = hi - lo
        return [(s - lo) / span for s in scores]
    if mode == "sigmoid":
        return [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return list(scores)


async def rerank_candidates(
    query_text: str,
    candidates: list[RetrievalHit],
    scorer: Scorer,
    config: RerankConfig,
) -> list[RetrievalHit]:
    """Rerank ``candidates`` using an injectable async ``scorer``.

    Implements the shared wrapper contract once:

    * ``max_candidates`` hard cap -> :class:`RerankError` if exceeded.
    * chunk_id dedup (first occurrence wins).
    * batched scoring via ``scorer`` sliced by ``batch_size``.
    * score normalization (``none`` / ``minmax`` / ``sigmoid``).
    * ``score_threshold`` filtering on the (possibly normalized) score.
    * ``top_k`` truncation, then 0-based rank reassignment.

    Output hits preserve the original retrieval signal for explainability:
    ``original_score`` / ``original_rank`` / ``original_model`` are stored on
    the hit's ``metadata``, while ``score`` / ``normalized_score`` carry the
    new reranker score and ``model`` is left for the caller to set. An empty
    candidate list yields an empty list (no error).
    """
    if len(candidates) > config.max_candidates:
        raise RerankError(
            f"too many rerank candidates ({len(candidates)}); "
            f"max_candidates={config.max_candidates}"
        )

    if config.dedup:
        seen: set[str] = set()
        deduped: list[RetrievalHit] = []
        for hit in candidates:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            deduped.append(hit)
        candidates = deduped

    # Batched scoring, preserving candidate order.
    raw_scores: list[float] = []
    batch_size = config.batch_size
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        batch_texts = [hit.text or "" for hit in batch]
        scores = await scorer(query_text, batch_texts)
        if len(scores) != len(batch):
            raise RerankError(f"scorer returned {len(scores)} scores for {len(batch)} candidates")
        raw_scores.extend(scores)

    # Normalize, threshold, sort, top-k.
    norm_scores = _normalize(raw_scores, config.normalize)
    scored = list(zip(candidates, raw_scores, norm_scores, strict=True))

    if config.score_threshold is not None:
        scored = [t for t in scored if t[2] >= config.score_threshold]

    scored.sort(key=lambda t: t[2], reverse=True)

    if config.top_k is not None:
        scored = scored[: config.top_k]

    out: list[RetrievalHit] = []
    for new_rank, (hit, _raw, norm) in enumerate(scored):
        out.append(
            RetrievalHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=norm,
                normalized_score=norm,
                rank=new_rank,
                strategy=hit.strategy,
                model=None,
                strategy_scores=hit.strategy_scores,
                filters_applied=hit.filters_applied,
                parent_chunk_id=hit.parent_chunk_id,
                text=hit.text,
                metadata={
                    **hit.metadata,
                    "original_score": hit.score,
                    "original_rank": hit.rank,
                    "original_model": hit.model,
                },
            )
        )
    return out
