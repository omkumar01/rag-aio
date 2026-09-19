"""Fusion strategies for combining ranked retrieval results.

Both :class:`ReciprocalRankFusion` and :class:`WeightedScoreFusion` accept an
optional ``weights`` argument (one weight per result list, in order) and a
``dedup`` flag. When ``dedup`` is true, hits sharing a ``chunk_id`` are merged
and their per-strategy contributions accumulated; when false, each occurrence is
emitted as its own (hybrid) hit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_core.errors import FusionError
from rag_core.retrieval import RetrievalHit, RetrievalResult


@runtime_checkable
class FusionStrategy(Protocol):
    """Merge ranked result lists from multiple strategies into one."""

    def fuse(
        self,
        results: Sequence[RetrievalResult],
        top_k: int,
        weights: list[float] | None = None,
        dedup: bool = True,
    ) -> RetrievalResult:
        """Fuse ``results`` and return the top ``top_k`` hits."""


def _strategy_of(result: RetrievalResult) -> str:
    """The (single) source strategy label for a retrieval result."""
    return result.strategies[0] if result.strategies else "unknown"


def _resolve_weights(
    results: Sequence[RetrievalResult], weights: list[float] | None
) -> list[float]:
    """Normalize the per-list weights, defaulting to 1.0 each."""
    if weights is None:
        return [1.0] * len(results)
    if len(weights) != len(results):
        raise FusionError(
            "weights length does not match number of result lists",
            details={"weights": len(weights), "results": len(results)},
        )
    return list(weights)


@dataclass
class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

    score(chunk) = sum over lists of weight / (k + rank)
    where ``rank`` is the 0-based position within a list.
    """

    k: float = 60.0

    def fuse(
        self,
        results: Sequence[RetrievalResult],
        top_k: int,
        weights: list[float] | None = None,
        dedup: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        weights = _resolve_weights(results, weights)
        query_id = results[0].query_id if results else ""

        entries: list[tuple[float, int, RetrievalHit, dict[str, float]]] = []
        if dedup:
            contributions: dict[str, dict[str, float]] = {}
            first_hit: dict[str, RetrievalHit] = {}
            order: dict[str, int] = {}

            for weight, result in zip(weights, results, strict=True):
                strategy = _strategy_of(result)
                for rank, hit in enumerate(result.hits):
                    contribution = weight / (self.k + rank)
                    cid = hit.chunk_id
                    if cid not in contributions:
                        contributions[cid] = {}
                        first_hit[cid] = hit
                        order[cid] = len(order)
                    contributions[cid][strategy] = (
                        contributions[cid].get(strategy, 0.0) + contribution
                    )

            entries = [
                (sum(c.values()), order[cid], first_hit[cid], c) for cid, c in contributions.items()
            ]
        else:
            seq = 0
            for weight, result in zip(weights, results, strict=True):
                strategy = _strategy_of(result)
                for rank, hit in enumerate(result.hits):
                    contribution = weight / (self.k + rank)
                    entries.append((contribution, seq, hit, {strategy: contribution}))
                    seq += 1

        fused = _build_hits(entries, top_k)
        t1 = time.perf_counter()
        return RetrievalResult(
            query_id=query_id,
            hits=fused,
            strategies=["hybrid"],
            timings_ms={"fusion_ms": (t1 - t0) * 1000.0},
            total_candidates=sum(len(r.hits) for r in results),
        )


@dataclass
class WeightedScoreFusion:
    """Weighted fusion over (optionally min-max) normalized scores.

    Each list's scores are min-max normalized to ``[0, 1]`` when ``normalize``
    is true, then scaled by the per-list weight and summed across lists for
    every chunk_id.
    """

    normalize: bool = True

    def fuse(
        self,
        results: Sequence[RetrievalResult],
        top_k: int,
        weights: list[float] | None = None,
        dedup: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        weights = _resolve_weights(results, weights)
        query_id = results[0].query_id if results else ""

        entries: list[tuple[float, int, RetrievalHit, dict[str, float]]] = []
        if dedup:
            contributions: dict[str, dict[str, float]] = {}
            first_hit: dict[str, RetrievalHit] = {}
            order: dict[str, int] = {}

            for weight, result in zip(weights, results, strict=True):
                strategy = _strategy_of(result)
                scores = [h.score for h in result.hits]
                norm_scores = _minmax(scores) if self.normalize else list(scores)
                for rank, hit in enumerate(result.hits):
                    contribution = weight * norm_scores[rank]
                    cid = hit.chunk_id
                    if cid not in contributions:
                        contributions[cid] = {}
                        first_hit[cid] = hit
                        order[cid] = len(order)
                    contributions[cid][strategy] = (
                        contributions[cid].get(strategy, 0.0) + contribution
                    )

            entries = [
                (sum(c.values()), order[cid], first_hit[cid], c) for cid, c in contributions.items()
            ]
        else:
            seq = 0
            for weight, result in zip(weights, results, strict=True):
                strategy = _strategy_of(result)
                scores = [h.score for h in result.hits]
                norm_scores = _minmax(scores) if self.normalize else list(scores)
                for rank, hit in enumerate(result.hits):
                    contribution = weight * norm_scores[rank]
                    entries.append((contribution, seq, hit, {strategy: contribution}))
                    seq += 1

        fused = _build_hits(entries, top_k)
        t1 = time.perf_counter()
        return RetrievalResult(
            query_id=query_id,
            hits=fused,
            strategies=["hybrid"],
            timings_ms={"fusion_ms": (t1 - t0) * 1000.0},
            total_candidates=sum(len(r.hits) for r in results),
        )


def _minmax(scores: list[float]) -> list[float]:
    """Min-max normalize scores to ``[0, 1]``."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi > lo:
        span = hi - lo
        return [(s - lo) / span for s in scores]
    # All scores equal: a strictly-positive value maps to 1.0, else 0.0.
    return [1.0 if s > 0 else 0.0 for s in scores]


def _build_hits(
    entries: list[tuple[float, int, RetrievalHit, dict[str, float]]],
    top_k: int,
) -> list[RetrievalHit]:
    """Sort merged entries and emit ranked, fused :class:`RetrievalHit` objects."""
    best = 0.0
    for score, _, _, _ in entries:
        if score > best:
            best = score

    entries.sort(key=lambda e: (-e[0], e[1]))
    hits: list[RetrievalHit] = []
    for rank, (score, _, first_hit, strategy_scores) in enumerate(entries[: max(top_k, 0)]):
        normalized = score / best if best > 0 else 0.0
        hits.append(
            first_hit.model_copy(
                update={
                    "score": score,
                    "normalized_score": normalized,
                    "rank": rank,
                    "strategy": "hybrid",
                    "strategy_scores": dict(strategy_scores),
                }
            )
        )
    return hits


__all__ = [
    "FusionStrategy",
    "ReciprocalRankFusion",
    "WeightedScoreFusion",
]
