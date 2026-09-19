"""Configuration models for rag-retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from rag_core.base import RagBaseModel

_Strategies = list[Literal["dense", "sparse", "bm25"]]


def _default_strategies() -> _Strategies:
    return ["dense"]


class RetrievalConfig(RagBaseModel):
    """Tunable configuration for a retrieval run.

    Controls which strategies run, how many candidates each strategy fetches
    before fusion, the fusion algorithm and its weights, score thresholding,
    per-strategy candidate limits, and deduplication of fused hits.
    """

    strategies: _Strategies = Field(default_factory=_default_strategies)
    top_k: int = 10
    candidate_k: int = 50
    fusion: Literal["rrf", "weighted"] = "rrf"
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    score_threshold: float | None = None
    per_source_limits: dict[str, int] = Field(default_factory=dict)
    dedup: bool = True


__all__ = ["RetrievalConfig"]
