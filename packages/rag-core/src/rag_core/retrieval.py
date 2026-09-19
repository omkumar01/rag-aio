"""Retrieval result models with full explainability."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import RagBaseModel


class RetrievalHit(RagBaseModel):
    """One retrieved chunk.

    Carries everything needed to explain why the chunk was retrieved: the
    strategy that produced it, raw and normalized scores, per-strategy
    contributions, rank, applied filters, and model provenance.
    """

    chunk_id: str
    document_id: str
    score: float
    normalized_score: float
    rank: int = Field(ge=0)
    strategy: str
    model: str | None = None
    strategy_scores: dict[str, float] = Field(default_factory=dict)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    parent_chunk_id: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(RagBaseModel):
    """The output of one or more retrieval strategies."""

    query_id: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_candidates: int | None = None
