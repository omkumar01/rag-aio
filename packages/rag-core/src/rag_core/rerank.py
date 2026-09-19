"""Reranking result models preserving pre-rerank scores for explainability."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import RagBaseModel


class RerankHit(RagBaseModel):
    """A candidate after reranking.

    ``original_score``/``original_rank`` preserve the first-stage retrieval
    signal so evaluation can separate retrieval quality from reranking quality.
    """

    chunk_id: str
    document_id: str
    score: float
    rank: int
    original_score: float
    original_rank: int
    model: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
