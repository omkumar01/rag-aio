"""Typed configuration for rerankers."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from rag_core import RagBaseModel

NormalizeMode = Literal["none", "minmax", "sigmoid"]


class RerankConfig(RagBaseModel):
    """Configuration governing every reranker in this package.

    ``max_candidates`` is a hard cap: reranking is the most expensive RAG
    stage, so callers must limit the candidate set *before* asking a reranker
    to score it. Passing more than ``max_candidates`` candidates raises
    :class:`~rag_core.RerankError`.
    """

    top_k: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None)
    normalize: NormalizeMode = "none"
    dedup: bool = True
    batch_size: int = Field(default=16, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_candidates: int = Field(default=100, ge=1)


__all__ = ["NormalizeMode", "RerankConfig"]
