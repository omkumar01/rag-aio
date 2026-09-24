"""Reranking adapters for rag-aio.

Provides:

* :class:`RemoteReranker` - OpenAI-compatible HTTP reranker with runtime
  detection of a native ``/rerank`` endpoint vs. a scoring-template fallback
  over ``/v1/chat/completions`` (the LM Studio qwen3-reranker question).
* :class:`CrossEncoderReranker` - local sentence-transformers reranker.
* :class:`HeuristicReranker` - zero-dependency Jaccard baseline.
* :class:`RerankPipeline` - thin orchestrator tagging hits ``strategy="reranked"``.
* :func:`rerank_candidates` - shared scoring/normalization wrapper.
* :func:`to_rerank_hits` / :func:`explain` - explainability helpers.
"""

from __future__ import annotations

from .base import Scorer, rerank_candidates
from .config import NormalizeMode, RerankConfig
from .explain import explain, to_rerank_hits
from .local import CrossEncoderReranker, HeuristicReranker
from .pipeline import RerankPipeline
from .remote import RemoteReranker

__version__ = "0.1.1"

__all__ = [
    "CrossEncoderReranker",
    "HeuristicReranker",
    "NormalizeMode",
    "RemoteReranker",
    "RerankConfig",
    "RerankPipeline",
    "Scorer",
    "__version__",
    "explain",
    "rerank_candidates",
    "to_rerank_hits",
]
