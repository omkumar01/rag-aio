"""rag-retrieval: dense, sparse, hybrid retrieval with fusion and explainability."""

from __future__ import annotations

from .config import RetrievalConfig
from .dense import DenseRetriever
from .explain import attach_text, explain_result
from .fusion import FusionStrategy, ReciprocalRankFusion, WeightedScoreFusion
from .hybrid import HybridRetriever
from .sparse import BM25Retriever, SparseRetriever, SparseSearchStore

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "FusionStrategy",
    "HybridRetriever",
    "ReciprocalRankFusion",
    "RetrievalConfig",
    "SparseRetriever",
    "SparseSearchStore",
    "WeightedScoreFusion",
    "attach_text",
    "explain_result",
]
