"""rag-query: query intelligence and parallel retrieval execution.

Provides query normalization, intent classification, pluggable transformation
strategies (rewrite, expansion, HyDE, decomposition), an orchestrating engine,
and a parallel retrieval executor that fans out retrievers across query
variants concurrently.
"""

from __future__ import annotations

from .classify import QueryClass, classify_query
from .config import QueryConfig
from .engine import QueryEngine, QueryResult
from .normalize import normalize_query
from .retrieval_exec import ParallelRetrievalExecutor
from .strategies.decompose import DecomposeStrategy
from .strategies.expansion import ExpansionStrategy
from .strategies.hyde import HyDEStrategy
from .strategies.rewrite import RewriteStrategy

__version__ = "0.1.0"

__all__ = [
    "DecomposeStrategy",
    "ExpansionStrategy",
    "HyDEStrategy",
    "ParallelRetrievalExecutor",
    "QueryClass",
    "QueryConfig",
    "QueryEngine",
    "QueryResult",
    "RewriteStrategy",
    "__version__",
    "classify_query",
    "normalize_query",
]
