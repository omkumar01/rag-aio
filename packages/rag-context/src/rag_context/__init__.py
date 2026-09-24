"""rag-context: context engineering between retrieval and generation.

Provides token budgeting, deduplication, neighbor expansion, overlap merging,
multi-strategy ordering, and citation mapping over :mod:`rag_core` retrieval
results. System instructions are assembled by the caller; this package only
shapes the retrieved evidence that is safe to inject into a generation prompt.
"""

from __future__ import annotations

from .builder import ContextBuilderImpl, sanitize_item_text
from .citations import SourceLookup, build_citations
from .config import CitationStyle, ContextConfig, Strategy
from .dedup import dedupe_hits
from .expansion import NeighborProvider, expand_with_neighbors
from .ordering import (
    STRATEGIES,
    OrderFn,
    chronological,
    diversity,
    document_grouped,
    order_by_strategy,
    relevance_first,
    section_aware,
)
from .render import render_context
from .tokenizer import ContextTokenizer, WhitespaceCounter

__version__ = "0.1.1"

__all__ = [
    "STRATEGIES",
    "CitationStyle",
    "ContextBuilderImpl",
    "ContextConfig",
    "ContextTokenizer",
    "NeighborProvider",
    "OrderFn",
    "SourceLookup",
    "Strategy",
    "WhitespaceCounter",
    "build_citations",
    "chronological",
    "dedupe_hits",
    "diversity",
    "document_grouped",
    "expand_with_neighbors",
    "order_by_strategy",
    "relevance_first",
    "render_context",
    "sanitize_item_text",
    "section_aware",
    "version",
]


def version() -> str:
    return __version__
