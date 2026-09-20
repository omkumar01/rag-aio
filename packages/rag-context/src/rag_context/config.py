"""Configuration for context assembly.

:class:`ContextConfig` describes how a :class:`~rag_core.protocols.ContextBuilder`
should turn retrieved hits into a prompt-sized, citation-preserving context: which
ordering strategy to use, how many tokens to budget, whether to deduplicate and
expand with neighbors, and how citations are formatted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from rag_core.base import RagBaseModel

Strategy = Literal[
    "relevance_first",
    "chronological",
    "section_aware",
    "document_grouped",
    "diversity",
]

CitationStyle = Literal["numeric", "none"]


class ContextConfig(RagBaseModel):
    """Tunables controlling context assembly."""

    strategy: Strategy = "relevance_first"
    token_budget: int = Field(default=2048, ge=0)
    reserve_for_answer: int = Field(default=0, ge=0)
    max_per_document: int = Field(default=0, ge=0)
    dedup: bool = True
    merge_overlapping: bool = True
    expand_neighbors: bool = False
    neighbor_window: int = Field(default=1, ge=1)
    min_score: float | None = None
    citation_style: CitationStyle = "numeric"
    max_item_chars: int = Field(default=8000, ge=0)
