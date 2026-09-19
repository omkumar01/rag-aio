"""Context and citation models for the retrieval→generation boundary."""

from __future__ import annotations

from pydantic import Field

from .base import RagBaseModel


class ContextItem(RagBaseModel):
    """One evidence unit selected for the generation prompt."""

    chunk_id: str
    document_id: str
    text: str
    token_count: int = Field(ge=0)
    citation_id: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    score: float | None = None
    document_title: str | None = None
    source_uri: str | None = None


class Citation(RagBaseModel):
    """A source citation linking generated output back to provenance."""

    citation_id: str
    document_id: str
    chunk_id: str
    source_uri: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    quote: str | None = None


class Context(RagBaseModel):
    """The token-budgeted context assembled for generation."""

    items: list[ContextItem] = Field(default_factory=list)
    token_budget: int = Field(ge=0)
    strategy: str
    truncated: bool = False

    @property
    def total_tokens(self) -> int:
        return sum(item.token_count for item in self.items)
