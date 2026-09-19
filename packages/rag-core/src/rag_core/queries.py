"""Query and query-variant models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .base import RagBaseModel
from .ids import new_id

QueryVariantKind = Literal["original", "rewrite", "expansion", "hyde", "decomposition", "subquery"]


class QueryVariant(RagBaseModel):
    """A transformed form of the original query produced by a QueryStrategy."""

    id: str = Field(default_factory=new_id)
    query_id: str
    text: str
    kind: QueryVariantKind
    weight: float = Field(default=1.0, ge=0.0)
    strategy: str | None = None


class Query(RagBaseModel):
    """A retrieval/generation query with filters and budgets."""

    id: str = Field(default_factory=new_id)
    text: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1)
    tenant: str | None = None
    namespace: str | None = None
    deadline_s: float | None = Field(default=None, gt=0.0)
    variants: list[QueryVariant] = Field(default_factory=list)
