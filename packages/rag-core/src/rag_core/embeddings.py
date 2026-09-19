"""Dense and sparse embedding models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .base import RagBaseModel
from .ids import new_id


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Embedding(RagBaseModel):
    """A dense embedding for one chunk."""

    id: str = Field(default_factory=new_id)
    chunk_id: str
    vector: list[float]
    model: str
    dimension: int | None = Field(default=None, ge=0)
    normalized: bool = False
    namespace: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _infer_dimension(self) -> Embedding:
        if self.dimension is None:
            self.dimension = len(self.vector)
        elif self.dimension != len(self.vector):
            raise ValueError(
                f"dimension mismatch: declared {self.dimension}, got {len(self.vector)}"
            )
        return self


class SparseEmbedding(RagBaseModel):
    """A sparse (e.g. learned-sparse/BM25-style) embedding for one chunk."""

    id: str = Field(default_factory=new_id)
    chunk_id: str
    indices: list[int]
    values: list[float]
    model: str
    namespace: str | None = None

    @model_validator(mode="after")
    def _check_alignment(self) -> SparseEmbedding:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must have equal length "
                f"({len(self.indices)} != {len(self.values)})"
            )
        if any(i < 0 for i in self.indices):
            raise ValueError("indices must be non-negative")
        return self


EmbeddingKind = Literal["dense", "sparse"]
