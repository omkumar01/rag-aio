"""Lightweight shared value types used across contracts."""

from __future__ import annotations

from pydantic import model_validator

from .base import RagBaseModel


class SparseVector(RagBaseModel):
    """A sparse vector: aligned parallel index/value arrays."""

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def _check_alignment(self) -> SparseVector:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must have equal length "
                f"({len(self.indices)} != {len(self.values)})"
            )
        if any(i < 0 for i in self.indices):
            raise ValueError("indices must be non-negative")
        return self
