"""Compact JSON serialization helpers for rag-core boundary objects."""

from __future__ import annotations

from pydantic import ValidationError

from .base import RagBaseModel


def to_json(model: RagBaseModel) -> str:
    """Serialize a rag-core model to compact JSON (no whitespace)."""
    return model.model_dump_json()


def from_json[M: RagBaseModel](data: str, model_type: type[M]) -> M:
    """Deserialize and validate JSON into ``model_type``.

    Raises ``pydantic.ValidationError`` on schema mismatch.
    """
    try:
        return model_type.model_validate_json(data)
    except ValidationError:
        raise
