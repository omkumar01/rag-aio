"""Shared Pydantic base model for all rag-core boundary objects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RagBaseModel(BaseModel):
    """Base for all rag-core models.

    Strict (``extra="forbid"``) so contract drift across module versions fails
    loudly instead of silently dropping fields.
    """

    model_config = ConfigDict(extra="forbid")
