"""Chunk models with deterministic ids and rich provenance metadata."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .base import RagBaseModel
from .ids import stable_id


class ChunkMetadata(RagBaseModel):
    document_id: str
    document_hash: str
    chunker: str
    chunker_version: str
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    parent_chunk_id: str | None = None
    language: str | None = None
    custom: dict[str, Any] = Field(default_factory=dict)


class Chunk(RagBaseModel):
    """A retrievable unit of content.

    The id is deterministic: the same document content, chunker, and index
    always produce the same chunk id, enabling incremental indexing.
    """

    id: str = ""
    document_id: str
    text: str
    index: int = Field(default=0, ge=0)
    metadata: ChunkMetadata
    token_count: int | None = None

    @model_validator(mode="after")
    def _deterministic_id(self) -> Chunk:
        if not self.id:
            self.id = stable_id(
                self.document_id,
                self.metadata.chunker,
                self.metadata.chunker_version,
                self.index,
                self.text,
            )
        return self
