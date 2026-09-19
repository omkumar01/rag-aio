"""Canonical document models: pages, blocks, assets, metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .base import RagBaseModel
from .ids import content_hash, new_id

BlockKind = Literal[
    "text", "heading", "table", "figure", "caption", "list", "header", "footer", "other"
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentMetadata(RagBaseModel):
    title: str | None = None
    mime_type: str | None = None
    language: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    custom: dict[str, str] = Field(default_factory=dict)


class BoundingBox(RagBaseModel):
    """Axis-aligned bounding box in page coordinates (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _check_order(self) -> BoundingBox:
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError(f"invalid bounding box: {self.model_dump()}")
        return self


class PageBlock(RagBaseModel):
    """A semantically typed region of a page, with optional provenance."""

    id: str = Field(default_factory=new_id)
    page_id: str
    kind: BlockKind = "text"
    text: str = ""
    bbox: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    order: int = 0


class DocumentPage(RagBaseModel):
    """One page/slide/sheet of a document."""

    id: str = Field(default_factory=new_id)
    page_number: int = Field(ge=0)
    text: str = ""
    width: float | None = None
    height: float | None = None
    image_ref: str | None = None
    language: str | None = None
    blocks: list[PageBlock] = Field(default_factory=list)


class DocumentAsset(RagBaseModel):
    """An extracted asset (image, table, attachment) referenced by the document."""

    id: str = Field(default_factory=new_id)
    asset_id: str
    kind: Literal["image", "table", "figure", "attachment", "other"] = "image"
    page_number: int | None = None
    bbox: BoundingBox | None = None
    mime_type: str | None = None
    content_ref: str | None = None
    description: str | None = None


class Document(RagBaseModel):
    """The canonical document produced by ingestion.

    ``content_hash`` is derived from the source URI and normalized text so
    duplicate detection and incremental ingestion are stable across runs.
    """

    id: str = Field(default_factory=new_id)
    source_uri: str
    text: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    pages: list[DocumentPage] = Field(default_factory=list)
    assets: list[DocumentAsset] = Field(default_factory=list)
    tenant: str | None = None
    namespace: str | None = None
    content_hash: str = ""
    parser_name: str | None = None
    parser_version: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(f"{self.source_uri}\x00{self.text}")
