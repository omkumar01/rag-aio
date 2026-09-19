"""OCR result models inheriting the rag-core boundary.

These are provenance-preserving value objects: every ``OCRRegion`` is owned by an
``OCRPageResult`` (which carries ``page_number`` and ``engine``) which is in turn owned
by an ``OCRResult`` carrying the ``document_id``. The semantic (VLM) pass produces
regions without bounding boxes; the mechanical (engine) pass produces regions with
bounding boxes derived from ``RawLine`` coordinates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from rag_core.base import RagBaseModel
from rag_core.documents import BoundingBox

RegionKind = Literal["text", "heading", "table", "figure", "caption", "list", "other"]


class OCRRegion(RagBaseModel):
    """A single semantically typed region extracted from a page image."""

    text: str
    bbox: BoundingBox | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    kind: RegionKind = "text"
    language: str | None = None


class OCRPageResult(RagBaseModel):
    """The OCR output for a single page, with provenance."""

    page_number: int = Field(ge=0)
    regions: list[OCRRegion] = Field(default_factory=list)
    engine: str = ""
    width: float | None = None
    height: float | None = None
    language: str | None = None

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across regions; ``0.0`` when there are no regions."""
        if not self.regions:
            return 0.0
        total = 0.0
        for region in self.regions:
            total += region.confidence
        return total / len(self.regions)


class OCRResult(RagBaseModel):
    """The full OCR output for a document, including escalation provenance."""

    document_id: str
    pages: list[OCRPageResult] = Field(default_factory=list)
    escalated_pages: list[int] = Field(default_factory=list)
