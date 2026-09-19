"""Confidence-driven escalation routing.

The router tries mechanical OCR engines in priority order. When the best engine
either returns no lines or produces a mean confidence below the configured
threshold, the page is *escalated* to the (optional) VLM semantic extractor.
Escalation is budgeted per document.
"""

from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image
from rag_core.base import RagBaseModel
from rag_core.documents import BoundingBox
from rag_core.errors import OCRError

from rag_ocr.engines import OCREngine, RawLine, aggregate_confidence
from rag_ocr.models import OCRPageResult, OCRRegion, OCRResult
from rag_ocr.semantic import VLMSemanticExtractor


def _lines_to_regions(lines: list[RawLine]) -> list[OCRRegion]:
    """Convert raw engine lines into traceable :class:`OCRRegion` objects."""
    regions: list[OCRRegion] = []
    for line in lines:
        x0, y0, x1, y1 = line.bbox
        bbox = None
        if x0 or y0 or x1 or y1:
            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        regions.append(
            OCRRegion(
                text=line.text,
                bbox=bbox,
                confidence=line.confidence,
                kind="text",
                language=None,
            )
        )
    return regions


def _read_size_sync(image_bytes: bytes) -> tuple[float, float]:
    with Image.open(BytesIO(image_bytes)) as img:
        return float(img.width), float(img.height)


async def _read_size(image_bytes: bytes) -> tuple[float | None, float | None]:
    """Best-effort image dimensions read off the event loop."""
    try:
        return await asyncio.to_thread(_read_size_sync, image_bytes)
    except Exception:
        return None, None


class OCRRouterConfig(RagBaseModel):
    """Tunable escalation policy."""

    confidence_threshold: float = 0.7
    min_lines: int = 1
    escalate_model: bool = True
    max_escalations_per_doc: int = 3


class OCRRouter:
    """Routes pages through mechanical engines and, when needed, a VLM fallback.

    ``process_document`` owns the per-document escalation budget; ``process_page`` is
    the single-page entry point compatible with the ``OCRProcessor`` protocol and
    operates with a fresh budget (so one page may always escalate).
    """

    def __init__(
        self,
        engines: list[OCREngine],
        semantic: VLMSemanticExtractor | None,
        config: OCRRouterConfig | None = None,
    ) -> None:
        if not engines:
            raise OCRError("OCRRouter requires at least one engine", code="config_error")
        self._engines: list[OCREngine] = list(engines)
        self._semantic: VLMSemanticExtractor | None = semantic
        self._config: OCRRouterConfig = config or OCRRouterConfig()

    async def _run_engines(self, image_bytes: bytes) -> tuple[list[RawLine], str]:
        """Try engines in order; return the first non-empty result (or the last)."""
        lines: list[RawLine] = []
        last_name = ""
        for engine in self._engines:
            last_name = engine.name
            lines = await engine.recognize(image_bytes)
            if lines:
                return lines, engine.name
        return lines, last_name

    def _needs_escalation(
        self, lines: list[RawLine], mean_conf: float | None, escalations_used: int
    ) -> bool:
        semantic = self._semantic
        if not self._config.escalate_model or semantic is None:
            return False
        if escalations_used >= self._config.max_escalations_per_doc:
            return False
        if len(lines) < self._config.min_lines:
            return True
        return mean_conf is not None and mean_conf < self._config.confidence_threshold

    async def _process_page(
        self,
        page_number: int,
        image_bytes: bytes,
        escalated: list[int],
        width: float | None,
        height: float | None,
    ) -> OCRPageResult:
        lines, engine_name = await self._run_engines(image_bytes)
        mean_conf = aggregate_confidence(lines)
        semantic = self._semantic

        if semantic is not None and self._needs_escalation(lines, mean_conf, len(escalated)):
            regions = await semantic.extract(image_bytes)
            escalated.append(page_number)
            engine = semantic.name
        else:
            regions = _lines_to_regions(lines)
            engine = engine_name

        return OCRPageResult(
            page_number=page_number,
            regions=regions,
            engine=engine,
            width=width,
            height=height,
        )

    async def process_page(
        self, document_id: str, page_number: int, image_bytes: bytes
    ) -> OCRPageResult:
        """Single-page OCR (OCRProcessor-compatible) with a fresh escalation budget."""
        width, height = await _read_size(image_bytes)
        return await self._process_page(
            page_number, image_bytes, escalated=[], width=width, height=height
        )

    async def process_document(self, document_id: str, pages: list[tuple[int, bytes]]) -> OCRResult:
        """Process every page, tracking the per-document escalation budget."""
        escalated: list[int] = []
        results: list[OCRPageResult] = []
        for page_number, image_bytes in pages:
            width, height = await _read_size(image_bytes)
            result = await self._process_page(page_number, image_bytes, escalated, width, height)
            results.append(result)
        return OCRResult(document_id=document_id, pages=results, escalated_pages=escalated)


__all__ = ["OCRRouter", "OCRRouterConfig"]
