"""End-to-end OCR pipeline: preprocess -> route -> produce :class:`OCRResult`.

PDF rendering is intentionally absent (no ``pymupdf``/``fitz`` dependency): callers
from ``rag-doc-handler`` render PDF pages to images and feed them here as
``list[tuple[int, bytes]]`` of ``(page_number, image_bytes)``.
"""

from __future__ import annotations

import asyncio
import io

from PIL import Image
from rag_core.documents import PageBlock
from rag_core.ids import new_id

from rag_ocr.models import OCRPageResult, OCRResult
from rag_ocr.preprocess import PreprocessedPage, image_to_bytes, preprocess
from rag_ocr.routing import OCRRouter


def regions_to_blocks(page_result: OCRPageResult, page_id: str) -> list[PageBlock]:
    """Project the regions of a page result into canonical :class:`PageBlock`s.

    Region order is preserved (``order`` field), so reading order stays stable.
    Each block retains the OCR region's bbox, kind, confidence and text, remaining
    traceable back to ``page_id`` and, transitively, the document/page number on
    ``page_result``.
    """
    blocks: list[PageBlock] = []
    for order, region in enumerate(page_result.regions):
        blocks.append(
            PageBlock(
                page_id=page_id,
                kind=region.kind,
                text=region.text,
                bbox=region.bbox,
                confidence=region.confidence,
                order=order,
            )
        )
    return blocks


class OCRPipeline:
    """Preprocesses images and routes them through an :class:`OCRRouter`."""

    def __init__(
        self,
        router: OCRRouter,
        preprocess_enabled: bool = True,
        max_pages: int = 200,
    ) -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self._router = router
        self._preprocess_enabled = preprocess_enabled
        self._max_pages = max_pages

    @staticmethod
    def _prepare(image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
        prepared: PreprocessedPage = preprocess(img)
        return image_to_bytes(prepared.image)

    async def process_images(
        self,
        images: list[tuple[int, bytes]],
        document_id: str | None = None,
    ) -> OCRResult:
        """Recognise ``images`` (page_number, image bytes) into an :class:`OCRResult`.

        ``document_id`` is generated when omitted. Pages beyond ``max_pages`` are
        silently truncated.
        """
        doc_id = document_id or new_id()
        pages = images[: self._max_pages]

        if self._preprocess_enabled:
            prepared: list[tuple[int, bytes]] = []
            for page_number, image_bytes in pages:
                prepared_bytes = await asyncio.to_thread(self._prepare, image_bytes)
                prepared.append((page_number, prepared_bytes))
            pages = prepared

        return await self._router.process_document(doc_id, pages)


__all__ = ["OCRPipeline", "regions_to_blocks"]
