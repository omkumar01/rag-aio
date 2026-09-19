"""PDF parsing via PyMuPDF (pymupdf), with an optional OCR fallback for sparse pages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag_core.documents import Document, DocumentAsset, DocumentPage, PageBlock
from rag_core.ids import new_id

from rag_doc_handler.parsers.base import BaseParser

# Imported via ``__import__`` so mypy treats pymupdf as ``Any``. The installed PyMuPDF
# ships ``py.typed`` stubs that reference ``numpy`` stubs requiring Python 3.12+
# syntax, which would abort strict checking under ``python_version = "3.11"``.
fitz: Any = __import__("pymupdf")

__all__ = ["OCRFallback", "PDFParser"]

#: Minimum characters of machine text on a page before the OCR fallback is invoked.
DEFAULT_MIN_CHARS = 20


@runtime_checkable
class OCRFallback(Protocol):
    """Callable invoked when a PDF page has too little machine text.

    ``document_id`` is a hint (the in-progress :class:`Document` id), ``page_number``
    is 1-based, ``image`` is the page rendered to PNG bytes. Returns page blocks that
    will be appended to the page's :attr:`~rag_core.documents.DocumentPage.blocks`.
    """

    async def __call__(
        self, document_id: str, page_number: int, image: bytes
    ) -> list[PageBlock]: ...


@dataclass(slots=True)
class _PageResult:
    number: int
    text: str
    width: float
    height: float
    image_xrefs: list[int] = field(default_factory=list)
    pixmap: bytes | None = None


@dataclass(slots=True)
class _PdfResult:
    pages: list[_PageResult]
    title: str | None = None
    author: str | None = None


def _render_page_png(page: fitz.Page) -> bytes:
    return bytes(page.get_pixmap().tobytes("png"))


def _parse_pdf(data: bytes, need_pixmap: bool) -> _PdfResult:
    """Synchronous PyMuPDF extraction — always run from a worker thread."""
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        metadata = pdf.metadata or {}
        pages: list[_PageResult] = []
        for index, page in enumerate(pdf):
            text = page.get_text("text")
            rect = page.rect
            xrefs: list[int] = []
            for img in page.get_images(full=True):
                # PyMuPDF returns (xref, sm, ...) tuples in most versions; be robust.
                xref = img[0] if isinstance(img, tuple) else img.get("xref")
                if xref is not None:
                    xrefs.append(int(xref))
            pages.append(
                _PageResult(
                    number=index + 1,
                    text=text,
                    width=float(rect.width),
                    height=float(rect.height),
                    image_xrefs=xrefs,
                    pixmap=_render_page_png(page) if need_pixmap else None,
                )
            )
        return _PdfResult(
            pages=pages,
            title=metadata.get("title"),
            author=metadata.get("author"),
        )
    finally:
        pdf.close()


class PDFParser(BaseParser):
    """Parses ``.pdf`` documents with PyMuPDF.

    If *ocr_fallback* is provided, pages whose extracted text is shorter than
    *min_chars* are rendered to PNG and handed to the callback; the returned
    :class:`PageBlock` objects are appended to that page.
    """

    def __init__(
        self,
        ocr_fallback: OCRFallback | None = None,
        min_chars: int = DEFAULT_MIN_CHARS,
    ) -> None:
        self._ocr: OCRFallback | None = ocr_fallback
        self._min_chars = min_chars

    @classmethod
    def supported_types(cls) -> set[str]:
        return {".pdf", "application/pdf"}

    async def parse(self, source: str, data: bytes) -> Document:
        need_pixmap = self._ocr is not None
        result = await asyncio.to_thread(_parse_pdf, data, need_pixmap)

        page_texts: list[str] = []
        pages: list[DocumentPage] = []
        assets: list[DocumentAsset] = []
        for pr in result.pages:
            page_texts.append(pr.text)
            pages.append(
                DocumentPage(
                    id=new_id(),
                    page_number=pr.number,
                    text=pr.text,
                    width=pr.width,
                    height=pr.height,
                    blocks=[],
                )
            )
            for xref in pr.image_xrefs:
                assets.append(
                    DocumentAsset(
                        asset_id=f"xref-{xref}",
                        kind="image",
                        page_number=pr.number,
                        content_ref=str(xref),
                    )
                )

        doc = Document(source_uri=source, text="\n\n".join(page_texts))
        doc.metadata.title = result.title
        doc.metadata.author = result.author
        doc.pages = pages
        doc.assets = assets

        if self._ocr is not None:
            for page, pr in zip(pages, result.pages, strict=True):
                if len(pr.text) < self._min_chars and pr.pixmap is not None:
                    regions = await self._ocr(doc.id, page.page_number, pr.pixmap)
                    page.blocks.extend(regions)

        return self._stamp(doc, "pdf")
