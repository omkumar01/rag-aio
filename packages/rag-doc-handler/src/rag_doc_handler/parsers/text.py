"""Plain-text and Markdown parsing."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from rag_core.documents import Document, DocumentPage
from rag_core.ids import new_id

from rag_doc_handler.detection import detect_extension
from rag_doc_handler.parsers.base import BaseParser

__all__ = ["TextParser"]

_MARKDOWN_EXTS = {".md", ".markdown"}


def _decode(data: bytes) -> str:
    """Decode *data* as UTF-8, falling back to latin-1 (never raises)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


class TextParser(BaseParser):
    """Parses plain text (``.txt``) and Markdown (``.md``/``.markdown``)."""

    supported_types_value: ClassVar[set[str]] = {".txt", ".md", ".markdown"}

    @classmethod
    def supported_types(cls) -> set[str]:
        return set(cls.supported_types_value)

    async def parse(self, source: str, data: bytes) -> Document:
        text = await asyncio.to_thread(_decode, data)
        doc = Document(source_uri=source, text=text)
        doc.metadata.custom["encoding"] = "utf-8" if _is_utf8(data) else "latin-1"
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=text, width=None, height=None)]
        if detect_extension(source) in _MARKDOWN_EXTS:
            doc.metadata.custom["format"] = "markdown"
        return self._stamp(doc, "text")
