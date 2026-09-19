"""Optional parsers guarding install-only dependencies.

Each parser raises :class:`~rag_core.errors.UnsupportedFormatError` (with an install
hint) when its backing library is not present. The package builds its default registry
from :func:`available_parsers`, which only includes parsers whose dependencies import
cleanly, keeping the core package runnable without these extras.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_core.documents import Document, DocumentPage
from rag_core.errors import UnsupportedFormatError
from rag_core.ids import new_id

from rag_doc_handler.parsers.base import BaseParser

__all__ = ["DoclingParser", "MsgParser", "RtfParser", "available_parsers"]

try:  # extract-msg
    import extract_msg as _extract_msg
except ImportError:  # pragma: no cover - dependency optional
    _extract_msg = None

try:  # striprtf
    from striprtf.striprtf import strip_rtf as _strip_rtf
except ImportError:  # pragma: no cover - dependency optional
    _strip_rtf = None

try:  # docling
    from docling.document_converter import DocumentConverter as _DocumentConverter
except ImportError:  # pragma: no cover - dependency optional
    _DocumentConverter = None  # Formats that docling can convert but the core parsers cannot.
_DOCLING_TYPES = {
    ".doc",
    ".xls",
    ".ppt",
    ".xlsm",
    ".pptm",
    ".dotx",
    ".dot",
    ".epub",
    ".csv",
}


def _raise(extra: str, missing: str) -> None:
    raise UnsupportedFormatError(
        f"{missing} is required for this format but is not installed",
        details={"install_hint": f"Install the '{extra}' extra: `uv add -e .[{extra}]`"},
    )


class MsgParser(BaseParser):
    """Parses ``.msg`` emails via :mod:`extract_msg` (``msg`` extra)."""

    @classmethod
    def supported_types(cls) -> set[str]:
        return {".msg"}

    @staticmethod
    def _available() -> bool:
        return _extract_msg is not None

    async def parse(self, source: str, data: bytes) -> Document:
        if _extract_msg is None:
            _raise("msg", "extract-msg")
        msg = _extract_msg.message_from_bytes(data)
        subject = str(getattr(msg, "subject", "") or "")
        body = str(getattr(msg, "body", "") or "")
        if not body:
            body = str(getattr(msg, "as_string", "") or "")
        from_ = str(getattr(msg, "from", "") or "")
        to = str(getattr(msg, "to", "") or "")

        doc = Document(source_uri=source, text=body)
        if subject:
            doc.metadata.title = subject
        if from_:
            doc.metadata.custom["from"] = from_
        if to:
            doc.metadata.custom["to"] = to
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=body, blocks=[])]
        return self._stamp(doc, "msg")


class RtfParser(BaseParser):
    """Parses ``.rtf`` documents via :mod:`striprtf` (``rtf`` extra)."""

    @classmethod
    def supported_types(cls) -> set[str]:
        return {".rtf"}

    @staticmethod
    def _available() -> bool:
        return _strip_rtf is not None

    async def parse(self, source: str, data: bytes) -> Document:
        if _strip_rtf is None:
            _raise("rtf", "striprtf")
        text = await asyncio.to_thread(_strip_rtf_obj, data)
        doc = Document(source_uri=source, text=text)
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=text, blocks=[])]
        return self._stamp(doc, "rtf")


def _strip_rtf_obj(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return str(_strip_rtf(text))


class DoclingParser(BaseParser):
    """Parses legacy/uncommon formats (``.doc``, ``.xls``, ``.ppt``, ``.epub``, ``.csv`` ...) via docling."""

    @classmethod
    def supported_types(cls) -> set[str]:
        return set(_DOCLING_TYPES)

    @staticmethod
    def _available() -> bool:
        return _DocumentConverter is not None

    async def parse(self, source: str, data: bytes) -> Document:
        if _DocumentConverter is None:
            _raise("docling", "docling")
        from io import BytesIO

        converter = _DocumentConverter()
        text = await asyncio.to_thread(_convert_with_docling, converter, BytesIO(data), source)
        doc = Document(source_uri=source, text=text)
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=text, blocks=[])]
        return self._stamp(doc, "docling")


def _convert_with_docling(converter: Any, stream: Any, source: str) -> str:
    converted = converter.convert(source, input_stream=stream)
    document = converted.document
    try:
        return str(document.export_to_text("markdown"))
    except AttributeError:
        return str(document.text) if hasattr(document, "text") else ""


def available_parsers() -> list[BaseParser]:
    """Return instances of optional parsers whose dependencies are importable."""
    parsers: list[BaseParser] = []
    if MsgParser._available():
        parsers.append(MsgParser())
    if RtfParser._available():
        parsers.append(RtfParser())
    if DoclingParser._available():
        parsers.append(DoclingParser())
    return parsers
