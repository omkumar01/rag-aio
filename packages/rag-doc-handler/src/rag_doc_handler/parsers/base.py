"""Parser contracts and the plugin registry.

Each concrete parser is a small class with:
  * ``supported_types() -> set[str]`` -- file extensions / MIME types it handles.
  * ``async parse(source, data) -> Document`` -- the rag-core :class:`DocumentParser`
    contract (see :mod:`rag_core.protocols`).

Parsers are plain objects, not imported from rag-ocr or other packages, so this
module stays dependency-free apart from rag-core.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from rag_core.documents import Document
from rag_core.errors import UnsupportedFormatError
from rag_core.protocols import DocumentParser  # noqa: F401  (re-exported for typing)

from rag_doc_handler.version import __version__

__all__ = [
    "BaseParser",
    "ParserRegistry",
    "install_hint_for",
]


class BaseParser:
    """Base class for document parsers.

    Subclasses override :meth:`supported_types` and :meth:`parse`. Registration works
    for both instances and classes (``registry.register(MyParser)`` returns the class
    unchanged so it can double as a decorator).
    """

    @classmethod
    def supported_types(cls) -> set[str]:
        """File extensions/MIME types this parser handles (lower-cased, with dot)."""
        return set()

    async def parse(self, source: str, data: bytes) -> Document:
        raise NotImplementedError(  # pragma: no cover - enforced by subclasses
            f"{type(self).__name__} does not implement parse()"
        )

    @staticmethod
    def _stamp(document: Document, name: str) -> Document:
        """Attach parser provenance (name + version) to a document."""
        document.parser_name = name
        document.parser_version = __version__
        # content_hash was computed in __post_init__ from source_uri+text; nothing
        # changes here so the hash stays stable.
        return document


# Mapping of file extensions (lowercase, with dot) to the optional extra that
# provides their parser. Used to give actionable install hints.
_OPTIONAL_EXTRA: dict[str, str] = {
    ".msg": "msg",
    ".rtf": "rtf",
    ".doc": "docling",
    ".xls": "docling",
    ".ppt": "docling",
    ".xlsm": "docling",
    ".pptm": "docling",
    ".dotx": "docling",
    ".epub": "docling",
}


def install_hint_for(extension_or_mime: str) -> str | None:
    """Return an install hint for an unsupported extension, or ``None``."""
    ext = extension_or_mime.lower()
    extra = _OPTIONAL_EXTRA.get(ext)
    if extra is None:
        return None
    return f"Install the '{extra}' extra: `uv add -e .[{extra}]`"


class ParserRegistry:
    """Maps file extensions / MIME types to parser instances."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: type[BaseParser] | BaseParser) -> type[BaseParser] | BaseParser:
        """Register a parser instance or class.

        Usable as a decorator on a class::

            @registry.register
            class MyParser(BaseParser): ...
        """
        instance: BaseParser = parser() if isinstance(parser, type) else parser
        for supported in instance.supported_types():
            self._parsers[supported.lower()] = instance
        return parser

    def get(self, extension_or_mime: str) -> BaseParser:
        """Return the parser for *extension_or_mime* or raise.

        Raises :class:`~rag_core.errors.UnsupportedFormatError` (with an install hint
        in ``details`` when the extension maps to a known optional extra).
        """
        parser = self.get_or_none(extension_or_mime)
        if parser is None:
            hint = install_hint_for(extension_or_mime)
            details: dict[str, Any] = {"extension": extension_or_mime}
            if hint is not None:
                details["install_hint"] = hint
                message = f"No parser registered for {extension_or_mime!r}; {hint}"
            else:
                message = f"No parser registered for {extension_or_mime!r}"
            raise UnsupportedFormatError(message, details=details)
        return parser

    def get_or_none(self, extension_or_mime: str) -> BaseParser | None:
        return self._parsers.get(extension_or_mime.lower())

    def registered_types(self) -> set[str]:
        return set(self._parsers.keys())

    def __iter__(self) -> Iterator[tuple[str, BaseParser]]:
        return iter(self._parsers.items())
