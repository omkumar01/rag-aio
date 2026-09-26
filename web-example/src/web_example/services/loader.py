"""Source-loader routing: http(s) URLs to :class:`WebLoader`, paths to files.

The stock ``default_pipeline()`` ships with a :class:`FileLoader` only, so a
URL source would fail with "Source is not a file". :class:`SchemeRouterLoader`
dispatches by scheme so one ingestion pipeline can serve uploaded files,
server-side paths, and web/sitemap URLs. Web fetching stays inside
doc-handler's SSRF-guarded :class:`WebLoader` (public hosts only, size-capped).

:class:`MimeAwareRegistry` complements it at parse time: web servers describe
content by MIME type, and URLs like ``https://example.com/`` carry no file
extension, so response types are mapped onto the registry's extension keys.
"""

from __future__ import annotations

import httpx
from rag_doc_handler.loader import FileLoader, LoaderLimits, WebLoader
from rag_doc_handler.models import LoadedSource
from rag_doc_handler.parsers.base import BaseParser, ParserRegistry

__all__ = ["MimeAwareRegistry", "SchemeRouterLoader"]

#: Content types returned by web servers mapped onto the extension keys the
#: parser registry understands.
_MIME_ALIASES: dict[str, str] = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/pdf": ".pdf",
}


class MimeAwareRegistry(ParserRegistry):
    """Parser registry that resolves response MIME types by extension alias."""

    def __init__(self, inner: ParserRegistry) -> None:
        super().__init__()
        self._inner = inner

    def get_or_none(self, extension_or_mime: str) -> BaseParser | None:
        key = str(extension_or_mime).lower()
        parser = self._inner.get_or_none(key)
        if parser is not None:
            return parser
        alias = _MIME_ALIASES.get(key)
        if alias is None:
            return None
        return self._inner.get_or_none(alias)

    def get(self, extension_or_mime: str) -> BaseParser:
        parser = self.get_or_none(extension_or_mime)
        if parser is None:
            return self._inner.get(extension_or_mime)
        return parser


class SchemeRouterLoader:
    """Dispatches a source by scheme: http(s) → web, everything else → file."""

    def __init__(self, limits: LoaderLimits | None = None) -> None:
        self._limits = limits
        self._file_loader = FileLoader(limits)
        self._client: httpx.AsyncClient | None = None

    def _web_loader(self) -> WebLoader:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return WebLoader(self._client, self._limits)

    async def load(self, source: str) -> LoadedSource:
        if source.startswith(("http://", "https://")):
            return await self._web_loader().load(source)
        return await self._file_loader.load(source)

    async def aclose(self) -> None:
        """Close the shared HTTP client, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
