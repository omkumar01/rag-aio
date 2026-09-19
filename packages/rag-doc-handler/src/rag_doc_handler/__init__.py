"""rag-doc-handler: unified document ingestion and parsing layer.

Public API
----------
Detection: :func:`detect_extension`, :func:`sniff_mime`, :func:`guess_mime`
Loaders: :class:`FileLoader`, :class:`WebLoader`, :class:`LoaderLimits`
Security: :func:`validate_public_url`
Parsers: :class:`TextParser`, :class:`HTMLParser`, :class:`PDFParser`,
:class:`DocxParser`, :class:`XlsxParser`, :class:`PptxParser`, :class:`EmlParser`
Registry/Plumbing: :class:`ParserRegistry`, :class:`BaseParser`
Pipeline: :class:`IngestionPipeline`, :class:`IngestionPipelineConfig`, :class:`DedupIndex`
Crawling: :class:`SitemapCrawler`, :class:`SiteCrawler`, :class:`CrawlLimits`
"""

from __future__ import annotations

from rag_doc_handler.detection import detect_extension, guess_mime, sniff_mime
from rag_doc_handler.loader import FileLoader, LoaderLimits, WebLoader
from rag_doc_handler.models import LoadedSource
from rag_doc_handler.parsers.base import BaseParser, ParserRegistry
from rag_doc_handler.parsers.email import EmlParser
from rag_doc_handler.parsers.html import HTMLParser
from rag_doc_handler.parsers.office import DocxParser, PptxParser, XlsxParser
from rag_doc_handler.parsers.pdf import PDFParser
from rag_doc_handler.parsers.text import TextParser
from rag_doc_handler.pipeline import (
    DedupIndex,
    IngestionPipeline,
    IngestionPipelineConfig,
)
from rag_doc_handler.security import validate_public_url
from rag_doc_handler.sitemap import CrawlLimits, SiteCrawler, SitemapCrawler
from rag_doc_handler.version import __version__

__all__ = [
    "BaseParser",
    "CrawlLimits",
    "DedupIndex",
    "DocxParser",
    "EmlParser",
    "FileLoader",
    "HTMLParser",
    "IngestionPipeline",
    "IngestionPipelineConfig",
    "LoadedSource",
    "LoaderLimits",
    "PDFParser",
    "ParserRegistry",
    "PptxParser",
    "SiteCrawler",
    "SitemapCrawler",
    "TextParser",
    "WebLoader",
    "XlsxParser",
    "__version__",
    "default_pipeline",
    "default_registry",
    "detect_extension",
    "guess_mime",
    "parser_registry",
    "sniff_mime",
    "validate_public_url",
]


def default_registry() -> ParserRegistry:
    """Build a registry with all core parsers and importable optional parsers."""
    from rag_doc_handler.parsers.optional import available_parsers

    registry = ParserRegistry()
    registry.register(TextParser())
    registry.register(HTMLParser())
    registry.register(PDFParser())
    registry.register(DocxParser())
    registry.register(XlsxParser())
    registry.register(PptxParser())
    registry.register(EmlParser())
    for parser in available_parsers():
        registry.register(parser)
    return registry


def default_pipeline(
    loader: FileLoader | None = None,
    *,
    registry: ParserRegistry | None = None,
    config: IngestionPipelineConfig | None = None,
) -> IngestionPipeline:
    """A ready-to-use pipeline backed by :func:`default_registry`."""
    return IngestionPipeline(
        registry=registry or default_registry(),
        loader=loader or FileLoader(),
        config=config,
    )


# Module-level default registry / pipeline for convenience.
parser_registry: ParserRegistry = default_registry()
