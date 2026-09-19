# rag-doc-handler

Unified ingestion and parsing for rag-aio: MIME/type detection, a plugin parser
registry, local/web loaders with SSRF protection, sitemap & site crawling, stable
content hashes, duplicate detection, and canonical `Document` output.

`rag-doc-handler` consumes raw source bytes (a local file or a remote URL) and
produces a `rag_core.documents.Document` — the same canonical model every downstream
stage (chunking, embedding, indexing) operates on.

## Installation

```bash
uv add -e packages/rag-doc-handler
```

Optional extras install the parsers that need a heavier dependency:

```bash
uv add -e packages/rag-doc-handler[docling,msg,rtf]
```

## Public API

```python
from rag_doc_handler import (
    # detection
    detect_extension,
    sniff_mime,
    guess_mime,
    # sources / loaders
    LoadedSource,
    FileLoader,
    WebLoader,
    LoaderLimits,
    # security
    validate_public_url,
    # parsers + registry
    ParserRegistry,
    TextParser,
    HTMLParser,
    PDFParser,
    DocxParser,
    XlsxParser,
    PptxParser,
    EmlParser,
    # pipeline
    IngestionPipeline,
    IngestionPipelineConfig,
    DedupIndex,
    # crawling
    CrawlLimits,
    SitemapCrawler,
    SiteCrawler,
    # convenience
    default_registry,
    default_pipeline,
    parser_registry,
    __version__,
)
```

## Parser support matrix

| Format      | Extension            | Parser        | Extra needed |
|-------------|----------------------|---------------|--------------|
| Text / MD   | `.txt`, `.md`, `.markdown` | `TextParser`        | — (core)     |
| HTML        | `.html`, `.htm`      | `HTMLParser`  | — (core)     |
| PDF         | `.pdf`               | `PDFParser`   | — (core; `pymupdf`) |
| Word        | `.docx`              | `DocxParser`  | — (core; `python-docx`) |
| Excel       | `.xlsx`              | `XlsxParser`  | — (core; `openpyxl`) |
| PowerPoint  | `.pptx`              | `PptxParser`  | — (core; `python-pptx`) |
| Email       | `.eml`               | `EmlParser`   | — (core; stdlib `email`) |
| MSG (Outlook) | `.msg`             | `MsgParser`   | `[msg]` (`extract-msg`) |
| RTF         | `.rtf`               | `RtfParser`   | `[rtf]` (`striprtf`) |
| Doc/Legacy office, `.epub`, `.csv`, … | various | `DoclingParser` | `[docling]` (`docling`) |

Optional parsers raise `UnsupportedFormatError` (with an install hint) when their
dependency is missing, and are only registered by `default_registry()` when importable.

### PDF OCR fallback

`PDFParser(ocr_fallback=..., min_chars=20)` renders (via PyMuPDF) and calls an
`OCRProcessor`-like callable `(document_id, page_number, image_png) -> list[PageBlock]`
for any page whose machine text is shorter than `min_chars`, attaching the returned
blocks to that page. The callback is duck-typed, so `rag-ocr` can be supplied without
this package depending on it.

## Quick start

### Ingest a local file (with dedup)

```python
import asyncio
from rag_doc_handler import default_pipeline, DedupIndex


async def main() -> None:
    pipeline = default_pipeline()
    dedup = DedupIndex()
    doc, is_new = await pipeline.ingest_file("/data/report.pdf", dedup=dedup)
    doc2, is_new2 = await pipeline.ingest_file("/data/report.pdf", dedup=dedup)
    assert is_new is True and is_new2 is False  # duplicate suppressed


asyncio.run(main())
```

### Ingest from raw bytes

```python
doc, _ = await pipeline.ingest("note.html", html_bytes)
```

### Fetch and parse from the web

```python
import httpx
from rag_doc_handler import WebLoader, LoaderLimits, default_registry, IngestionPipeline


async def main() -> None:
    async with httpx.AsyncClient() as client:
        loader = WebLoader(client, LoaderLimits(max_bytes=5 * 1024 * 1024))
        pipeline = IngestionPipeline(default_registry(), loader)
        doc, _ = await pipeline.ingest("https://example.com/page.html")


asyncio.run(main())
```

`WebLoader` and the crawlers validate every outbound URL with
`validate_public_url`, which blocks private/loopback/link-local/reserved IPs to prevent
SSRF (`./...` and `allow_private=True` opt into localhost testing).

### Crawl a site

```python
import httpx
from rag_doc_handler import SiteCrawler, CrawlLimits, SitemapCrawler


async def main() -> None:
    limits = CrawlLimits(max_pages=50, max_depth=2, max_concurrency=8)
    async with httpx.AsyncClient() as client:
        pages = await SiteCrawler(client, limits).crawl("https://example.com")
        urls = await SitemapCrawler(client, limits).discover("https://example.com/sitemap.xml")


asyncio.run(main())
```

## Design notes

- **Async-first.** CPU-bound parse/load work is offloaded with `asyncio.to_thread` so the
  event loop is never blocked.
- **Plugin parsers.** `ParserRegistry.register` accepts an instance or class (usable as a
  decorator); `get(extension_or_mime)` raises `UnsupportedFormatError` with an install
  hint for known optional formats.
- **Stable deduplication.** `Document.content_hash` is derived from `source_uri` and
  normalized text; `DedupIndex` tracks seen hashes in-process (single event loop).
- **SSRF safety.** `validate_public_url` resolves hostnames with `socket.getaddrinfo` and
  rejects non-public IPs unless `allow_private=True`.
