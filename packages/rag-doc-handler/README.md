# rag-doc-handler

Unified document ingestion and parsing for rag-aio: MIME/type detection, a plugin
parser registry, local/web loaders with SSRF protection, sitemap and site crawling,
stable content hashes, duplicate detection, and canonical `Document` output.

`rag-doc-handler` consumes raw source bytes (a local file or a remote URL) and
produces a `rag_core.documents.Document` — the same canonical model every downstream
stage (chunking, embedding, indexing) operates on. It is the single front door of the
data plane (see [architecture.md](../../architecture.md)): `rag-doc-handler → rag-ocr →
rag-embedder → rag-db-handler`.

## Installation

```bash
uv add -e packages/rag-doc-handler
```

Optional extras install the parsers that need a heavier dependency and register
themselves only when importable:

```bash
uv add -e packages/rag-doc-handler[docling,msg,rtf]
```

## Overview

`rag-doc-handler` does three things in order: **load** raw bytes (from a file or a
validated URL), **detect** the document type cheaply (extension + magic bytes, never
by importing a heavy parser), and **parse** into a canonical `Document` via a registry
of pluggable `BaseParser` adapters. CPU-bound parse work runs off the event loop with
`asyncio.to_thread`, and every outbound URL passes through an SSRF guard before a
single byte is fetched.

| Format      | Extension            | Parser        | Extra needed   |
|-------------|----------------------|---------------|----------------|
| Text / MD   | `.txt`, `.md`, `.markdown` | `TextParser`        | — (core)     |
| HTML        | `.html`, `.htm`      | `HTMLParser`  | — (core)       |
| PDF         | `.pdf`               | `PDFParser`   | — (core; `pymupdf`) |
| Word        | `.docx`              | `DocxParser`  | — (core; `python-docx`) |
| Excel       | `.xlsx`              | `XlsxParser`  | — (core; `openpyxl`) |
| PowerPoint  | `.pptx`              | `PptxParser`  | — (core; `python-pptx`) |
| Email       | `.eml`               | `EmlParser`   | — (core; stdlib `email`) |
| MSG (Outlook) | `.msg`             | `MsgParser`   | `[msg]` (`extract-msg`) |
| RTF         | `.rtf`               | `RtfParser`   | `[rtf]` (`striprtf`) |
| Legacy office, `.epub`, `.csv`, … | various | `DoclingParser` | `[docling]` (`docling`) |

Optional parsers raise `rag_core.errors.UnsupportedFormatError` (with an install hint
in `details["install_hint"]`) when their dependency is missing, and are only added to
the default registry by `default_registry()` when importable — so the core package is
usable without any extras.

## Architecture / Design Principles

- **Contracts first (ADR-0002).** Parsers implement the
  `rag_core.protocols.DocumentParser` contract (`supported_types`, `async parse`);
  the loader implements `SourceLoader`, a richer sibling of
  `rag_core.protocols.DocumentLoader` that also carries the sniffed MIME type.
- **Canonical boundary model.** Everything produces a `rag_core.documents.Document`
  with pages, blocks, assets, and a stable `content_hash` derived from
  `source_uri + text` (via `rag_core.ids.content_hash`). Deduplication and
  incremental ingestion key off this hash.
- **Plugin registry.** `ParserRegistry.register` accepts an instance *or* a class (so
  it doubles as a decorator) and indexes parsers by every supported extension and
  MIME type. `get` raises `UnsupportedFormatError` with an actionable install hint
  for known optional formats; `get_or_none` returns `None` for silent fallback.
- **Lazy, cheap detection.** `detect_extension`, `sniff_mime` and `guess_mime` do not
  import heavy parsers — they work on paths and magic bytes so the pipeline can pick a
  parser (or fail fast with a hint) without paying for `pymupdf`/`openpyxl` on every
  file.
- **Async-first, CPU off the loop.** All parsers and loaders perform blocking work via
  `asyncio.to_thread` so a single ingestion batch never stalls the event loop
  (see design.md: "inference runs off the event loop").
- **SSRF safety.** `validate_public_url` resolves hostnames with
  `socket.getaddrinfo` and refuses private/loopback/link-local/reserved/multicast/
  unspecified IPs. `WebLoader` and the crawlers call it on every outbound URL; pass
  `allow_private=True` only for intentional localhost testing.
- **Opt-in OCR, duck-typed.** `PDFParser`'s `ocr_fallback` is a `Protocol`
  (`(document_id, page_number, image_bytes) -> list[PageBlock]`) — `rag-ocr`'s
  `OCRPipeline.process_page` satisfies it without a hard dependency, preserving the
  data-plane direction (doc-handler → ocr, never the reverse).
- **Provenance.** Every `Document` records `parser_name` and `parser_version`;
  `PageBlock`s carry `order`, block `kind`, and optional `bbox`; assets and extracted
  images keep their source references.

### Source tree

```
src/rag_doc_handler/
├── __init__.py            # public API re-exports + default_registry / default_pipeline
├── version.py             # __version__
├── detection.py           # detect_extension / sniff_mime / guess_mime
├── loader.py              # FileLoader / WebLoader / LoaderLimits / SourceLoader
├── security.py            # validate_public_url (SSRF guard)
├── sitemap.py             # SitemapCrawler / SiteCrawler / CrawlLimits
├── pipeline.py            # IngestionPipeline / IngestionPipelineConfig / DedupIndex
├── models.py              # LoadedSource
└── parsers/
    ├── __init__.py
    ├── base.py            # BaseParser / ParserRegistry / install_hint_for
    ├── text.py            # TextParser
    ├── html.py            # HTMLParser
    ├── pdf.py             # PDFParser (OCR fallback protocol)
    ├── office.py          # DocxParser / XlsxParser / PptxParser
    ├── email.py           # EmlParser
    └── optional.py        # MsgParser / RtfParser / DoclingParser (lazy, guarded)
```

## Public API

```python
from rag_doc_handler import (
    # detection
    detect_extension, sniff_mime, guess_mime,
    # sources / loaders
    LoadedSource, FileLoader, WebLoader, LoaderLimits, SourceLoader,
    # security
    validate_public_url,
    # parsers + registry
    BaseParser, ParserRegistry,
    TextParser, HTMLParser, PDFParser, OCRFallback,
    DocxParser, XlsxParser, PptxParser,
    EmlParser, MsgParser, RtfParser, DoclingParser,
    # pipeline / dedup
    IngestionPipeline, IngestionPipelineConfig, DedupIndex,
    # crawling
    SitemapCrawler, SiteCrawler, CrawlLimits,
    # convenience
    default_registry, default_pipeline, parser_registry,
    __version__,
)
```

### Parsers and the registry

```python
from rag_doc_handler import BaseParser, ParserRegistry

class UpperCaseParser(BaseParser):
    @classmethod
    def supported_types(cls) -> set[str]:
        return {".upper"}

    async def parse(self, source: str, data: bytes) -> Document:
        text = data.decode("utf-8").upper()
        doc = Document(source_uri=source, text=text)
        # _stamp() writes parser_name/parser_version provenance for you
        return self._stamp(doc, "uppercase")

registry = ParserRegistry()
registry.register(UpperCaseParser)        # or: @registry.register as a decorator
parser = registry.get(".upper")           # -> UpperCaseParser
await parser.parse("note.upper", b"hi")   # -> Document with text "HI"
```

`ParserRegistry.get_or_none(key)` returns `None` instead of raising, enabling a
"try, then fall back" pattern. `install_hint_for(".msg")` returns a ready-made
`uv add` command for known optional extras.

### Loaders

```python
from rag_doc_handler import FileLoader, WebLoader, LoaderLimits, LoadedSource

FileLoader(limits=LoaderLimits(max_bytes=100 * 1024 * 1024))   # default 100 MiB
    .load("/path/to/file.pdf")                                # -> LoadedSource

async with httpx.AsyncClient() as client:
    WebLoader(client, LoaderLimits(max_bytes=5 * 1024 * 1024)).load("https://...")
```

`LoadedSource` carries `source_uri`, `data`, optional `mime_type`, and a `metadata`
dict (e.g. `{"content_type": ..., "final_url": ...}`).

### PDF parser with OCR fallback

`PDFParser` accepts a duck-typed `ocr_fallback` callable and a `min_chars` threshold
(default `20`). Pages whose machine-extracted text is shorter than `min_chars` are
rendered to PNG via PyMuPDF and handed to the callback:

```python
from rag_doc_handler import PDFParser
from rag_ocr import OCRPipeline  # rag-ocr's OCRPipeline satisfies OCRFallback

pdf_parser = PDFParser(ocr_fallback=OCRPipeline(...).process_page, min_chars=20)
```

### Crawlers

```python
from rag_doc_handler import SiteCrawler, SitemapCrawler, CrawlLimits
import httpx

limits = CrawlLimits(max_pages=50, max_depth=2, max_concurrency=8)
async with httpx.AsyncClient() as client:
    pages = await SiteCrawler(client, limits).crawl("https://example.com")          # list[(url, bytes)]
    urls  = await SitemapCrawler(client, limits).discover("https://example.com/sitemap.xml")
```

### Detection helpers

```python
from rag_doc_handler import detect_extension, sniff_mime, guess_mime

detect_extension("https://x.com/a/p.pdf?q=1")   # -> ".pdf"
sniff_mime(b"%PDF-1.4 ...")                    # -> "application/pdf"
guess_mime("a.html", b"<html>...")             # sniffs first, then falls back to mimetypes
```

## Usage Guides

### Beginner — ingest a local file (with deduplication)

```python
import asyncio
from rag_doc_handler import default_pipeline, DedupIndex

async def main() -> None:
    pipeline = default_pipeline()
    dedup = DedupIndex()
    doc, is_new = await pipeline.ingest_file("/data/report.pdf", dedup=dedup)
    doc2, is_new2 = await pipeline.ingest_file("/data/report.pdf", dedup=dedup)
    print(is_new, is_new2)                     # True False  (duplicate suppressed)
    print(doc.content_hash == doc2.content_hash)
    print(doc.metadata.title, doc.parser_name)  # from PDF metadata / "pdf"

asyncio.run(main())
```

### Beginner — ingest raw bytes with no file

```python
from rag_doc_handler import default_pipeline

doc, _ = await default_pipeline().ingest("note.html", b"<h1>Hello</h1><p>World</p>")
print(doc.text)                       # "Hello\nWorld"
print(doc.pages[0].blocks[0].kind)    # "heading"
```

### Intermediate — fetch and parse a remote page (SSRF-safe)

```python
import asyncio, httpx
from rag_doc_handler import WebLoader, LoaderLimits, default_registry, IngestionPipeline

async def main() -> None:
    async with httpx.AsyncClient() as client:
        loader = WebLoader(client, LoaderLimits(max_bytes=5 * 1024 * 1024))
        pipeline = IngestionPipeline(default_registry(), loader)
        doc, _ = await pipeline.ingest("https://example.com/page.html")
        print(doc.text[:120])

asyncio.run(main())
```

`WebLoader` validates every URL with `validate_public_url`, which blocks
private/loopback/link-local/reserved IPs to prevent SSRF (`allow_private=True`
opt into localhost only).

### Intermediate — crawl a site and ingest the results

```python
import asyncio, httpx
from rag_doc_handler import SiteCrawler, SitemapCrawler, CrawlLimits, default_pipeline, IngestionPipeline, WebLoader

async def main() -> None:
    limits = CrawlLimits(max_pages=50, max_depth=2, max_concurrency=8)
    async with httpx.AsyncClient() as client:
        # sitemap first (cheap, complete), then BFS crawl for discovery
        urls   = await SitemapCrawler(client, limits).discover("https://example.com/sitemap.xml")
        pages  = await SiteCrawler(client, limits).crawl("https://example.com")
        pipeline = IngestionPipeline(default_registry(), WebLoader(client))
        for url, bytes_ in pages:
            doc, _ = await pipeline.ingest(url, bytes_)   # ingest bytes already in hand
            print(url, "->", doc.parser_name, len(doc.pages), "pages")

asyncio.run(main())
```

### Advanced — register a custom parser and use a dedicated pipeline

```python
from rag_doc_handler import BaseParser, ParserRegistry, IngestionPipeline, IngestionPipelineConfig, FileLoader
from rag_core.documents import Document

class CsvParser(BaseParser):
    @classmethod
    def supported_types(cls) -> set[str]:
        return {".csv", "text/csv"}

    async def parse(self, source: str, data: bytes) -> Document:
        import asyncio
        rows = await asyncio.to_thread(lambda: data.decode("utf-8").splitlines())
        text = "\n".join(rows)
        doc = Document(source_uri=source, text=text)
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=text, blocks=[])]
        return self._stamp(doc, "csv")

registry = ParserRegistry()
registry.register(CsvParser())
pipeline = IngestionPipeline(registry, FileLoader(), config=IngestionPipelineConfig(auto_detect=True))
doc, _ = await pipeline.ingest_file("/data/numbers.csv")
```

### Advanced — inject HTTP transports for testing the semantic/VLM path

While `rag-ocr` is the package that calls a VLM, `rag-doc-handler`'s `PDFParser` OCR
fallback is duck-typed, so you can wire a mock engine and assert only doc-handler
behavior:

```python
from rag_doc_handler import PDFParser

async def fake_ocr(document_id, page_number, image_png):
    return []   # no-op; real extractors return list[PageBlock]

parser = PDFParser(ocr_fallback=fake_ocr, min_chars=0)   # force OCR on every page
```

## Configuration

### `IngestionPipelineConfig`

| Field         | Type   | Default | Notes                                   |
| ------------- | ------ | ------- | --------------------------------------- |
| `auto_detect` | `bool` | `True`  | Let the pipeline sniff MIME if not given. |

The pipeline uses extension first, then MIME type, to select a parser from the
`ParserRegistry`. When neither matches, it raises
`rag_core.errors.UnsupportedFormatError` with `details["install_hint"]` populated
for known optional formats (e.g. `.msg`, `.rtf`, `.doc`).

### `LoaderLimits` / `CrawlLimits`

| Field             | Default            | Notes                                  |
| ----------------- | ------------------ | -------------------------------------- |
| `LoaderLimits.max_bytes`   | `100 * 1024**2` | Hard cap on loaded file/URL size.        |
| `CrawlLimits.max_pages`    | `500`          | Stop discovery after this many pages.  |
| `CrawlLimits.max_depth`    | `1`          | BFS depth budget.                      |
| `CrawlLimits.max_response_bytes` | `10*1024**2` | Per-response byte cap.          |
| `CrawlLimits.timeout_s`    | `30.0`        | Per-request HTTP timeout.              |
| `CrawlLimits.max_concurrency`| `8`         | Concurrent fetches (semaphore).        |
| `CrawlLimits.same_domain_only` | `True`     | Restrict SiteCrawler to the start host. |

## Testing

```bash
uv run pytest packages/rag-doc-handler -q
```

- **Unit tests** (always run): detection, parser extraction, and registry selection
  with no network. The shared `conftest.py` patches `socket.getaddrinfo` so
  `validate_public_url` classifies IP literals offline (domains resolve to a synthetic
  public IP) and no test performs real DNS/HTTP.
- **Integration tests** (`pytest.mark.integration`): exercises real `pymupdf` PDF
  rendering and full `default_pipeline()` round-trips on generated fixtures
  (e.g. a PDF built in-memory with `pymupdf` + the markdown/text parsers).

Run only fast (non-integration) tests with the repo-wide `--fast` gate
(`python scripts/check.py --fast`).

## Dependencies

`pymupdf>=1.24`, `beautifulsoup4>=4.12`, `lxml>=5.0`, `httpx>=0.27`,
`python-docx>=1.1`, `openpyxl>=3.1`, `python-pptx>=0.6.23`, and `rag-core`.

Optional extras declared in `pyproject.toml`:

| Extra     | Packages            | Parsers enabled                          |
| --------- | ------------------- | ---------------------------------------- |
| `docling` | `docling>=2.0`      | `DoclingParser` (`.doc`, `.xls`, `.epub`, …) |
| `msg`     | `extract-msg>=0.50` | `MsgParser` (`.msg`)                     |
| `rtf`     | `striprtf>=0.0.26`  | `RtfParser` (`.rtf`)                     |

## Cross-Package Relationships

- **`rag-core`** — `Document` is the canonical output of every parser; `DocumentPage`,
  `PageBlock`, `BlockKind`, `DocumentAsset`, and `DocumentMetadata` shape that output;
  `UnsupportedFormatError`/`IngestionError`/`CrawlError` are raised by type detection,
  the loader, and the crawlers; `errors.ConfigError` is raised by config; `ids.new_id`
  and `ids.content_hash` seed page/block ids and drive dedup. Parsers satisfy the
  `rag_core.protocols.DocumentParser` contract.
- **`rag-ocr`** — `PDFParser`'s `ocr_fallback` accepts any `OCRFallback`-compatible
  callable; `rag-ocr`'s `OCRPipeline.process_page` satisfies it. OCR is *consumed* by
  doc-handler, never the reverse (data-plane direction is doc-handler → ocr). `rag-ocr`
  also produces `PageBlock`s that can be fed back into the document's pages.
- **`rag-embedder`** — consumes the `Document`s produced here for chunking and
  embedding; treats doc-handler as an opaque producer of canonical documents.
- **`rag-mass-inject`** — the bulk ingestion orchestrator drives many
  `IngestionPipeline` instances, passing `DedupIndex` (or backing it with a
  `rag-db-handler` `KeyValueStore` for durable, multi-process dedup) and aggregating
  crawl results into ingestion batches.
- **`rag-orchestrator`** — wires a pipeline and loader together from
  configuration and may expose them over the FastAPI service boundary (ADR-0005) with
  typed ingest/crawl endpoints per ADR-0007.
- **`rag-cache` / `rag-observe`** — may cache parsed documents by content hash and
  instrument loader/crawler/parse spans without modifying this package.

Service boundary: like all data-plane modules, `rag-doc-handler` runs in-process for
latency; the production scale path wraps `IngestionPipeline`/`SiteCrawler` behind
FastAPI endpoints (`/health`, `/ready`, `/metrics`, and typed ingest/crawl endpoints)
per ADR-0005 and ADR-0007.
