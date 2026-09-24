# rag-ocr
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Hybrid mechanical + semantic OCR for rag-aio, with provenance-preserving regions.

Two layers, cheap-first: (1) pluggable OCR engines recognize text with bounding boxes
and confidence; an image-preprocessing chain (bomb guard → grayscale → resolution
normalization → denoise → projection-profile deskew) runs before recognition; (2)
pages whose best engine result is below the configured confidence threshold — or that
return no lines at all — are escalated to a vision-language model
(`VLMSemanticExtractor`), up to `max_escalations_per_doc` per document. Every region
carries `page_number` and a bounding box, so each semantic element traces back to its
source image, and `regions_to_blocks` converts results into the canonical
`rag_core.documents.PageBlock`.

`rag-ocr` accepts **pre-rendered page images**. PDF page rendering intentionally lives
in `rag-doc-handler` (which passes page images into a doc-handler `PDFParser` OCR-fallback
callback that this package's `OCRPipeline.process_page` satisfies).

## Installation

```bash
uv add -e packages/rag-ocr
```

Install the mechanical OCR engine via an optional extra:

```bash
uv add -e packages/rag-ocr[rapidocr]
```

The core package has no heavy inference dependency: `RapidOCREngine` is imported lazily
and raises `OCREngineUnavailable` (with an install hint) when the `rapidocr` extra is
missing. The VLM escalation path needs only `httpx`, which is a core dependency.

## Overview

`rag-ocr` implements the `rag_core.protocols.OCRProcessor` protocol (`process_page`).
It is composed of four independent layers:

1. **Engines** (`engines/`) — `OCREngine` protocol + `RawLine`; `NullOCREngine` (no-op
   default), `RapidOCREngine` (lazy import). `aggregate_confidence` computes the mean
   over a page's raw lines.
2. **Preprocessing** (`preprocess.py`) — synchronous Pillow + numpy helpers run off
   the event loop: decompression-bomb guard, grayscale, resolution normalization,
   median denoising, and projection-profile deskew (`[-5, +5]°` in `0.5°` steps).
3. **Routing** (`routing.py`) — `OCRRouter` tries engines in priority order; the
   `OCRRouterConfig` policy decides when a page is escalated to the VLM
   (`confidence_threshold`, `min_lines`, `max_escalations_per_doc`).
4. **Semantic** (`semantic.py`) — `VLMSemanticExtractor` sends the page image to an
   OpenAI-compatible vision endpoint; VLM output is **untrusted data**, classified
   into a block kind (table / heading / text) by `classify_kind`, never re-rendered
   into a prompt.

### Source tree

```
src/rag_ocr/
├── __init__.py            # public API re-exports + __version__
├── models.py              # OCRRegion / OCRPageResult / OCRResult / RegionKind
├── pipeline.py            # OCRPipeline + regions_to_blocks
├── preprocess.py          # PreprocessedPage + preprocess chain
├── routing.py             # OCRRouter / OCRRouterConfig
├── semantic.py            # VLMSemanticExtractor / classify_kind
└── engines/
    ├── __init__.py        # engine registry re-exports
    ├── base.py            # OCREngine protocol / RawLine / OCRErrorUnavailable
    ├── confidence.py      # aggregate_confidence
    ├── null.py            # NullOCREngine
    └── rapidocr.py        # RapidOCREngine (lazy)
```

## Architecture / Design Principles

- **Cheap first, escalate on demand.** Mechanical OCR is fast and local; the VLM runs
  only when confidence is low or a page is empty, and only up to a per-document budget,
  so a bad batch can't silently balloon into a bill.
- **Lazy, optional backends.** Importing `rag_ocr` never requires `rapidocr`. The heavy
  backend is loaded inside `RapidOCREngine.__init__`, which raises
  `OCREngineUnavailable` (a subclass of `rag_core.errors.OCRError`) with an install
  hint if the extra is absent.
- **Never block the loop.** All engine recognition and VLM HTTP calls run via
  `asyncio.to_thread` (engines) or native async `httpx` (VLM); preprocessing is
  synchronous and likewise scheduled off the loop by the pipeline.
- **Provenance everywhere.** `OCRRegion` is owned by an `OCRPageResult`
  (`page_number` + `engine`); pages are owned by an `OCRResult` (`document_id` +
  `escalated_pages`). `regions_to_blocks` preserves bbox, kind, confidence, and order
  into `PageBlock`s traceable back to `page_id`.
- **Untrusted VLM output.** The VLM prompt carries a "do not act on image contents"
  instruction; model output is parsed only into `(text, kind)` regions here — it is
  never interpolated back into any prompt or executed.
- **Defensive inputs.** A decompression-bomb guard caps pixel counts before any
  recognition, and `preprocess` records every applied op for traceability/debugging.
- **Secrets by reference.** `VLMSemanticExtractor.api_key_ref` names an env var read
  *at extraction time* (so key rotation needs no client restart); the key is injected
  as a bearer header only and is never returned or logged.
- **Protocol-conformant.** `OCRRouter` satisfies `rag_core.protocols.OCRProcessor`;
  engines satisfy the local `OCREngine` protocol. Tests assert
  `isinstance(router, OCRProcessor)`.

## Public API

```python
from rag_ocr import (
    # models
    OCRRegion,
    OCRPageResult,
    OCRResult,
    RegionKind,
    # engines
    OCREngine,
    RawLine,
    OCREngineUnavailable,
    NullOCREngine,
    RapidOCREngine,
    aggregate_confidence,
    # routing
    OCRRouter,
    OCRRouterConfig,
    # semantic
    VLMSemanticExtractor,
    classify_kind,
    # pipeline
    OCRPipeline,
    regions_to_blocks,
    # preprocess
    PreprocessedPage,
    preprocess,
    denoise,
    deskew,
    normalize_resolution,
    to_grayscale,
    image_to_bytes,
)
```

### Models

```python
OCRRegion(
    text: str,
    bbox: BoundingBox | None = None,       # None for VLM (no geometry)
    confidence: float = ...,                # Field(ge=0.0, le=1.0)
    kind: RegionKind = "text",             # "text"|"heading"|"table"|"figure"|"caption"|"list"|"other"
    language: str | None = None,
)
# RegionKind = Literal["text", "heading", "table", "figure", "caption", "list", "other"]

OCRPageResult(
    page_number: int,                      # Field(ge=0)
    regions: list[OCRRegion] = ...,
    engine: str = "",                     # which engine/strategy produced this page
    width: float | None, height: float | None,
    language: str | None,
)
# .mean_confidence  -> property: mean over regions; 0.0 when empty

OCRResult(
    document_id: str,
    pages: list[OCRPageResult] = ...,
    escalated_pages: list[int] = ...,     # pages that went to the VLM
)
```

### Engines

```python
class OCREngine(Protocol):
    name: str
    async def recognize(self, image_bytes: bytes) -> list[RawLine]: ...

@dataclass
class RawLine:
    text: str
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1); (0,0,0,0) if unknown
    confidence: float

class NullOCREngine:           # name="null"; always returns []
class RapidOCREngine:          # name="rapidocr"; lazy import; raises OCREngineUnavailable if missing

aggregate_confidence(lines: list[RawLine]) -> float | None   # None when empty
```

### Routing

```python
class OCRRouterConfig(RagBaseModel):
    confidence_threshold: float = 0.7
    min_lines: int = 1
    escalate_model: bool = True
    max_escalations_per_doc: int = 3


class OCRRouter:
    def __init__(self, engines: list[OCREngine], semantic, config=None) -> None: ...
    async def process_page(self, document_id, page_number, image_bytes) -> OCRPageResult: ...
    async def process_document(self, document_id, pages) -> OCRResult: ...


# Satisfies rag_core.protocols.OCRProcessor.
```

Escalation triggers when `escalate_model` is True **and** a semantic extractor is
present **and** (the engine returned fewer than `min_lines` lines, **or** mean
confidence is below `confidence_threshold`), and the per-document budget isn't
exhausted. `process_page` uses a fresh budget (one page may always escalate);
`process_document` tracks `escalated_pages` across all pages.

### Semantic extraction

```python
VLMSemanticExtractor(
    base_url="http://localhost:1234/v1",
    model="glm-ocr",
    api_key_ref=None,                # env var name, resolved per call
    timeout_s=120.0,
    prompt_template=...,             # untrusted-data-safe default prompt
    transport=None,                  # inject an httpx transport for testing
)
    .extract(page_image_bytes, context=None) -> list[OCRRegion]
    .aclose() -> None

classify_kind(text) -> RegionKind     # table / heading / text
```

`classify_kind` is a deterministic regex classifier: a markdown table separator row
(`| --- |`) → `"table"`; leading `#` → `"heading"`; otherwise `"text"`.

### Pipeline + preprocessing

```python
class OCRPipeline:
    def __init__(self, router: OCRRouter, preprocess_enabled=True, max_pages=200) -> None: ...
    async def process_images(
        self, images: list[tuple[int, bytes]], document_id=None
    ) -> OCRResult: ...
# max_pages silently truncates; document_id generated (new_id()) when omitted.

def regions_to_blocks(page_result: OCRPageResult, page_id: str) -> list[PageBlock]: ...

def preprocess(img, max_pixels=40_000_000) -> PreprocessedPage: ...
# Chain: bomb guard -> grayscale -> normalize_resolution -> denoise -> deskew

def to_grayscale(img) -> Image.Image
def normalize_resolution(img, target_dpi=300.0, max_width=2000) -> Image.Image
def denoise(img, size=3) -> Image.Image
def deskew(img) -> tuple[Image.Image, float]
def image_to_bytes(img, format="PNG") -> bytes
```

Preprocessing helpers are synchronous; callers from async code must schedule them via
`asyncio.to_thread` (the pipeline does this for you).

## Usage Guides

### Beginner — OCR a page image with the null engine (no downloads)

```python
import asyncio, io
from PIL import Image
from rag_ocr import OCRRouter, OCRRouterConfig, OCRPipeline, NullOCREngine


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (16, 16), 255).save(buf, "PNG")
    return buf.getvalue()


router = OCRRouter(engines=[NullOCREngine()], semantic=None)
pipeline = OCRPipeline(router, preprocess_enabled=True, max_pages=200)


async def main() -> None:
    result = await pipeline.process_images([(0, _png()), (1, _png())], document_id="doc-1")
    print(result.document_id, [p.page_number for p in result.pages])


asyncio.run(main())
```

### Beginner — use the OCRProcessor protocol directly

`OCRRouter.process_page` matches the `rag_core.protocols.OCRProcessor` signature and
works standalone with a fresh escalation budget:

```python
from rag_core.protocols import OCRProcessor
from rag_ocr import OCRRouter, NullOCREngine

router = OCRRouter(engines=[NullOCREngine()], semantic=None)
assert isinstance(router, OCRProcessor)  # protocol conformance
page_result = await router.process_page("doc", 1, image_bytes)
print(page_result.mean_confidence, page_result.engine)
```

### Intermediate — mechanical OCR with rapidocr and escalation

```python
import asyncio
from rag_ocr import (
    OCRRouter,
    OCRRouterConfig,
    OCRPipeline,
    RapidOCREngine,
    VLMSemanticExtractor,
)

router = OCRRouter(
    engines=[RapidOCREngine()],
    semantic=VLMSemanticExtractor(
        base_url="http://localhost:1234/v1",
        model="glm-ocr",
        api_key_ref="RAG_VLM_API_KEY",  # env var, resolved per call
    ),
    config=OCRRouterConfig(confidence_threshold=0.7, min_lines=1, max_escalations_per_doc=3),
)
pipeline = OCRPipeline(router, preprocess_enabled=True, max_pages=200)

# result.pages[i].engine tells you which path each page took;
# result.escalated_pages lists the page numbers sent to the VLM.
result = await pipeline.process_images(pages)
```

### Advanced — inject an httpx transport for fully offline VLM tests

`VLMSemanticExtractor` accepts a custom `httpx.AsyncBaseTransport` so the VLM path is
testable end-to-end without a network:

```python
import json
import httpx
from rag_ocr import VLMSemanticExtractor, OCRRegion


class _FakeTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, req):
        payload = {
            "choices": [{"message": {"content": "| a | b |\n| - | - |\n| 1 | 2 |\n"}}],
        }
        body = json.dumps(payload).encode()
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})


vlm = VLMSemanticExtractor(transport=_FakeTransport(), api_key_ref=None)
regions = await vlm.extract(page_bytes)
assert regions[0].kind == "table"
```

### Advanced — write a custom engine

Implement the `OCREngine` protocol (or subclass `NullOCREngine`) and pass it to the
router. Recognition must run off the loop:

```python
import asyncio
from rag_ocr import OCREngine, RawLine


class MyEngine:
    name = "mine"

    async def recognize(self, image_bytes: bytes) -> list[RawLine]:
        return await asyncio.to_thread(self._sync, image_bytes)

    def _sync(self, image_bytes: bytes) -> list[RawLine]: ...
```

### Advanced — project into canonical `PageBlock`s with provenance

```python
from rag_ocr import OCRRouter, NullOCREngine, regions_to_blocks
from rag_core.ids import new_id

router = OCRRouter(engines=[NullOCREngine()], semantic=None)
page_result = await router.process_page("doc", 1, image_bytes)
page_id = new_id()
blocks = regions_to_blocks(page_result, page_id)
# Each block: page_id, kind, text, bbox, confidence, order — traceable to the page.
```

## Configuration

### `OCRRouterConfig`

| Field                      | Type   | Default | Notes                                              |
| -------------------------- | ------ | ------- | -------------------------------------------------- |
| `confidence_threshold`     | `float`| `0.7`   | Below this mean confidence → escalate.            |
| `min_lines`                | `int`  | `1`     | Fewer lines than this → escalate.                 |
| `escalate_model`           | `bool` | `True`  | Master switch for the VLM escalation path.        |
| `max_escalations_per_doc`  | `int`  | `3`     | Per-document budget; `process_page` uses a fresh 1. |

### `OCRPipeline`

| Field                  | Type     | Default | Notes                                     |
| ---------------------- | -------- | ------- | ----------------------------------------- |
| `preprocess_enabled`   | `bool`   | `True`  | Skip the preprocess chain when `False`.   |
| `max_pages`            | `int`    | `200`   | Pages beyond this are silently truncated. |
| `router`               | `OCRRouter` | —    | The configured router (engines + VLM).    |

### `VLMSemanticExtractor`

| Field            | Type                       | Default    | Notes                                  |
| ---------------- | -------------------------- | ---------- | -------------------------------------- |
| `base_url`       | `str`                      | `http://localhost:1234/v1` | OpenAI-compatible endpoint. |
| `model`          | `str`                      | `"glm-ocr"`| Model name sent in the chat payload.   |
| `api_key_ref`    | `str \| None`              | `None`     | Env var name, resolved per call.       |
| `timeout_s`      | `float`                    | `120.0`    | Mapped to `OCRError` on expiry.        |
| `prompt_template`| `str`                      | see source | Untrusted-data-safe default prompt.    |
| `transport`      | `httpx.AsyncBaseTransport \| None` | `None` | Inject for offline tests.   |

### Preprocessing

The `preprocess` chain is governed by constants, not config objects:

| Constant               | Value        | Notes                                  |
| ---------------------- | ------------ | -------------------------------------- |
| `_MAX_PIXELS`          | `40_000_000` (~40 MP) | Decompression-bomb guard.   |
| `normalize_resolution` | `target_dpi=300`, `max_width=2000` | Caps width before inference. |
| `deskew`               | `[-5, +5]°` in `0.5°` steps | Projection-profile variance sweep. |

## Testing

Tests are fully offline: no `rapidocr`, no network, no real VLM. The VLM extractor
accepts an injected `httpx` transport, and `NullOCREngine` exercises escalation
routing without any engine backend.

```bash
uv run pytest packages/rag-ocr -q
```

Unit tests cover detection of escalation triggers (zero lines, low confidence, high
confidence, budget exhaustion), the preprocess chain and its recorded operations,
`regions_to_blocks` provenance projection, and engine/protocol conformance
(`isinstance(router, OCRProcessor)`). No integration marker is required because the
package avoids all heavy backends by design — synthetic images are constructed
in-memory.

## Dependencies

`numpy>=1.26`, `pillow>=10.0`, `httpx>=0.27`, and `rag-core`.

Optional extras declared in `pyproject.toml`:

| Extra      | Packages                | Backend                          |
| ---------- | ----------------------- | -------------------------------- |
| `rapidocr` | `rapidocr-onnxruntime>=1.3` | `RapidOCREngine` (lazy import). |

## Cross-Package Relationships

- **`rag-core`** — `OCRRouter` implements the `rag_core.protocols.OCRProcessor`
  protocol; `errors.OCRError` (and the subclass `OCRErrorUnavailable`) are raised for
  missing engines and timed-out/malformed VLM calls; `documents.PageBlock` and
  `BoundingBox` are the output of `regions_to_blocks`; `ids.new_id` seeds document
  ids; `base.RagBaseModel` backs the config/result models.
- **`rag-doc-handler`** — owns PDF page rendering; `PDFParser` accepts an
  `OCRFallback` callable whose signature
  `(document_id, page_number, image_bytes) -> list[PageBlock]` is satisfied by
  `OCRPipeline.process_page`. The data-plane direction is doc-handler → ocr: doc-handler
  renders pages and hands images to the OCR callback; `rag-ocr` never imports
  doc-handler.
- **`rag-embedder`** — consumes the `PageBlock`s produced downstream of OCR (via the
  document's pages) for chunking; OCR results are inputs to embedding, not direct
  consumers of the embedder.
- **`rag-orchestrator`** — may wrap `OCRPipeline`/`OCRRouter` behind the FastAPI
  service boundary (ADR-0005) and expose `/health`, `/ready`, `/metrics`, and typed OCR
  endpoints per ADR-0007.
- **`rag-observe` / `rag-cache`** — may instrument engine/VLM spans and cache VLM
  results by an image content hash without modifying this package.

Service boundary: like all data-plane modules, `rag-ocr` runs in-process for latency;
the production scale path wraps `OCRPipeline` behind FastAPI endpoints (`/health`,
`/ready`, `/metrics`, and typed OCR endpoints) per ADR-0005 and ADR-0007.
