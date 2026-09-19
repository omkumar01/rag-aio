# rag-ocr

Hybrid mechanical + semantic OCR for rag-aio, with provenance-preserving regions.

## Design

Two layers, cheap-first:

1. **Mechanical** — pluggable OCR engines recognize text with bounding boxes and
   confidence. Image preprocessing (grayscale, resolution normalization, denoise,
   projection-profile deskew) runs before recognition; decompression-bomb guards cap
   pixel counts. All inference runs off the event loop via `asyncio.to_thread`.
2. **Semantic escalation** — pages whose best engine result is below
   `OCRRouterConfig.confidence_threshold` (or empty) are escalated to a VLM
   (`VLMSemanticExtractor`, e.g. `glm-ocr` on an OpenAI-compatible endpoint such as
   LM Studio), up to `max_escalations_per_doc`. VLM output is treated as untrusted
   data and classified into a block kind (table/heading/text).

Every region carries `page_number` + bounding box, so each semantic element traces back
to its source. `regions_to_blocks` converts OCR results into rag-core `PageBlock`s with
provenance preserved.

## Public API

Models (`models.py`): `OCRRegion`, `OCRPageResult` (with `mean_confidence`),
`OCRResult` (with `escalated_pages`).

Engines (`engines/`): `OCREngine` protocol + `RawLine`; `NullOCREngine` (no-op),
`RapidOCREngine` (lazy import; raises `OCREngineUnavailable` with an install hint when
the `rapidocr` extra is missing), `aggregate_confidence`.

Routing (`routing.py`): `OCRRouterConfig` (`confidence_threshold=0.7`, `min_lines`,
`max_escalations_per_doc=3`) and `OCRRouter` with `process_page(document_id, page_number,
image_bytes) -> OCRPageResult` (fresh escalation budget per call) and
`process_document(document_id, pages) -> OCRResult` (shared escalation budget, tracked in
`escalated_pages`). Satisfies the rag-core `OCRProcessor` protocol.

Semantic (`semantic.py`): `VLMSemanticExtractor(base_url="http://localhost:1234/v1",
model="glm-ocr", api_key_ref=None, ...)` — `api_key_ref` names an env var, resolved at
call time, never stored or logged; timeouts map to `OCRError`.

Pipeline (`pipeline.py`): `OCRPipeline.process_images([(page_number, bytes), ...]) ->
OCRResult` with preprocessing and a `max_pages` guard; `preprocess`/`deskew`/`denoise`/
`normalize_resolution`/`to_grayscale` helpers; `regions_to_blocks(page_result, page_id)`.

## Notes

- PDF page rendering intentionally lives in `rag-doc-handler`; this package accepts
  pre-rendered page images.
- Engine routing tries engines in order until one produces lines; difficult pages
  escalate to the semantic layer only when configured (never silently).
- Tests run offline (no rapidocr, no network); the VLM extractor accepts an injected
  `httpx` transport.
