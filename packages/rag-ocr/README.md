# rag-ocr

Hybrid OCR for rag-aio: a mechanical pipeline (preprocessing, deskew, layout detection,
text recognition with bounding boxes and confidence, tables/figures/reading order) with
deterministic-first semantic extraction, escalating difficult pages to a VLM (e.g.
glm-ocr via an OpenAI-compatible endpoint). Pluggable engines (RapidOCR behind an extra),
confidence-threshold routing, multilingual support, and full provenance from every
semantic element back to page and bounding box. CPU/GPU inference runs off the event loop.
