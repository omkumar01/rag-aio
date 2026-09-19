"""rag-ocr: hybrid mechanical + semantic OCR with provenance-preserving regions."""

from __future__ import annotations

from rag_ocr.engines import (
    NullOCREngine,
    OCREngine,
    OCREngineUnavailable,
    RapidOCREngine,
    RawLine,
    aggregate_confidence,
)
from rag_ocr.models import OCRPageResult, OCRRegion, OCRResult
from rag_ocr.pipeline import OCRPipeline, regions_to_blocks
from rag_ocr.preprocess import (
    PreprocessedPage,
    denoise,
    deskew,
    image_to_bytes,
    normalize_resolution,
    preprocess,
    to_grayscale,
)
from rag_ocr.routing import OCRRouter, OCRRouterConfig
from rag_ocr.semantic import VLMSemanticExtractor, classify_kind

__version__ = "0.1.0"

__all__ = [
    "NullOCREngine",
    "OCREngine",
    "OCREngineUnavailable",
    "OCRPageResult",
    "OCRPipeline",
    "OCRRegion",
    "OCRResult",
    "OCRRouter",
    "OCRRouterConfig",
    "PreprocessedPage",
    "RapidOCREngine",
    "RawLine",
    "VLMSemanticExtractor",
    "__version__",
    "aggregate_confidence",
    "classify_kind",
    "denoise",
    "deskew",
    "image_to_bytes",
    "normalize_resolution",
    "preprocess",
    "regions_to_blocks",
    "to_grayscale",
]
