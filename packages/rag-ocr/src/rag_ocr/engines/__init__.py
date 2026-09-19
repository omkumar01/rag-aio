"""OCR engine registry and primitives."""

from __future__ import annotations

from rag_ocr.engines.base import OCREngine, OCREngineUnavailable, RawLine
from rag_ocr.engines.confidence import aggregate_confidence
from rag_ocr.engines.null import NullOCREngine
from rag_ocr.engines.rapidocr import RapidOCREngine

__all__ = [
    "NullOCREngine",
    "OCREngine",
    "OCREngineUnavailable",
    "RapidOCREngine",
    "RawLine",
    "aggregate_confidence",
]
