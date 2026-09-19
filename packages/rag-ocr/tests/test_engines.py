"""Tests for OCR engines and confidence aggregation."""

from __future__ import annotations

import pytest
from rag_core.errors import OCRError
from rag_ocr.engines import (
    NullOCREngine,
    OCREngine,
    OCREngineUnavailable,
    RapidOCREngine,
    RawLine,
    aggregate_confidence,
)


def _rapidocr_importable() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def test_null_engine_satisfies_protocol() -> None:
    engine = NullOCREngine()
    assert engine.name == "null"
    assert isinstance(engine, OCREngine)


@pytest.mark.asyncio
async def test_null_engine_recognize_empty() -> None:
    engine = NullOCREngine()
    assert await engine.recognize(b"not-an-image") == []


def test_rapidocr_unavailable_without_extra() -> None:
    if _rapidocr_importable():
        pytest.skip("rapidocr extra is installed in this environment")
    with pytest.raises(OCREngineUnavailable) as exc_info:
        RapidOCREngine()
    assert issubclass(exc_info.type, OCRError)
    assert "rapidocr" in str(exc_info.value).lower()


def test_aggregate_confidence_mean_and_none() -> None:
    assert aggregate_confidence([]) is None
    lines = [
        RawLine(text="a", bbox=(0, 0, 1, 1), confidence=0.5),
        RawLine(text="b", bbox=(0, 0, 1, 1), confidence=0.9),
    ]
    assert aggregate_confidence(lines) == pytest.approx(0.7)


def test_raw_line_is_dataclass() -> None:
    line = RawLine(text="hi", bbox=(1, 2, 3, 4), confidence=0.8)
    assert line.text == "hi"
    assert line.bbox == (1, 2, 3, 4)
    assert line.confidence == 0.8
