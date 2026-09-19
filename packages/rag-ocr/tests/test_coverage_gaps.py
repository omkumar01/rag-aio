"""Targeted coverage for engine mapping, preprocessing branches, and VLM error paths."""

from __future__ import annotations

import io
import sys
import types

import httpx
import numpy as np
import pytest
from PIL import Image
from rag_core.errors import OCRError
from rag_ocr.engines.rapidocr import RapidOCREngine, _line_from_result
from rag_ocr.preprocess import deskew, normalize_resolution
from rag_ocr.semantic import VLMSemanticExtractor

# --- rapidocr mapping ---------------------------------------------------------


def test_line_from_result_maps_box_to_bbox() -> None:
    line = _line_from_result([[[10, 5], [30, 5], [30, 25], [10, 25]], "hello", 0.9])
    assert line.text == "hello"
    assert line.bbox == (10.0, 5.0, 30.0, 25.0)
    assert line.confidence == pytest.approx(0.9)


def _install_fake_rapidocr(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    fake_module = types.ModuleType("rapidocr_onnxruntime")

    class FakeRapidOCR:
        def __init__(self) -> None:
            self.ocr_calls: list[bytes] = []

        def ocr(self, data: bytes) -> object:
            self.ocr_calls.append(data)
            return result

    fake_module.RapidOCR = FakeRapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)


async def test_rapidocr_engine_recognize_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], "alpha", 0.9],
            [[[5, 5], [20, 5], [20, 15], [5, 15]], "beta", 0.5],
        ]
    ]
    _install_fake_rapidocr(monkeypatch, result)
    engine = RapidOCREngine()
    lines = await engine.recognize(b"png-bytes")
    assert [ln.text for ln in lines] == ["alpha", "beta"]
    assert lines[0].bbox == (0.0, 0.0, 10.0, 10.0)
    assert lines[0].confidence == pytest.approx(0.9)


async def test_rapidocr_engine_handles_nested_pages_and_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RapidOCR may return a tuple of pages; garbage pages/items are skipped.
    result = (
        [[[[0, 0], [1, 0], [1, 1], [0, 1]], "one", 0.8]],
        "not-a-page",
        [["bad"]],
    )
    _install_fake_rapidocr(monkeypatch, result)
    engine = RapidOCREngine()
    lines = await engine.recognize(b"img")
    assert [ln.text for ln in lines] == ["one"]


async def test_rapidocr_engine_malformed_line_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = [[[[["not", "numbers"]], "text", 0.5]]]
    _install_fake_rapidocr(monkeypatch, result)
    engine = RapidOCREngine()
    with pytest.raises(OCRError, match="malformed rapidocr"):
        await engine.recognize(b"img")


# --- preprocessing branches ----------------------------------------------------


def _png(dpi: tuple[int, int] | None = None, size: tuple[int, int] = (100, 40)) -> bytes:
    img = Image.new("L", size, color=255)
    if dpi is not None:
        img.info["dpi"] = dpi
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=dpi) if dpi else img.save(buf, format="PNG")
    return buf.getvalue()


def test_normalize_resolution_scales_by_dpi() -> None:
    img = Image.new("L", (100, 50), color=255)
    img.info["dpi"] = (150, 150)
    out = normalize_resolution(img, target_dpi=300, max_width=10_000)
    assert out.size == (200, 100)


def test_normalize_resolution_broken_dpi_info_is_ignored() -> None:
    img = Image.new("L", (100, 50), color=255)
    img.info["dpi"] = ("not", "numbers")  # type: ignore[assignment]
    out = normalize_resolution(img, target_dpi=300, max_width=10_000)
    assert out.size == (100, 50)


def test_normalize_resolution_caps_width() -> None:
    img = Image.new("L", (400, 100), color=255)
    img.info["dpi"] = (600, 600)
    out = normalize_resolution(img, target_dpi=300, max_width=100)
    assert out.size[0] <= 100


def test_deskew_empty_image_returns_flat() -> None:
    img = Image.new("L", (0, 0))
    out, angle = deskew(img)
    assert angle == 0.0
    assert out.size == (0, 0)


def test_deskew_level_image_near_zero_angle() -> None:
    arr = np.full((40, 200), 255, dtype=np.uint8)
    arr[10:14, 20:180] = 0  # a horizontal bar
    img = Image.fromarray(arr)
    _, angle = deskew(img)
    assert abs(angle) <= 1.0


# --- VLM semantic error paths ---------------------------------------------------


def test_media_type_detection_branches() -> None:
    from rag_ocr.semantic import _media_type

    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"rest"
    webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBPVP8 "
    assert _media_type(jpeg) == "image/jpeg"
    assert _media_type(webp) == "image/webp"
    assert _media_type(b"unknown") == "image/png"


async def test_vlm_malformed_response_raises_ocr_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    with pytest.raises(OCRError, match="malformed VLM response"):
        await extractor.extract(b"img")


async def test_vlm_http_error_maps_to_ocr_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    with pytest.raises(OCRError, match="VLM request failed"):
        await extractor.extract(b"img")


async def test_vlm_non_string_content_is_stringified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": {"odd": 1}}}]})

    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    regions = await extractor.extract(b"img")
    assert len(regions) == 1
    assert "odd" in regions[0].text
