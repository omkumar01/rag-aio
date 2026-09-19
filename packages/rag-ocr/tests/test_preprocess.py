"""Tests for preprocessing helpers."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw
from rag_core.errors import OCRError
from rag_ocr.preprocess import (
    PreprocessedPage,
    denoise,
    deskew,
    normalize_resolution,
    preprocess,
    to_grayscale,
)


def _barcode(width: int, height: int, n_lines: int = 12) -> Image.Image:
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    step = max(1, height // (n_lines + 1))
    for i in range(n_lines):
        y = step * (i + 1)
        draw.line((0, y, width, y), fill=0, width=3)
    return img


def test_to_grayscale_preserves_already_gray() -> None:
    gray = _barcode(100, 100)
    assert to_grayscale(gray).mode == "L"
    rgb = gray.convert("RGB")
    assert to_grayscale(rgb).mode == "L"


def test_normalize_resolution_caps_width() -> None:
    big = _barcode(5000, 500)
    out = normalize_resolution(big, max_width=2000)
    assert out.size[0] == 2000
    assert out.size[1] < 500


def test_normalize_resolution_no_op_small_image() -> None:
    img = _barcode(200, 100)
    out = normalize_resolution(img)
    assert out.size == (200, 100)


def test_preprocess_runs_chain_and_records_ops() -> None:
    img = _barcode(400, 400)
    result = preprocess(img)
    assert isinstance(result, PreprocessedPage)
    assert result.original_size == (400, 400)
    assert result.applied_ops == [
        "grayscale",
        "normalize_resolution",
        "denoise",
        "deskew(angle=0.00)",
    ]


def test_deskew_horizontal_is_near_zero() -> None:
    img = _barcode(400, 400)
    _, angle = deskew(img)
    assert abs(angle) < 0.5


def test_deskew_recovers_rotation() -> None:
    base = _barcode(400, 400)
    skewed = base.rotate(3.0, resample=Image.BICUBIC, expand=False)
    _, angle = deskew(skewed)
    # Correction should be -3 degrees (rotate back by -3).
    assert abs(angle + 3.0) < 1.0


def test_decompression_bomb_guard_raises() -> None:
    # A small image whose declared pixel count is forced beyond the guard by passing
    # a tiny max_pixels budget (avoids allocating a real giant image).
    img = _barcode(600, 600)
    with pytest.raises(OCRError):
        preprocess(img, max_pixels=100_000)


def test_denoise_returns_same_size() -> None:
    img = _barcode(200, 100)
    out = denoise(img)
    assert out.size == img.size


def test_preprocess_preserves_size_in_record() -> None:
    img = _barcode(321, 654)
    result = preprocess(img)
    assert result.original_size == (321, 654)
    assert result.image.size[0] <= 2000
