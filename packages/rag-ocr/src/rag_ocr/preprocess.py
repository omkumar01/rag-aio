"""Image preprocessing helpers (Pillow + numpy).

Functions here are intentionally synchronous; callers must schedule them off the
event loop via :func:`asyncio.to_thread` so CPU-bound work never blocks the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
from rag_core.errors import OCRError

# 40 megapixel decompression-bomb guard (roughly a 6600x6600 image).
_MAX_PIXELS: int = 40_000_000


@dataclass(slots=True)
class PreprocessedPage:
    """Outcome of the preprocessing pipeline for one page image."""

    original_size: tuple[int, int]
    image: Image.Image
    applied_ops: list[str] = field(default_factory=list)


def to_grayscale(img: Image.Image) -> Image.Image:
    """Return an ``L``-mode copy of ``img``."""
    if img.mode == "L":
        return img
    return img.convert("L")


def normalize_resolution(
    img: Image.Image, target_dpi: float = 300.0, max_width: int = 2000
) -> Image.Image:
    """Normalise pixel density toward ``target_dpi`` and cap width at ``max_width``.

    Images carrying DPI metadata are rescaled to ``target_dpi``; images without DPI
    metadata keep their native resolution. Either way the resulting width is never
    allowed to exceed ``max_width`` (preventing pathologically large inputs reaching
    inference).
    """
    width, height = img.size
    new_w, new_h = width, height

    info = img.info.get("dpi")
    if info:
        try:
            src_dpi = float(info[0])
        except (IndexError, TypeError, ValueError):
            src_dpi = 0.0
        if src_dpi > 0:
            scale = target_dpi / src_dpi
            new_w = max(1, round(width * scale))
            new_h = max(1, round(height * scale))

    if new_w > max_width:
        scale = max_width / new_w
        new_w = max_width
        new_h = max(1, round(new_h * scale))

    if (new_w, new_h) != (width, height):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def deskew(img: Image.Image) -> tuple[Image.Image, float]:
    """Deskew a text page image via a projection-profile variance sweep.

    Returns ``(corrected_image, angle_degrees)`` where ``angle_degrees`` is the
    correction applied (positive => counter-clockwise). The sweep is limited to
    ``[-5, +5]`` degrees in ``0.5`` degree steps, which is sufficient for typical
    scanned-page skew while keeping the computation deterministic and cheap.
    """
    gray = to_grayscale(img)
    arr = np.asarray(gray, dtype=np.float64)
    if arr.size == 0:
        return img, 0.0

    # Binarise: dark text becomes 1, light background 0.
    threshold = float(arr.mean())
    binary = (arr < threshold).astype(np.uint8) * 255
    bin_img = Image.fromarray(binary, mode="L")

    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-5.0, 5.0001, 0.5):
        rotated = bin_img.rotate(float(angle), resample=Image.Resampling.NEAREST)
        # Row projection: high variance when horizontal text lines are aligned.
        projection = np.asarray(rotated, dtype=np.float64).sum(axis=1)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    corrected = img.rotate(best_angle, resample=Image.Resampling.BILINEAR)
    return corrected, best_angle


def denoise(img: Image.Image, size: int = 3) -> Image.Image:
    """Apply a median filter (salt/pepper noise removal)."""
    return img.filter(ImageFilter.MedianFilter(size=size))


def _check_decompression_bomb(img: Image.Image, max_pixels: int = _MAX_PIXELS) -> None:
    """Raise :class:`OCRError` if the image exceeds the pixel-count guard."""
    width, height = img.size
    if width * height > max_pixels:
        raise OCRError(
            f"image too large: {width}x{height}={width * height} pixels "
            f"(limit {max_pixels}); refusing OCR of a likely decompression bomb",
            code="image_too_large",
        )


def preprocess(img: Image.Image, max_pixels: int = _MAX_PIXELS) -> PreprocessedPage:
    """Run the standard preprocessing chain on ``img``.

    Chain: bomb guard -> grayscale -> normalise resolution -> denoise -> deskew.
    The returned :class:`PreprocessedPage` records the original size, the processed
    image, and the ordered list of applied operations (useful for provenance/debug).
    """
    _check_decompression_bomb(img, max_pixels)
    original_size = img.size
    applied: list[str] = []

    img = to_grayscale(img)
    applied.append("grayscale")

    img = normalize_resolution(img)
    applied.append("normalize_resolution")

    img = denoise(img)
    applied.append("denoise")

    img, angle = deskew(img)
    applied.append(f"deskew(angle={angle:.2f})")

    return PreprocessedPage(original_size=original_size, image=img, applied_ops=applied)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Serialise a PIL image to the given format (default PNG)."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()
