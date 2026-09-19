"""RapidOCR engine (optional, behind the ``rapidocr`` extra).

RapidOCR is imported lazily inside ``__init__`` so that merely *importing*
:mod:`rag_ocr` never requires the (heavy, GPU-capable) backend to be installed.
Constructing :class:`RapidOCREngine` without the extra raises
:class:`~rag_ocr.engines.OCREngineUnavailable` with an install hint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_core.errors import OCRError

from rag_ocr.engines.base import OCREngineUnavailable, RawLine

_INSTALL_HINT = (
    "RapidOCREngine requires the 'rapidocr' extra. Install it with 'pip install rag-ocr[rapidocr]'."
)


def _line_from_result(item: Any) -> RawLine:
    """Map a RapidOCR ``[box, text, score]`` item to a :class:`RawLine`."""
    box, text, score = item
    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        x, y = point
        xs.append(float(x))
        ys.append(float(y))
    return RawLine(
        text=str(text),
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        confidence=float(score),
    )


class RapidOCREngine:
    """OCR engine backed by ``rapidocr-onnxruntime``.

    Recognition runs in a worker thread because the ONNX runtime call is synchronous
    and may use the GPU/CPU for extended periods.
    """

    def __init__(self) -> None:
        self.name = "rapidocr"
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OCREngineUnavailable(_INSTALL_HINT) from exc
        self._client: Any = RapidOCR()

    async def recognize(self, image_bytes: bytes) -> list[RawLine]:
        def _run() -> list[RawLine]:
            result = self._client.ocr(image_bytes)
            lines: list[RawLine] = []
            # Result is a list of pages; each page is a list of [box, text, score].
            pages = result if isinstance(result, list) else [result]
            for page in pages:
                if not isinstance(page, list):
                    continue
                for item in page:
                    if not (isinstance(item, (list, tuple)) and len(item) >= 3):
                        continue
                    try:
                        lines.append(_line_from_result(item))
                    except (ValueError, TypeError) as exc:
                        raise OCRError(
                            "malformed rapidocr line",
                            code="ocr_engine",
                            details={"item": str(item)},
                        ) from exc
            return lines

        return await asyncio.to_thread(_run)
