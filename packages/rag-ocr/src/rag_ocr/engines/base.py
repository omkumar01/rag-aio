"""OCR engine contracts and shared primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_core.errors import OCRError


@dataclass(slots=True)
class RawLine:
    """One raw recognition line emitted by an OCR engine.

    ``bbox`` is the axis-aligned ``(x0, y0, x1, y1)`` box of the line in page
    coordinates; engines that cannot produce a box set it to ``(0, 0, 0, 0)``.
    """

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


class OCREngineUnavailable(OCRError):  # type: ignore[valid-type,misc]
    """Raised when an OCR engine backend is not installed or cannot initialise."""

    default_code = "engine_unavailable"


@runtime_checkable
class OCREngine(Protocol):
    """Pluggable OCR engine contract.

    Implementations wrap any synchronous CPU/GPU recognition work inside
    :meth:`recognize` (which must offload to a thread) so the event loop is never
    blocked.
    """

    name: str

    async def recognize(self, image_bytes: bytes) -> list[RawLine]:
        """Recognize text lines in ``image_bytes``; empty list if nothing found."""
        ...


__all__ = ["OCREngine", "OCREngineUnavailable", "RawLine"]
