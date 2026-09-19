"""Null / no-op OCR engine (testing + safe fallback)."""

from __future__ import annotations

from rag_ocr.engines.base import RawLine


class NullOCREngine:
    """An OCR engine that performs no recognition and always returns no lines.

    Useful as a safe default and in tests where mechanical OCR behaviour is
    irrelevant but escalation routing must still be exercised.
    """

    def __init__(self) -> None:
        self.name = "null"

    async def recognize(self, image_bytes: bytes) -> list[RawLine]:
        return []
