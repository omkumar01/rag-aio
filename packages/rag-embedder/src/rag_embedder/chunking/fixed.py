"""Fixed-size character-window chunker with overlap."""

from __future__ import annotations

from rag_core.chunks import Chunk
from rag_core.documents import Document

from .base import BaseChunker

__all__ = ["FixedChunker"]


class FixedChunker(BaseChunker):
    """Split text into fixed-size character windows with a configurable overlap."""

    chunker_name = "fixed"

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        cs = max(self.config.chunk_size, 1)
        ov = min(self.config.overlap, cs // 2)
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + cs, len(text))
            spans.append((start, end))
            if end >= len(text):
                break
            start = end - ov if ov > 0 else end
        return self.make_chunks(document, spans)
