"""Token-aware sliding-window chunker.

Encodes the document, then slides a window of exactly ``chunk_size`` tokens
with ``overlap`` tokens of stride.  Each window is decoded back to text.
"""

from __future__ import annotations

from rag_core.chunks import Chunk
from rag_core.documents import Document

from .base import BaseChunker, build_page_map

__all__ = ["TokenChunker"]


class TokenChunker(BaseChunker):
    """Split on exact token boundaries with a sliding window."""

    chunker_name = "token"

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        tokens = self.tokenizer.encode(text)
        cs = max(self.config.chunk_size, 1)
        ov = min(self.config.overlap, cs - 1) if self.config.overlap > 0 else 0
        step = cs - ov
        if step <= 0:
            step = 1
        page_map = build_page_map(document)
        chunks: list[Chunk] = []
        search_pos = 0
        i = 0
        index = 0
        while i < len(tokens):
            end = min(i + cs, len(tokens))
            decoded = self.tokenizer.decode(tokens[i:end])
            idx = text.find(decoded, search_pos)
            if idx == -1:
                idx = text.find(decoded)
            char_start = idx if idx != -1 else None
            char_end = (idx + len(decoded)) if idx != -1 else None
            if char_start is not None:
                search_pos = char_start + len(decoded)
            chunks.append(
                self._build_chunk_text(
                    document,
                    decoded,
                    index=index,
                    page_map=page_map,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            index += 1
            if end >= len(tokens):
                break
            i += step
        return chunks
