"""Recursive separator-hierarchy chunker (default strategy).

Mimics the LangChain ``RecursiveCharacterTextSplitter``: a list of separators
is tried in order.  If a piece still exceeds the token budget after splitting
on the current separator, the next, finer separator is used.  Finally the
small pieces are merged into token-budgeted chunks with a token-based overlap.
"""

from __future__ import annotations

from rag_core.chunks import Chunk
from rag_core.documents import Document
from rag_core.protocols import Tokenizer

from .base import BaseChunker, ChunkerConfig

__all__ = ["RecursiveChunker"]

# Ordered from coarsest to finest.  The final "" separator splits by character.
DEFAULT_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Default chunker: recursive separator splitting with token-aware merge."""

    chunker_name = "recursive"

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ChunkerConfig | None = None,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(tokenizer, config)
        self.separators: list[str] = list(separators) if separators else list(DEFAULT_SEPARATORS)

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        spans = self._split_spans(text, 0, list(self.separators))
        spans = self._merge_spans(spans, text)
        return self.make_chunks(document, spans)

    # -- recursive split ---------------------------------------------------

    def _split_spans(self, text: str, offset: int, seps: list[str]) -> list[tuple[int, int]]:
        if not seps:
            return [(offset, offset + len(text))]
        sep = seps[0]
        if sep == "":
            return [(offset, offset + len(text))]
        pieces = text.split(sep)
        spans: list[tuple[int, int]] = []
        pos = offset
        for piece in pieces:
            if not piece:
                pos += len(sep)
                continue
            tokens = self.tokenizer.count_tokens(piece)
            if tokens <= self.config.chunk_size or len(seps) == 1:
                spans.append((pos, pos + len(piece)))
            else:
                spans.extend(self._split_spans(piece, pos, seps[1:]))
            pos += len(piece) + len(sep)
        return spans

    # -- merge with overlap ------------------------------------------------

    def _merge_spans(self, spans: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
        if not spans:
            return []
        cs = self.config.chunk_size
        ov = self.config.overlap
        result: list[tuple[int, int]] = []
        s, e = spans[0]
        for ns, ne in spans[1:]:
            candidate = text[s:ne]
            if self.tokenizer.count_tokens(candidate) <= cs:
                e = ne
            else:
                result.append((s, e))
                if ov > 0:
                    overlap_chars = self._overlap_chars(text[s:e], ov)
                    ns = max(ns - overlap_chars, s)
                s, e = ns, ne
        result.append((s, e))
        return result
