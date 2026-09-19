"""Parent/child chunking strategy.

Produces large *parent* chunks (for reranking / context) and small *child*
chunks (for retrieval).  Each child carries ``metadata.parent_chunk_id``
pointing to its parent.  Parents are returned first, then children.
"""

from __future__ import annotations

from rag_core.chunks import Chunk
from rag_core.documents import Document

from .base import BaseChunker, ChunkerConfig
from .recursive import RecursiveChunker

__all__ = ["ParentChildChunker"]


class ParentChildChunker(BaseChunker):
    """Split into parent and child chunks with cross-references.

    The parent config uses ``chunk_size * 4`` as the token budget; the child
    config inherits the caller's ``chunk_size``.  Both use the
    :class:`RecursiveChunker` splitting strategy internally.
    """

    chunker_name = "parent_child"

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        parent_size = max(self.config.chunk_size * 4, 1)
        child_size = max(self.config.chunk_size, 1)

        parent_cfg = ChunkerConfig(
            strategy="recursive",
            chunk_size=parent_size,
            overlap=self.config.overlap,
            min_chunk_size=self.config.min_chunk_size,
            respect_boundaries=self.config.respect_boundaries,
        )
        child_cfg = ChunkerConfig(
            strategy="recursive",
            chunk_size=child_size,
            overlap=self.config.overlap,
            min_chunk_size=self.config.min_chunk_size,
            respect_boundaries=self.config.respect_boundaries,
        )

        parent_rc = RecursiveChunker(self.tokenizer, parent_cfg)
        child_rc = RecursiveChunker(self.tokenizer, child_cfg)

        parent_spans_raw = parent_rc._split_spans(text, 0, list(parent_rc.separators))
        parent_spans = parent_rc._merge_spans(parent_spans_raw, text)
        child_spans_raw = child_rc._split_spans(text, 0, list(child_rc.separators))
        child_spans = child_rc._merge_spans(child_spans_raw, text)

        child_parent_spans = [
            self._find_parent_span(parent_spans, cs, ce) for cs, ce in child_spans
        ]
        return self.make_chunks(document, child_spans, parent_spans=child_parent_spans)

    @staticmethod
    def _find_parent_span(
        parent_spans: list[tuple[int, int]], child_start: int, child_end: int
    ) -> tuple[int, int]:
        for ps, pe in parent_spans:
            if ps <= child_start and pe >= child_end:
                return (ps, pe)
        if parent_spans:
            ps, pe = parent_spans[0]
            return (ps, pe)
        return (child_start, child_end)
