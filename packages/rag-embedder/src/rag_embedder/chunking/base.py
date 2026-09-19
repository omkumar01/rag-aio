"""Shared chunking infrastructure.

Defines :class:`ChunkerConfig`, the :class:`BaseChunker` ABC, and the
:func:`build_page_map` helper that maps character offsets to page numbers.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, TypeVar

from rag_core.base import RagBaseModel
from rag_core.chunks import Chunk, ChunkMetadata
from rag_core.documents import Document
from rag_core.protocols import Tokenizer

__all__ = [
    "CHUNKER_VERSION",
    "BaseChunker",
    "ChunkerConfig",
    "build_page_map",
]

log = logging.getLogger(__name__)

CHUNKER_VERSION: str = "0.1.0"

_CHARS_PER_TOKEN = 4


class ChunkerConfig(RagBaseModel):
    """Configuration shared by all chunkers."""

    strategy: Literal["fixed", "sentence", "token", "recursive", "structural", "parent_child"] = (
        "recursive"
    )
    chunk_size: int = 512
    overlap: int = 64
    min_chunk_size: int = 32
    respect_boundaries: bool = True


def build_page_map(document: Document) -> list[tuple[int, int, int]]:
    """Build ``(start_char, end_char, page_number)`` ranges from document pages.

    Locates each page's text within ``document.text`` and records the character
    span.  Returns an empty list for textless documents or documents without
    pages.
    """
    if not document.pages or not document.text:
        return []
    page_map: list[tuple[int, int, int]] = []
    search_pos = 0
    full_text = document.text
    for page in document.pages:
        if not page.text:
            continue
        idx = full_text.find(page.text, search_pos)
        if idx == -1:
            idx = full_text.find(page.text)
            if idx == -1:
                continue
        end = idx + len(page.text)
        page_map.append((idx, end, page.page_number))
        search_pos = end
    return page_map


_T = TypeVar("_T")


class BaseChunker(ABC):
    """ABC for all chunkers.

    Holds a :class:`~rag_core.protocols.Tokenizer` and a
    :class:`ChunkerConfig`.  Sub-classes implement :meth:`chunk`.
    """

    chunker_name: str = "base"

    def __init__(self, tokenizer: Tokenizer, config: ChunkerConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config: ChunkerConfig = config or ChunkerConfig()

    @property
    def chunker_version(self) -> str:
        return CHUNKER_VERSION

    # -- metadata helpers --------------------------------------------------

    @staticmethod
    def _pages_for_span(page_map: list[tuple[int, int, int]], start: int, end: int) -> list[int]:
        """Return sorted unique page numbers overlapping ``[start, end)``."""
        if not page_map:
            return []
        pages: set[int] = set()
        for p_start, p_end, page_num in page_map:
            if p_start < end and p_end > start:
                pages.add(page_num)
        return sorted(pages)

    # -- span helpers ------------------------------------------------------

    @staticmethod
    def _merge_small_spans(
        spans: list[tuple[int, int]], text: str, min_size: int
    ) -> list[tuple[int, int]]:
        """Merge any span whose text length is below *min_size* into the previous span.

        The first span is never dropped even if tiny.
        """
        if min_size <= 0 or len(spans) <= 1:
            return list(spans)
        merged: list[tuple[int, int]] = []
        for span in spans:
            s, e = span
            if merged and (e - s) < min_size:
                ps, _ = merged[-1]
                merged[-1] = (ps, e)
            else:
                merged.append(span)
        return merged

    def _overlap_chars(self, text: str, overlap_tokens: int) -> int:
        """Estimate character count for *overlap_tokens* at the end of *text*."""
        if overlap_tokens <= 0 or not text:
            return 0
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return len(text)
        tail = self.tokenizer.decode(tokens[-overlap_tokens:])
        idx = text.rfind(tail)
        if idx == -1:
            return min(len(text), overlap_tokens * _CHARS_PER_TOKEN)
        return len(text) - idx

    # -- chunk construction ------------------------------------------------

    def make_chunks(
        self,
        document: Document,
        spans: list[tuple[int, int]],
        *,
        parent_spans: list[tuple[int, int]] | None = None,
    ) -> list[Chunk]:
        """Build :class:`Chunk` objects from character spans.

        Parameters
        ----------
        document:
            The source document.
        spans:
            List of ``(start_char, end_char)`` character offsets.
        parent_spans:
            Optional list parallel to *spans* giving the parent span for each
            chunk.  When provided the unique parent spans are materialised as
            parent chunks first; each child then receives
            ``metadata.parent_chunk_id`` pointing at its parent.  Parents are
            returned before children.
        """
        # Apply min_chunk_size filtering (non-parent path only).
        if parent_spans is None:
            spans = self._merge_small_spans(spans, document.text, self.config.min_chunk_size)

        page_map = build_page_map(document)
        chunks: list[Chunk] = []

        if parent_spans is not None:
            unique_parents: dict[tuple[int, int], Chunk] = {}
            parent_keys: list[tuple[int, int]] = []
            for p_start, p_end in parent_spans:
                key = (p_start, p_end)
                if key not in unique_parents:
                    unique_parents[key] = self._build_chunk(
                        document, key, index=len(unique_parents), page_map=page_map
                    )
                parent_keys.append(key)
            for parent in unique_parents.values():
                chunks.append(parent)
            for idx, (start, end) in enumerate(spans):
                p_span = parent_spans[idx]
                parent = unique_parents[(p_span[0], p_span[1])]
                chunks.append(
                    self._build_chunk(
                        document,
                        (start, end),
                        index=idx,
                        page_map=page_map,
                        parent_chunk_id=parent.id,
                    )
                )
        else:
            for idx, (start, end) in enumerate(spans):
                chunks.append(
                    self._build_chunk(document, (start, end), index=idx, page_map=page_map)
                )
        return chunks

    def _build_chunk(
        self,
        document: Document,
        span: tuple[int, int],
        *,
        index: int,
        page_map: list[tuple[int, int, int]],
        parent_chunk_id: str | None = None,
        section_path: list[str] | None = None,
    ) -> Chunk:
        start, end = span
        text = document.text[start:end]
        token_count = self.tokenizer.count_tokens(text)
        page_numbers = self._pages_for_span(page_map, start, end)
        metadata = ChunkMetadata(
            document_id=document.id,
            document_hash=document.content_hash,
            chunker=self.chunker_name,
            chunker_version=self.chunker_version,
            page_numbers=page_numbers,
            char_start=start,
            char_end=end,
            token_count=token_count,
            parent_chunk_id=parent_chunk_id,
            section_path=section_path or [],
        )
        return Chunk(
            document_id=document.id,
            text=text,
            index=index,
            metadata=metadata,
            token_count=token_count,
        )

    def _build_chunk_text(
        self,
        document: Document,
        text: str,
        *,
        index: int,
        page_map: list[tuple[int, int, int]] | None = None,
        parent_chunk_id: str | None = None,
        section_path: list[str] | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Chunk:
        """Build a Chunk from *text* directly (used by token-aware chunkers)."""
        token_count = self.tokenizer.count_tokens(text)
        pm = page_map or build_page_map(document)
        page_numbers: list[int] = []
        if char_start is not None and char_end is not None:
            page_numbers = self._pages_for_span(pm, char_start, char_end)
        metadata = ChunkMetadata(
            document_id=document.id,
            document_hash=document.content_hash,
            chunker=self.chunker_name,
            chunker_version=self.chunker_version,
            page_numbers=page_numbers,
            char_start=char_start,
            char_end=char_end,
            token_count=token_count,
            parent_chunk_id=parent_chunk_id,
            section_path=section_path or [],
        )
        return Chunk(
            document_id=document.id,
            text=text,
            index=index,
            metadata=metadata,
            token_count=token_count,
        )

    # -- abstract interface ------------------------------------------------

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into chunks."""
        raise NotImplementedError

    # -- thread helpers ----------------------------------------------------

    async def _to_thread(self, fn: Callable[..., _T], *args: object) -> _T:
        """Offload a synchronous CPU-bound callable to a worker thread."""
        return await asyncio.to_thread(fn, *args)
