"""Structural (heading-aware) chunker for Markdown and HTML.

Splits on ``#``/``##``/``###`` Markdown lines and ``<h1>``-``<h6>`` HTML
markers, maintaining a hierarchical ``section_path`` in each chunk's metadata.
Sections that exceed the token budget are further split recursively.
"""

from __future__ import annotations

import re

from rag_core.chunks import Chunk
from rag_core.documents import Document

from .base import BaseChunker, build_page_map

__all__ = ["StructuralChunker"]

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)
_HTML_HEADING = re.compile(r"<h([1-6])(?:\s[^>]*)?>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
_SPLIT_FALLBACK = ["\n\n", "\n", ". ", " ", ""]


class StructuralChunker(BaseChunker):
    """Split documents along Markdown/HTML heading boundaries."""

    chunker_name = "structural"

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        page_map = build_page_map(document)
        sections = self._sections(text)
        chunks: list[Chunk] = []
        for section_path, start, end in sections:
            if start >= end:
                continue
            spans = self._split_within(start, end, text)
            spans = self._merge_spans_in_section(spans, text)
            for _idx, (s, e) in enumerate(spans):
                chunks.append(
                    self._build_chunk(
                        document,
                        (s, e),
                        index=len(chunks),
                        page_map=page_map,
                        section_path=list(section_path),
                    )
                )
        return chunks

    # -- heading detection -------------------------------------------------

    def _sections(self, text: str) -> list[tuple[list[str], int, int]]:
        """Return ``[(section_path, start, end), ...]`` covering the whole text."""
        raw: list[tuple[int, str, int, int]] = []
        for m in _MD_HEADING.finditer(text):
            level = len(m.group(1))
            raw.append((level, m.group(2).strip(), m.start(), m.end()))
        for m in _HTML_HEADING.finditer(text):
            level = int(m.group(1))
            raw.append((level, _strip_html(m.group(2)), m.start(), m.end()))
        raw.sort(key=lambda h: h[2])

        if not raw:
            return [([], 0, len(text))]

        sections: list[tuple[list[str], int, int]] = []
        stack: list[tuple[int, str]] = []
        stack_top: list[str] = []
        pos = 0
        for level, title, h_start, h_end in raw:
            if h_start > pos:
                sections.append((list(stack_top), pos, h_start))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            stack_top = [s[1] for s in stack]
            pos = h_end
        if pos < len(text):
            sections.append((list(stack_top), pos, len(text)))
        return sections

    # -- overflow splitting ------------------------------------------------

    def _split_within(self, start: int, end: int, text: str) -> list[tuple[int, int]]:
        """Split a section range into token-budgeted spans."""
        seps = self._resolve_separators()
        return self._rec_split(text, start, end, seps)

    @staticmethod
    def _resolve_separators() -> list[str]:
        seps = list(_SPLIT_FALLBACK)
        if "\n\n\n" not in seps:
            seps.insert(0, "\n\n\n")
        if "\n\n" not in seps:
            seps.insert(0, "\n\n")
        return seps

    def _rec_split(
        self, text: str, offset: int, limit: int, seps: list[str]
    ) -> list[tuple[int, int]]:
        if not seps:
            return [(offset, limit)]
        sep = seps[0]
        if sep == "":
            return [(offset, limit)]
        sub = text[offset:limit]
        pieces = sub.split(sep)
        spans: list[tuple[int, int]] = []
        pos = offset
        for piece in pieces:
            if not piece:
                pos += len(sep)
                continue
            piece_end = pos + len(piece)
            tokens = self.tokenizer.count_tokens(piece)
            if tokens <= self.config.chunk_size or len(seps) == 1:
                if not spans or piece_end - pos >= self.config.min_chunk_size:
                    spans.append((pos, piece_end))
                else:
                    ps, _pe = spans[-1]
                    spans[-1] = (ps, piece_end)
            else:
                spans.extend(self._rec_split(text, pos, piece_end, seps[1:]))
            pos += len(piece) + len(sep)
        return spans

    def _merge_spans_in_section(
        self, spans: list[tuple[int, int]], text: str
    ) -> list[tuple[int, int]]:
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


def _strip_html(text: str) -> str:
    """Remove inline HTML tags from extracted heading text."""
    return re.sub(r"<[^>]+>", "", text).strip()
