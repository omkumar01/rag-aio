"""Sentence-boundary chunker.

Splits on sentence-ending punctuation (``.`` ``!`` ``?``) and paragraph
breaks, then packs sentences into token-budgeted chunks with a configurable
sentence overlap.
"""

from __future__ import annotations

import re

from rag_core.chunks import Chunk
from rag_core.documents import Document

from .base import BaseChunker

__all__ = ["SentenceChunker"]

# Splits on whitespace that follows a sentence-ending punctuation mark.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Paragraph boundary (blank line).
_PARAGRAPH_END = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return ``(sentence, start, end)`` tuples covering *text* exactly."""
    sentences: list[tuple[str, int, int]] = []
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if pos < end:
            sentences.append((text[pos:end], pos, end))
        pos = match.end()
    # Paragraph breaks that aren't preceded by sentence-end punctuation.
    if pos < len(text):
        sentences.append((text[pos:], pos, len(text)))
    return sentences


class SentenceChunker(BaseChunker):
    """Pack sentences into chunks whose token count stays within budget."""

    chunker_name = "sentence"

    async def chunk(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        return list(await self._to_thread(self._chunk_sync, document))

    def _chunk_sync(self, document: Document) -> list[Chunk]:
        text = document.text
        sentences = _split_sentences(text)
        if not sentences:
            return []
        cs = self.config.chunk_size
        ov = self.config.overlap
        spans: list[tuple[int, int]] = []
        sent_buf: list[tuple[str, int, int]] = [sentences[0]]
        cur_start = sentences[0][1]
        cur_end = sentences[0][2]
        for sent in sentences[1:]:
            sent_text, s_start, s_end = sent
            candidate = text[cur_start:s_end]
            if self.tokenizer.count_tokens(candidate) <= cs:
                sent_buf.append(sent)
                cur_end = s_end
            else:
                spans.append((cur_start, cur_end))
                if ov > 0 and len(sent_buf) > 1:
                    # Carry over the last *ov* sentences as overlap.
                    carry = sent_buf[-ov:] if ov < len(sent_buf) else list(sent_buf)
                    cur_start = carry[0][1]
                    cur_end = carry[-1][2]
                else:
                    cur_start = s_start
                    cur_end = s_end
                sent_buf = [(sent_text, s_start, s_end)]
        spans.append((cur_start, cur_end))
        return self.make_chunks(document, spans)
