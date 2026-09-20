"""Tokenizer abstraction for counting and truncating context text.

:class:`ContextTokenizer` wraps any tokenizer exposing one of two duck-typed
shapes and normalizes them behind a tiny, predictable interface:

* HuggingFace ``tokenizers`` shape: ``encode(text)`` returns an object with an
  ``.ids`` attribute and ``decode(ids)`` returns a string.
* rag-core :class:`~rag_core.protocols.Tokenizer` shape: ``count_tokens(text)``
  returns an int, ``encode(text)`` returns a list of ids and ``decode(ids)``
  returns a string.

If neither capability is needed for truncation, ``count_tokens`` alone is enough
to drive token budgeting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["ContextTokenizer", "WhitespaceCounter"]


class _WhitespaceEncoding:
    """Minimal stand-in for ``tokenizers.Encoding`` used by :class:`WhitespaceCounter`."""

    __slots__ = ("ids",)

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class WhitespaceCounter:
    """Whitespace tokenizer usable as a tokenizer backend or a test fallback.

    Counts and truncates on whitespace-delimited words. Each distinct word
    receives a stable integer id so :meth:`decode` can reconstruct text.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, text: str) -> _WhitespaceEncoding:
        ids: list[int] = []
        for word in text.split():
            tok = self._vocab.get(word)
            if tok is None:
                tok = len(self._vocab) + 1
                self._vocab[word] = tok
                self._id_to_word[tok] = word
            ids.append(tok)
        return _WhitespaceEncoding(ids)

    def decode(self, ids: Sequence[int]) -> str:
        words = [self._id_to_word.get(int(i), "") for i in ids]
        return " ".join(words)


class ContextTokenizer:
    """Unified tokenizer interface around a backend of variable shape."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._has_count_tokens = callable(getattr(tokenizer, "count_tokens", None))
        # Probe encode/decode capability used for truncation.
        self._has_encode = callable(getattr(tokenizer, "encode", None))
        self._has_decode = callable(getattr(tokenizer, "decode", None))

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        if self._has_count_tokens:
            value = self._tokenizer.count_tokens(text)
            return int(value)
        if self._has_encode:
            return len(self._encode_ids(text))
        # Final fallback: whitespace count.
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Return ``text`` trimmed to at most ``max_tokens`` tokens."""
        if max_tokens <= 0:
            return ""
        if not self._has_encode or not self._has_decode:
            # No real decode capability: fall back to a word-based truncation.
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens])
        ids = self._encode_ids(text)
        if len(ids) <= max_tokens:
            return text
        return str(self._tokenizer.decode(list(ids[:max_tokens])))

    def _encode_ids(self, text: str) -> list[int]:
        enc = self._tokenizer.encode(text)
        if hasattr(enc, "ids"):
            return list(enc.ids)
        # rag-core Tokenizer.encode returns a list[int].
        return list(enc)
