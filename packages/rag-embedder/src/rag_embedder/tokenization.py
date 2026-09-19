"""Tokenizers for rag-aio embedding strategies.

Implements the :class:`rag_core.protocols.Tokenizer` protocol for both
production (``fastembed`` / HuggingFace ``tokenizers``) and a pure-Python
fallback used in tests and offline environments.

Design notes
------------
* ``HFTokenizer`` prefers a real ``tokenizers.Tokenizer`` but never hard-fails
  at construction time when the model cannot be fetched (e.g. no network).
  It falls back to :class:`SimpleTokenizer` so chunking can proceed.
* ``SimpleTokenizer`` is a whitespace + regex word splitter with a lazily-built
  vocabulary.  ``encode``/``decode`` are exact inverses for any text the
  tokenizer has already seen, satisfying the round-trip contract the
  token-aware chunker relies on.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from tokenizers import Tokenizer as _HfTokenizer

__all__ = ["HFTokenizer", "SimpleTokenizer"]

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_UNK = "[UNK]"


class SimpleTokenizer:
    """A deterministic, dependency-light tokenizer using whitespace + regex.

    Suitable as a fallback when a HuggingFace tokenizer cannot be loaded, and
    as a fast stand-in in tests.  The vocabulary is built lazily on first use;
    for a given tokenizer instance ``encode``/``decode`` round-trip correctly.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {_UNK: 0}
        self._inv: dict[int, str] = {0: _UNK}

    # -- helpers -----------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    def _to_id(self, token: str) -> int:
        idx = self._vocab.get(token)
        if idx is None:
            idx = len(self._vocab)
            self._vocab[token] = idx
            self._inv[idx] = token
        return idx

    # -- Tokenizer protocol ------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def encode(self, text: str) -> list[int]:
        return [self._to_id(tok) for tok in self._tokenize(text)]

    def decode(self, tokens: Sequence[int]) -> str:
        parts = [self._inv.get(int(t), _UNK) for t in tokens]
        return " ".join(p for p in parts if p)


class HFTokenizer:
    """Wraps a HuggingFace ``tokenizers.Tokenizer``.

    Parameters
    ----------
    model_name_or_path:
        Model identifier or local path passed to ``Tokenizer.from_pretrained``.
        Only used when *tokenizer* is not supplied.
    tokenizer:
        An already-constructed ``tokenizers.Tokenizer``.  When provided the
        ``from_pretrained`` download path is skipped entirely — this is how
        tests supply a small locally-built tokenizer without network access.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        tokenizer: _HfTokenizer | None = None,
    ) -> None:
        self._model_name = model_name_or_path
        if tokenizer is not None:
            self._tokenizer: _HfTokenizer | None = tokenizer
        else:
            self._tokenizer = self._try_load(model_name_or_path)
        self._fallback = SimpleTokenizer() if self._tokenizer is None else None

    # -- construction ------------------------------------------------------

    @staticmethod
    def _try_load(model_name_or_path: str) -> _HfTokenizer | None:
        try:
            return _HfTokenizer.from_pretrained(model_name_or_path)
        except Exception as exc:
            log.debug(
                "HFTokenizer could not load %r, falling back to SimpleTokenizer: %s",
                model_name_or_path,
                exc,
            )
            return None

    @classmethod
    def from_identifier(cls, model_name: str) -> HFTokenizer:
        """Create an ``HFTokenizer`` that will attempt to load *model_name*.

        This may trigger a network/model download.  Tests should prefer passing a
        locally-built tokenizer instead.
        """
        return cls(model_name_or_path=model_name)

    @classmethod
    def from_tokenizer(cls, tokenizer: _HfTokenizer, model_name: str = "local") -> HFTokenizer:
        """Wrap an existing ``tokenizers.Tokenizer`` without any download."""
        return cls(model_name_or_path=model_name, tokenizer=tokenizer)

    # -- properties --------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    # -- Tokenizer protocol ------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return self._fallback.count_tokens(text)  # type: ignore[union-attr]

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            return list(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return self._fallback.encode(text)  # type: ignore[union-attr]

    def decode(self, tokens: Sequence[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(list(tokens))
        return self._fallback.decode(tokens)  # type: ignore[union-attr]
