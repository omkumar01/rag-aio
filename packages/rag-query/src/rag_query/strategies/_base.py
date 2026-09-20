"""Shared primitives for query strategies.

Defines the duck-typed :class:`LLM` protocol (only ``generate`` is required by
strategies) plus small helpers shared by every strategy implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_core.generation import GenerationRequest, GenerationResult

# A query with fewer than this many whitespace tokens is considered too short
# to transform. Kept in sync with ``QueryConfig.min_query_length``'s default.
_MIN_QUERY_WORDS: int = 2


@runtime_checkable
class LLM(Protocol):
    """Minimal duck-typed generator interface used by query strategies.

    Only :meth:`generate` is required. Strategies never echo LLM output back
    into a prompt: model text is always treated as untrusted data parsed into
    a variant, never re-injected as a system message.
    """

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


def is_short_query(text: str) -> bool:
    """Return ``True`` when *text* is blank or has fewer than two tokens."""
    return len(text.split()) < _MIN_QUERY_WORDS
