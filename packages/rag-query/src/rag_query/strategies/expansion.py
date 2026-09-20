"""Query expansion strategies: LLM-based or deterministic keyword extraction."""

from __future__ import annotations

import re
from typing import Final

from rag_core.generation import GenerationRequest, Message
from rag_core.queries import Query, QueryVariant

from ._base import LLM, is_short_query

# A modest english stop-word set sufficient for naive keyword extraction.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "so",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "not",
        "no",
        "what",
        "why",
        "how",
        "who",
        "when",
        "where",
        "which",
        "us",
        "them",
        "my",
        "your",
    }
)

_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Split text into alphanumeric tokens, preserving original case."""
    return _TOKEN_RE.findall(text)


def _extract_keywords(text: str, top_n: int) -> list[str]:
    """Return the ``top_n`` most frequent non-stopword tokens.

    Ties in frequency are broken by order of first appearance. Original casing
    of each token is preserved in the returned strings.
    """
    tokens = _tokenize(text)
    freq: dict[str, int] = {}
    first_index: dict[str, int] = {}
    original: dict[str, str] = {}
    for idx, token in enumerate(tokens):
        key = token.lower()
        if key in _STOPWORDS:
            continue
        freq[key] = freq.get(key, 0) + 1
        if key not in first_index:
            first_index[key] = idx
            original[key] = token
    ranked = sorted(freq, key=lambda w: (-freq[w], first_index[w]))
    return [original[w] for w in ranked[:top_n]]


def _parse_lines(text: str, cap: int) -> list[str]:
    """Parse LLM output into deduplicated, capped variant strings."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= cap:
            break
    return out


class ExpansionStrategy:
    """Expand a query into alternative phrasings.

    Without an LLM this performs deterministic keyword extraction: the most
    frequent non-stopword tokens are joined into a single focused variant
    (``"term1 term2"``). With an LLM it prompts for *count* alternative
    phrasings and parses the emitted lines.
    """

    def __init__(self, llm: LLM | None = None, count: int = 2) -> None:
        self._llm = llm
        self._count = max(1, count)

    async def transform(self, query: Query) -> list[QueryVariant]:
        text = query.text
        if is_short_query(text):
            return []
        llm = self._llm
        if llm is None:
            return self._deterministic(text, query.id)
        return await self._llm_expand(text, query.id, llm)

    def _deterministic(self, text: str, query_id: str) -> list[QueryVariant]:
        keywords = _extract_keywords(text, self._count)
        if not keywords:
            return []
        return [
            QueryVariant(
                query_id=query_id,
                text=" ".join(keywords),
                kind="expansion",
                strategy="expansion",
            )
        ]

    async def _llm_expand(self, text: str, query_id: str, llm: LLM) -> list[QueryVariant]:
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a query-expansion assistant. Produce alternative "
                        "phrasings of the user's query, one per line, without extra "
                        "formatting or commentary."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Generate {self._count} alternative phrasings of the "
                        f"following query, one per line:\n\n{text}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=128,
        )
        result = await llm.generate(request)
        lines = _parse_lines(result.text, self._count)
        if not lines:
            return []
        return [
            QueryVariant(
                query_id=query_id,
                text=line,
                kind="expansion",
                strategy="expansion",
            )
            for line in lines
        ]
