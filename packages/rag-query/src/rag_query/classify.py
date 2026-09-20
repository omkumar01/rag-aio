"""Heuristic query classification into intent buckets."""

from __future__ import annotations

from enum import StrEnum

# Question words that indicate an interrogative query.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "why",
        "how",
        "who",
        "when",
        "where",
        "which",
        "does",
        "is",
        "are",
        "can",
    }
)

# Phrases indicating navigational intent.
_NAV_PHRASES: tuple[str, ...] = ("show me", "find me")

# A query with fewer words than this is treated as a bare keyword query.
_KEYWORD_MAX_WORDS: int = 3


class QueryClass(StrEnum):
    """Coarse intent classification for a query string."""

    keyword = "keyword"
    question = "question"
    conversational = "conversational"
    navigational = "navigational"


def classify_query(text: str) -> QueryClass:
    """Classify a query by intent using lightweight heuristics.

    The rules, evaluated in priority order:

    1. The query ends with ``?`` **or** starts with a question word
       (``what``, ``why``, ... ``can``) -> :attr:`QueryClass.question`.
    2. The query contains ``show me`` / ``find me`` ->
       :attr:`QueryClass.navigational`.
    3. The query has at most three tokens -> :attr:`QueryClass.keyword`.
    4. Otherwise -> :attr:`QueryClass.conversational`.
    """
    normalized = text.strip()
    lowered = normalized.lower()
    words = lowered.split()

    if normalized.endswith("?") or (words and words[0] in _QUESTION_WORDS):
        return QueryClass.question
    if any(phrase in lowered for phrase in _NAV_PHRASES):
        return QueryClass.navigational
    if len(words) <= _KEYWORD_MAX_WORDS:
        return QueryClass.keyword
    return QueryClass.conversational
