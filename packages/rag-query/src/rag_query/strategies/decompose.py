"""Query decomposition strategies: deterministic split or LLM-generated subquestions."""

from __future__ import annotations

import json
import re

from rag_core.generation import GenerationRequest, Message
from rag_core.queries import Query, QueryVariant

from ._base import LLM, is_short_query

# Match " and then ", " and ", or a semicolon separator (case-insensitive).
# "and then" is listed first so it takes priority over the plain "and" split.
_CONJUNCTION_RE: re.Pattern[str] = re.compile(r"\s+and\s+then\s+|\s+and\s+|;\s*")


def _split_conjunctions(text: str) -> list[str]:
    """Split a query on coordinating conjunctions and semicolons."""
    parts = _CONJUNCTION_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def _clean_item(item: object) -> str:
    """Coerce a JSON array element into a stripped, non-quoted string."""
    if isinstance(item, str):
        return item.strip().strip('"').strip()
    return str(item).strip().strip('"').strip()


def _extract_json_array(text: str) -> list[str]:
    """Robustly extract a JSON string array from *text*.

    First attempts to locate the outermost ``[...]`` block and parse it. If that
    fails (or there is no bracketed block), falls back to a newline-delimited
    parse of the raw text.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            items = [_clean_item(x) for x in data]
            return [item for item in items if item]

    # Fallback: treat each non-empty line as a subquestion.
    return [_clean_item(line) for line in text.splitlines() if line.strip()]


class DecomposeStrategy:
    """Decompose a multi-part query into subqueries.

    Without an LLM this splits on coordinating conjunctions ("and", "and then")
    and semicolons into ``"subquery"`` variants. With an LLM it prompts for a
    JSON array of subquestions and parses the result with a robust fallback to
    line-based parsing.
    """

    def __init__(self, llm: LLM | None = None, max_subqueries: int = 3) -> None:
        self._llm = llm
        self._max_subqueries = max(1, max_subqueries)

    async def transform(self, query: Query) -> list[QueryVariant]:
        text = query.text
        if is_short_query(text):
            return []
        llm = self._llm
        if llm is None:
            subqueries = _split_conjunctions(text)
        else:
            subqueries = await self._llm_decompose(text, llm)
        subqueries = subqueries[: self._max_subqueries]
        if not subqueries:
            return []
        return [
            QueryVariant(
                query_id=query.id,
                text=sq,
                kind="subquery",
                strategy="decompose",
            )
            for sq in subqueries
        ]

    async def _llm_decompose(self, text: str, llm: LLM) -> list[str]:
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You decompose queries into sub-questions. Return a JSON "
                        "array of strings, one per sub-question."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Decompose the following query into a JSON array of "
                        f"sub-questions:\n\n{text}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=512,
        )
        result = await llm.generate(request)
        # Treat model output as untrusted data; never re-inject into prompts.
        return _extract_json_array(result.text)
