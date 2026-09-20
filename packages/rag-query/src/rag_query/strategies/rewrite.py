"""Query rewrite strategy: resolve conversational references into a standalone query."""

from __future__ import annotations

import re

from rag_core.generation import GenerationRequest, Message
from rag_core.queries import Query, QueryVariant

from ._base import LLM, is_short_query

_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


def _clean_whitespace(text: str) -> str:
    """Collapse internal whitespace and strip the result."""
    return _WHITESPACE_RE.sub(" ", text).strip()


class RewriteStrategy:
    """Rewrite a query into a standalone search query.

    Without an LLM this is a documented pass-through that only cleans
    whitespace. With an LLM it prompts for a single standalone search query
    that resolves conversational references (e.g. "it", "they", "previous").
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def transform(self, query: Query) -> list[QueryVariant]:
        text = query.text
        if is_short_query(text):
            return []
        llm = self._llm
        if llm is None:
            rewritten = _clean_whitespace(text)
        else:
            rewritten = await self._llm_rewrite(text, llm)
        if not rewritten:
            return []
        return [
            QueryVariant(
                query_id=query.id,
                text=rewritten,
                kind="rewrite",
                strategy="rewrite",
            )
        ]

    async def _llm_rewrite(self, text: str, llm: LLM) -> str:
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a query-rewriting assistant. Rewrite the user's "
                        "query into a single standalone search query that resolves "
                        "any conversational references. Return only the rewritten "
                        "query text."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Rewrite the following into a single standalone search "
                        f"query, resolving conversational references:\n\n{text}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=128,
        )
        result = await llm.generate(request)
        # Treat model output as untrusted data: take a single, cleaned phrase.
        return _clean_whitespace(result.text.strip().splitlines()[0] if result.text.strip() else "")
