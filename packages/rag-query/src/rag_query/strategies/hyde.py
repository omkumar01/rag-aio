"""HyDE (Hypothetical Document Embeddings) query strategy.

Generates a hypothetical answer passage for a question and uses that passage
as a query variant for dense retrieval.
"""

from __future__ import annotations

from rag_core.errors import ConfigError
from rag_core.generation import GenerationRequest, Message
from rag_core.queries import Query, QueryVariant

from ._base import LLM, is_short_query


class HyDEStrategy:
    """Generate a hypothetical answer passage via an LLM.

    An LLM is mandatory: constructing this strategy without one raises
    :class:`~rag_core.errors.ConfigError`. The hypothetical answer is treated as
    untrusted data and is never echoed back into a prompt.
    """

    def __init__(self, llm: LLM | None = None) -> None:
        if llm is None:
            raise ConfigError(
                "HyDEStrategy requires an LLM; pass llm=... or use a "
                "deterministic strategy instead",
            )
        self._llm: LLM = llm

    async def transform(self, query: Query) -> list[QueryVariant]:
        if is_short_query(query.text):
            return []
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant. Write a concise hypothetical "
                        "answer passage for the question below."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Write a hypothetical answer passage for the following "
                        f"question:\n\n{query.text}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=512,
        )
        result = await self._llm.generate(request)
        answer = result.text.strip()
        if not answer:
            return []
        return [
            QueryVariant(
                query_id=query.id,
                text=answer,
                kind="hyde",
                strategy="hyde",
            )
        ]
