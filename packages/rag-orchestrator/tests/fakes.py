"""Shared offline fakes for orchestrator tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from rag_core import (
    Context,
    ContextItem,
    GenerationRequest,
    GenerationResult,
    Query,
    QueryVariant,
    RetrievalHit,
    RetrievalResult,
    Usage,
)
from rag_core.base import RagBaseModel
from rag_orchestrator.services import OrchestratorServices


class StubQueryEngine:
    """Returns the query unchanged with a single original variant."""

    def __init__(self) -> None:
        self.calls = 0

    async def process(self, query: Query) -> RagBaseModel:
        from rag_query.engine import QueryResult

        self.calls += 1
        variant = QueryVariant(query_id=query.id, text=query.text, kind="original")
        return QueryResult(query=query, variants=[variant], query_class="keyword", timings_ms={})


class StubRetriever:
    """Returns canned hits regardless of the query."""

    def __init__(self, hits: list[RetrievalHit], strategy: str = "dense") -> None:
        self.hits = hits
        self.strategy = strategy
        self.calls = 0

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(query_id=query.id, hits=list(self.hits), strategies=[self.strategy])


class StubReranker:
    """Pass-through reranker; optionally raises to exercise fallback."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    async def rerank(
        self, query: Query, candidates: list[RetrievalHit], top_k: int | None = None
    ) -> list[RetrievalHit]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(candidates)[: top_k or len(candidates)]


class StubContextBuilder:
    """Turns hits into context items, one citation each."""

    def __init__(self) -> None:
        self.calls = 0

    async def build(self, query: Query, hits: list[RetrievalHit], token_budget: int) -> Context:
        self.calls += 1
        items = [
            ContextItem(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                text=h.text or "",
                token_count=max(1, len((h.text or "").split())),
                citation_id=f"[{i + 1}]",
            )
            for i, h in enumerate(hits)
        ]
        return Context(items=items, token_budget=token_budget, strategy="stub")


class StubGenerator:
    """Deterministic generator; optionally slow or failing."""

    def __init__(
        self,
        answer: str = "stub answer",
        delay_s: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.delay_s = delay_s
        self.error = error
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return GenerationResult(
            text=self.answer,
            model="stub",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        for word in self.answer.split(" "):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield word + " "


class StubVectorStore:
    """Minimal vector store stub with ``health`` and ``count`` for app tests.

    ``async_count=True`` mirrors the real Qdrant/SQL stores, whose ``count``
    is a coroutine rather than a plain method.
    """

    def __init__(self, *, count: int = 0, healthy: bool = True, async_count: bool = False) -> None:
        self._count = count
        self._healthy = healthy
        self._async_count = async_count

    async def health(self) -> bool:
        return self._healthy

    def count(self, namespace: str | None = None) -> int:
        if self._async_count:
            return self._acount()
        return self._count

    async def _acount(self) -> int:
        return self._count


class StubCache:
    """Minimal cache stub with ``stats`` for app endpoint tests."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._store)}


def make_hits(n: int = 3) -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id=f"c{i}",
            document_id=f"d{i % 2}",
            score=1.0 - i * 0.1,
            normalized_score=1.0 - i * 0.1,
            rank=i,
            strategy="dense",
            text=f"document text {i} about authentication requirements",
        )
        for i in range(n)
    ]


def make_services(
    generator: StubGenerator | None = None,
    hits: list[RetrievalHit] | None = None,
    fallback_generators: list[StubGenerator] | None = None,
) -> OrchestratorServices:
    from rag_retrieval.fusion import ReciprocalRankFusion

    return OrchestratorServices(
        query_engine=StubQueryEngine(),  # type: ignore[arg-type]
        retrievers=[StubRetriever(hits or make_hits())],  # type: ignore[list-item]
        fusion=ReciprocalRankFusion(),
        reranker=StubReranker(),  # type: ignore[arg-type]
        context_builder=StubContextBuilder(),  # type: ignore[arg-type]
        generator=generator or StubGenerator(),  # type: ignore[arg-type]
        fallback_generators=fallback_generators or [],  # type: ignore[list-item]
    )


def make_hit_lookup(hits: list[RetrievalHit]) -> dict[str, Any]:
    return {h.chunk_id: h for h in hits}
