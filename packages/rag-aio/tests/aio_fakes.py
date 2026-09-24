"""Shared offline fakes for rag-aio tests, mirroring rag-orchestrator/tests/fakes.py."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_core import (
    GenerationRequest,
    GenerationResult,
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

    async def process(self, query: Any) -> RagBaseModel:
        from rag_core.queries import QueryVariant
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

    async def retrieve(self, query: Any) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(query_id=query.id, hits=list(self.hits), strategies=[self.strategy])


class StubReranker:
    """Pass-through reranker."""

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(
        self, query: Any, candidates: list[RetrievalHit], top_k: int | None = None
    ) -> list[RetrievalHit]:
        self.calls += 1
        return list(candidates)[: top_k or len(candidates)]


class StubContextBuilder:
    """Turns hits into context items, one citation each."""

    def __init__(self) -> None:
        self.calls = 0

    async def build(self, query: Any, hits: list[RetrievalHit], token_budget: int) -> Any:
        from rag_core.context import Context, ContextItem

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
    """Deterministic generator for tests."""

    def __init__(self, answer: str = "stub answer") -> None:
        self.answer = answer
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.answer,
            model="stub",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        for word in self.answer.split(" "):
            yield word + " "


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
) -> OrchestratorServices:
    from rag_retrieval.fusion import ReciprocalRankFusion

    return OrchestratorServices(
        query_engine=StubQueryEngine(),  # type: ignore[arg-type]
        retrievers=[StubRetriever(hits or make_hits())],  # type: ignore[list-item]
        fusion=ReciprocalRankFusion(),
        reranker=StubReranker(),  # type: ignore[arg-type]
        context_builder=StubContextBuilder(),  # type: ignore[arg-type]
        generator=generator or StubGenerator(),  # type: ignore[arg-type]
    )
