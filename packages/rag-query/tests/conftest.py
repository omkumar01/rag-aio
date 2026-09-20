"""Shared test doubles for rag-query (offline fakes only)."""

from __future__ import annotations

import asyncio

import pytest
from rag_core.generation import GenerationRequest, GenerationResult
from rag_core.queries import Query, QueryVariant
from rag_core.retrieval import RetrievalResult


class FakeLLM:
    """A duck-typed generator returning a fixed response."""

    def __init__(self, text: str = "", model: str = "fake") -> None:
        self.text = text
        self.model = model
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        await asyncio.sleep(0)
        return GenerationResult(text=self.text, model=self.model, finish_reason="stop")


class RecordingRetriever:
    """Retriever recording call count and returning a tagged result."""

    def __init__(self, tag: str = "r") -> None:
        self.tag = tag
        self.calls = 0

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.calls += 1
        await asyncio.sleep(0)
        return RetrievalResult(query_id=query.id, hits=[], strategies=[self.tag])


class FailingRetriever:
    """Retriever that always raises."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc if exc is not None else RuntimeError("retrieval failed")
        self.calls = 0

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.calls += 1
        await asyncio.sleep(0)
        raise self.exc


class CountingRetriever:
    """Retriever that tracks the peak number of concurrent invocations."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.calls = 0

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.current += 1
        self.peak = max(self.peak, self.current)
        self.calls += 1
        await asyncio.sleep(0.02)
        self.current -= 1
        return RetrievalResult(query_id=query.id, hits=[], strategies=["counting"])


class VariantStrategy:
    """Strategy emitting a fixed list of variant texts for a given kind."""

    def __init__(
        self,
        texts: list[str],
        kind: str = "rewrite",
        strategy: str = "rewrite",
    ) -> None:
        self._texts = texts
        self._kind = kind
        self._strategy = strategy

    async def transform(self, query: Query) -> list[QueryVariant]:
        await asyncio.sleep(0)
        return [
            QueryVariant(
                query_id=query.id,
                text=text,
                kind=self._kind,
                strategy=self._strategy,
            )
            for text in self._texts
        ]


@pytest.fixture
def fake_llm():
    """Factory returning a :class:`FakeLLM` with the given response text."""

    def _make(text: str = "", model: str = "fake") -> FakeLLM:
        return FakeLLM(text=text, model=model)

    return _make


@pytest.fixture
def recording_retriever():
    """Factory returning a :class:`RecordingRetriever`."""

    def _make(tag: str = "r") -> RecordingRetriever:
        return RecordingRetriever(tag=tag)

    return _make


@pytest.fixture
def failing_retriever():
    """Factory returning a :class:`FailingRetriever`."""

    def _make(exc: Exception | None = None) -> FailingRetriever:
        return FailingRetriever(exc=exc)

    return _make


@pytest.fixture
def counting_retriever():
    """Factory returning a :class:`CountingRetriever`."""

    def _make() -> CountingRetriever:
        return CountingRetriever()

    return _make


@pytest.fixture
def variant_strategy():
    """Factory returning a :class:`VariantStrategy`."""

    def _make(
        texts: list[str], kind: str = "rewrite", strategy: str = "rewrite"
    ) -> VariantStrategy:
        return VariantStrategy(texts=texts, kind=kind, strategy=strategy)

    return _make
