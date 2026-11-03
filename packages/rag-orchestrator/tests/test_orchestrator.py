"""Tests for the Orchestrator runtime: ask, streaming, timeout, fallback, caching."""

from __future__ import annotations

from typing import Any

import fakes
import pytest
from rag_core import (
    GenerationRequest,
    OperationTimeout,
    ProviderError,
    ProviderUnavailableError,
)
from rag_core.errors import RateLimitError, RerankError
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.orchestrator import (
    SYSTEM_PROMPT,
    AskResult,
    Orchestrator,
    OrchestratorConfigError,
    PreparedQuery,
    StreamDelta,
    StreamEvent,
    StreamFinal,
)
from rag_orchestrator.services import NoOpObserver, OrchestratorServices

# -- construction --------------------------------------------------------------


def test_missing_components_raise() -> None:
    with pytest.raises(OrchestratorConfigError):
        Orchestrator(OrchestratorServices())


# -- ask -----------------------------------------------------------------------


async def test_ask_returns_answer_with_citations_and_timings() -> None:
    gen = fakes.StubGenerator(answer="the answer")
    orch = Orchestrator(fakes.make_services(generator=gen))
    result = await orch.ask("what are the authentication requirements?", query_id="q-fixed")
    assert isinstance(result, AskResult)
    assert result.answer == "the answer"
    assert result.query_id == "q-fixed"
    assert result.citations, "citations must be derived from the context"
    assert result.citations[0].citation_id == "[1]"
    assert result.context is not None and result.context.items
    for stage in ("query", "retrieve", "rerank", "context", "generate", "total_ms"):
        assert stage in result.timings_ms
    assert result.metrics["prompt_tokens"] == 10
    assert result.metrics["cached"] is False


async def test_system_prompt_is_separate_from_document_content() -> None:
    gen = fakes.StubGenerator()
    orch = Orchestrator(fakes.make_services(generator=gen))
    await orch.ask("question?")
    request = gen.requests[0]
    assert isinstance(request, GenerationRequest)
    assert request.messages[0].role == "system"
    assert request.messages[0].content == SYSTEM_PROMPT
    user_content = request.messages[1].content
    assert "Question: question?" in user_content


async def test_rerank_disabled_skips_reranker() -> None:
    cfg = PipelineConfig(rerank={"enable": False})
    reranker = fakes.StubReranker()
    services = fakes.make_services()
    services.reranker = reranker
    orch = Orchestrator(services, cfg)
    await orch.ask("q")
    assert reranker.calls == 0


async def test_rerank_error_falls_back_to_retrieval_scores() -> None:
    """A RerankError in the rerank stage falls back to un-reranked retrieval hits."""
    reranker = fakes.StubReranker(error=RerankError("rerank unavailable"))
    services = fakes.make_services()
    services.reranker = reranker
    orch = Orchestrator(services)
    result = await orch.ask("q")
    assert result.answer == "stub answer"
    assert reranker.calls == 1  # was invoked (and failed)


# -- timeout / cancellation ------------------------------------------------------


async def test_timeout_raises_operation_timeout() -> None:
    gen = fakes.StubGenerator(delay_s=5.0)
    cfg = PipelineConfig(timeout_s=0.05)
    orch = Orchestrator(fakes.make_services(generator=gen), cfg)
    with pytest.raises(OperationTimeout):
        await orch.ask("slow query")


# -- fallback --------------------------------------------------------------


async def test_fallback_generator_used_on_transient_failure() -> None:
    primary = fakes.StubGenerator(error=ProviderUnavailableError("down"))
    fallback = fakes.StubGenerator(answer="fallback answer")
    orch = Orchestrator(fakes.make_services(generator=primary, fallback_generators=[fallback]))
    result = await orch.ask("q")
    assert result.answer == "fallback answer"


async def test_non_transient_provider_error_propagates() -> None:
    primary = fakes.StubGenerator(error=ProviderError("bad api key"))
    fallback = fakes.StubGenerator(answer="fallback answer")
    orch = Orchestrator(fakes.make_services(generator=primary, fallback_generators=[fallback]))
    with pytest.raises(ProviderError):
        await orch.ask("q")


async def test_rate_limit_triggers_fallback() -> None:
    primary = fakes.StubGenerator(error=RateLimitError("429"))
    fallback = fakes.StubGenerator(answer="ok")
    orch = Orchestrator(fakes.make_services(generator=primary, fallback_generators=[fallback]))
    assert (await orch.ask("q")).answer == "ok"


async def test_all_fallbacks_exhausted_raises_last_error() -> None:
    """When every fallback also fails, the last ProviderError is re-raised."""
    primary = fakes.StubGenerator(error=ProviderUnavailableError("primary down"))
    fb1 = fakes.StubGenerator(error=ProviderUnavailableError("fallback1 down"))
    fb2 = fakes.StubGenerator(error=ProviderUnavailableError("fallback2 down"))
    orch = Orchestrator(fakes.make_services(generator=primary, fallback_generators=[fb1, fb2]))
    with pytest.raises(ProviderUnavailableError, match="fallback2 down"):
        await orch.ask("q")


# -- streaming --------------------------------------------------------------


async def test_stream_yields_answer_deltas() -> None:
    gen = fakes.StubGenerator(answer="one two three")
    orch = Orchestrator(fakes.make_services(generator=gen))
    stream = await orch.ask("q", stream=True)
    assert not isinstance(stream, AskResult)
    chunks = [chunk async for chunk in stream]
    assert "".join(chunks).split() == ["one", "two", "three"]


async def test_stream_reconstructs_full_answer() -> None:
    """collect_stream: reassemble deltas back into the generator's answer."""
    answer = "the reconstructed answer"
    gen = fakes.StubGenerator(answer=answer)
    orch = Orchestrator(fakes.make_services(generator=gen))
    stream = await orch.ask("q", stream=True)
    assert not isinstance(stream, AskResult)
    collected = ""
    async for delta in stream:
        collected += delta
    assert collected.strip() == answer
    assert len(gen.requests) == 1


async def test_stream_timeout_raises_operation_timeout() -> None:
    gen = fakes.StubGenerator(delay_s=5.0)
    cfg = PipelineConfig(timeout_s=0.05)
    orch = Orchestrator(fakes.make_services(generator=gen), cfg)
    stream = await orch.ask("q", stream=True)
    assert not isinstance(stream, AskResult)
    with pytest.raises(OperationTimeout):
        async for _ in stream:
            pass


# -- prepare -----------------------------------------------------------------


async def test_prepare_returns_prepared_query_without_generating() -> None:
    gen = fakes.StubGenerator(answer="not generated")
    orch = Orchestrator(fakes.make_services(generator=gen))
    prepared = await orch.prepare("what are the authentication requirements?", query_id="q-prepare")
    assert isinstance(prepared, PreparedQuery)
    assert prepared.query_id == "q-prepare"
    assert prepared.citations, "prepare must build citations from the context"
    assert prepared.citations[0].citation_id == "[1]"
    assert prepared.context is not None and prepared.context.items
    for stage in ("query", "retrieve", "rerank", "context"):
        assert stage in prepared.timings_ms, "per-stage timings must be captured"
    # prepare stops before generation, but builds the exact GenerationRequest.
    assert gen.requests == []
    assert prepared.request.stream is False
    assert prepared.request.messages[0].content == SYSTEM_PROMPT


async def test_prepare_records_prepare_end_event() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class RecordingObserver(NoOpObserver):
        def record(self, event: str, attributes: dict[str, Any] | None = None) -> None:
            events.append((event, attributes or {}))

    services = fakes.make_services()
    services.observer = RecordingObserver()
    orch = Orchestrator(services)
    await orch.prepare("q", query_id="q-obs")
    assert any(name == "prepare.end" and attrs.get("query_id") == "q-obs" for name, attrs in events)


# -- ask_stream ----------------------------------------------------------------


async def test_ask_stream_matches_ask_stream_true() -> None:
    gen = fakes.StubGenerator(answer="alpha beta gamma")
    orch = Orchestrator(fakes.make_services(generator=gen))
    legacy = await orch.ask("q", stream=True)
    modern = await orch.ask_stream("q")
    assert not isinstance(modern, AskResult)
    legacy_text = "".join([chunk async for chunk in legacy])
    modern_text = "".join([chunk async for chunk in modern])
    assert legacy_text.strip() == "alpha beta gamma"
    assert modern_text == legacy_text


async def test_ask_stream_timeout_raises_operation_timeout() -> None:
    gen = fakes.StubGenerator(delay_s=5.0)
    cfg = PipelineConfig(timeout_s=0.05)
    orch = Orchestrator(fakes.make_services(generator=gen), cfg)
    stream = await orch.ask_stream("slow query")
    with pytest.raises(OperationTimeout):
        async for _ in stream:
            pass


# -- stream_events ---------------------------------------------------------------


async def test_stream_events_yields_deltas_then_single_final() -> None:
    answer = "one two three"
    gen = fakes.StubGenerator(answer=answer)
    orch = Orchestrator(fakes.make_services(generator=gen))
    events: list[StreamEvent] = [
        event async for event in await orch.stream_events("q", query_id="q-sf")
    ]
    deltas = [e for e in events if isinstance(e, StreamDelta)]
    finals = [e for e in events if isinstance(e, StreamFinal)]
    assert deltas, "must yield at least one delta frame"
    assert len(events) == len(deltas) + len(finals), "only StreamDelta/StreamFinal frames"
    assert "".join(d.text for d in deltas).strip() == answer
    assert len(finals) == 1, "exactly one terminal StreamFinal"
    final = finals[0]
    assert final.type == "final" and deltas[0].type == "delta"
    assert final.answer == "".join(d.text for d in deltas)
    assert final.citations and final.citations[0].citation_id == "[1]"
    assert "query" in final.timings_ms and "retrieve" in final.timings_ms
    assert final.query_id == "q-sf"


async def test_stream_events_does_not_call_generate() -> None:
    class CountingGenerator(fakes.StubGenerator):
        generate_calls = 0

        async def generate(self, request: GenerationRequest) -> Any:
            self.generate_calls += 1
            return await super().generate(request)

    gen = CountingGenerator(answer="ignored")
    orch = Orchestrator(fakes.make_services(generator=gen))
    events = [event async for event in await orch.stream_events("q")]
    assert gen.generate_calls == 0, "stream_events must not hit the sync generate path"
    assert isinstance(events[-1], StreamFinal)


# -- namespace threading ---------------------------------------------------------


async def test_namespace_threaded_to_retriever_via_ask() -> None:
    retriever = fakes.StubRetriever(fakes.make_hits())
    services = fakes.make_services()
    services.retrievers = [retriever]
    orch = Orchestrator(services)
    await orch.ask("q", namespace="tenant-a")
    assert retriever.queries, "retriever must have been called"
    assert retriever.queries[-1].namespace == "tenant-a"


async def test_namespace_default_is_none() -> None:
    retriever = fakes.StubRetriever(fakes.make_hits())
    services = fakes.make_services()
    services.retrievers = [retriever]
    orch = Orchestrator(services)
    await orch.prepare("q")
    assert retriever.queries[-1].namespace is None


async def test_namespace_threaded_via_prepare_and_stream_events() -> None:
    retriever = fakes.StubRetriever(fakes.make_hits())
    services = fakes.make_services()
    services.retrievers = [retriever]
    orch = Orchestrator(services)
    await orch.prepare("q", namespace="ns-prepare")
    assert retriever.queries[-1].namespace == "ns-prepare"
    async for _ in await orch.stream_events("q", namespace="ns-stream"):
        pass
    assert retriever.queries[-1].namespace == "ns-stream"


# -- correlation ids ---------------------------------------------------------


async def test_correlation_id_recorded_in_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_observe.logging import get_correlation_id

    gen = fakes.StubGenerator()
    orch = Orchestrator(fakes.make_services(generator=gen))
    result = await orch.ask("q", correlation_id="cid-42")
    assert result.metrics["correlation_id"] == "cid-42"
    assert get_correlation_id() is None  # reset after the call
