"""The rag-aio runtime: composes wired components into an ``ask`` pipeline.

This module is pure orchestration glue. It sequences the contract-bound
components held in :class:`~rag_orchestrator.services.OrchestratorServices`
(query intelligence -> parallel retrieval -> fusion -> rerank -> context ->
generation), enforces budget/cancellation semantics, and assembles an
:class:`AskResult`. No vendor logic is reimplemented here.

Heavy collaborators (``rag_context``, ``rag_query.retrieval_exec``) are imported
lazily inside methods so ``import rag_orchestrator`` stays dependency-light.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import Field
from rag_core import (
    Citation,
    Context,
    GenerationRequest,
    GenerationResult,
    Message,
    OperationTimeout,
    ProviderError,
    ProviderUnavailableError,
    Query,
    RateLimitError,
    RerankError,
    RetrievalHit,
    RetrievalResult,
    new_id,
)
from rag_core.base import RagBaseModel

from .config import PipelineConfig
from .services import OrchestratorServices

if TYPE_CHECKING:
    from rag_query.engine import QueryResult

__all__ = [
    "SYSTEM_PROMPT",
    "AskResult",
    "Orchestrator",
    "OrchestratorConfigError",
]

_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Answer the question using only the "
    "provided context. Cite sources numerically as [n]. If the context does "
    "not contain enough information, say so."
)
SYSTEM_PROMPT: str = _SYSTEM_PROMPT


class OrchestratorConfigError(Exception):
    """Raised when the orchestrator is constructed without required components."""


class AskResult(RagBaseModel):
    """The synchronous result of :meth:`Orchestrator.ask`.

    Attributes:
        answer: The generated answer text.
        citations: Source citations derived from the assembled context.
        context: The token-budgeted context (useful for inspection/debugging).
        timings_ms: Per-stage wall-clock timings (``query``, ``retrieve``,
            ``rerank``, ``context``, ``generate``, ``total``).
        query_id: Identifier for the query (stable within one ask).
        metrics: Usage/cost metrics from generation plus a ``cached`` flag.
    """

    answer: str
    citations: list[Citation]
    context: Context | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    query_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)


def _stage_overrides(config: PipelineConfig, stage: str) -> dict[str, Any]:
    stages = {
        "ingestion": config.ingestion,
        "embedding": config.embedding,
        "retrieval": config.retrieval,
        "rerank": config.rerank,
        "context": config.context,
        "generation": config.generation,
    }
    return stages[stage].overrides


def _top_k(config: PipelineConfig, default: int = 10) -> int:
    return int(_stage_overrides(config, "retrieval").get("top_k", default))


def _rerank_top_k(config: PipelineConfig) -> int:
    return int(
        _stage_overrides(config, "retrieval").get(
            "top_k", _stage_overrides(config, "rerank").get("top_k", 10)
        )
    )


def _rerank_cap(config: PipelineConfig) -> int:
    return int(_stage_overrides(config, "rerank").get("max_candidates", 50))


def _context_budget(config: PipelineConfig) -> int:
    return int(_stage_overrides(config, "context").get("token_budget", config.token_budget or 2048))


def _max_tokens(config: PipelineConfig) -> int:
    return int(_stage_overrides(config, "generation").get("max_tokens", config.token_budget or 256))


def _temperature(config: PipelineConfig) -> float:
    return float(_stage_overrides(config, "generation").get("temperature", 0.2))


def _generation_model(config: PipelineConfig) -> str | None:
    return _stage_overrides(config, "generation").get("model") or None


def _render_prompt(context_text: str, query_text: str) -> str:
    if context_text:
        return f"{context_text}\n\nQuestion: {query_text}\nAnswer:"
    return f"Question: {query_text}\nAnswer:"


class Orchestrator:
    """Runs a configured RAG pipeline for a single query.

    Stateless across requests beyond its injected
    :class:`~rag_orchestrator.services.OrchestratorServices` and
    :class:`~rag_orchestrator.config.PipelineConfig`. Each ``ask`` builds an
    ephemeral :class:`~rag_query.retrieval_exec.ParallelRetrievalExecutor` from
    ``services.retrievers`` bounded by ``config.concurrency``.
    """

    def __init__(
        self,
        services: OrchestratorServices,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        self.services = services
        self.config = pipeline_config or PipelineConfig.local_default()
        self._validate_required()

    # -- construction validation --------------------------------------------

    def _validate_required(self) -> None:
        missing: list[str] = []
        if self.services.query_engine is None:
            missing.append("query_engine")
        if not self.services.ensure_retrievers():
            missing.append("retrievers")
        if self.services.fusion is None:
            missing.append("fusion")
        if self.services.context_builder is None:
            missing.append("context_builder")
        if self.services.generator is None:
            missing.append("generator")
        if missing:
            msg = "Orchestrator requires: " + ", ".join(missing)
            raise OrchestratorConfigError(msg)

    # -- public API ---------------------------------------------------------

    async def ask(
        self,
        query_text: str,
        *,
        stream: bool = False,
        overrides: dict[str, Any] | None = None,
        query_id: str | None = None,
        correlation_id: str | None = None,
    ) -> AskResult | AsyncIterator[str]:
        """Run the pipeline for ``query_text``.

        ``stream=True`` returns an async iterator of text deltas. Timeouts come
        from ``pipeline_config.timeout_s`` and surface as
        :class:`~rag_core.OperationTimeout`; client disconnects propagate as
        ``asyncio.CancelledError``.
        """
        config = self.config.merge_overrides(overrides)
        qid = query_id or new_id()
        cid = correlation_id or qid
        self.services.observer.record("ask.start", {"query_id": qid, "correlation_id": cid})
        if stream:
            return self._stream(query_text, config, qid, cid)
        try:
            return await self._ask(query_text, config, qid, cid)
        finally:
            self.services.observer.record("ask.end", {"query_id": qid})

    # -- sync pipeline ------------------------------------------------------

    async def _ask(self, query_text: str, config: PipelineConfig, qid: str, cid: str) -> AskResult:
        from rag_observe import set_correlation_id

        set_correlation_id(cid)
        timings: dict[str, float] = {}
        start = time.perf_counter()
        try:
            async with asyncio.timeout(config.timeout_s):
                request, context, citations = await self._prepare(query_text, config, qid, timings)
                result = await self._timed("generate", self._generate(request), timings)
                assert isinstance(result, GenerationResult)
                return self._build_result(
                    result, citations, context, timings, qid, cid, start, cached=False
                )
        except TimeoutError as exc:
            raise OperationTimeout(
                "ask exceeded configured timeout",
                details={"timeout_s": config.timeout_s, "query_id": qid},
            ) from exc
        finally:
            set_correlation_id(None)

    async def _prepare(
        self, query_text: str, config: PipelineConfig, qid: str, timings: dict[str, float]
    ) -> tuple[GenerationRequest, Context | None, list[Citation]]:
        from rag_context.citations import build_citations
        from rag_context.render import render_context

        query, qres = await self._process_query(query_text, config, qid, timings)
        results = await self._timed("retrieve", self._retrieve(qres, config, timings), timings)
        fused = self._fuse(results, config)
        hits = await self._timed("rerank", self._rerank(query, fused, config), timings)
        context = await self._timed("context", self._build_context(query, hits, config), timings)
        citations = build_citations(context)
        rendered = render_context(context)
        request = self._build_request(query, rendered, config, stream=False)
        return request, context, citations

    async def _process_query(
        self, query_text: str, config: PipelineConfig, qid: str, timings: dict[str, float]
    ) -> tuple[Query, QueryResult]:
        engine = self.services.query_engine
        assert engine is not None  # validated in __init__
        query = Query(text=query_text, id=qid, top_k=_top_k(config), deadline_s=config.timeout_s)
        qres = await self._timed("query", engine.process(query), timings)
        return qres.query, qres

    async def _retrieve(
        self, qres: QueryResult, config: PipelineConfig, timings: dict[str, float]
    ) -> list[RetrievalResult]:
        from rag_query.retrieval_exec import ParallelRetrievalExecutor

        retrievers = self.services.ensure_retrievers()
        executor = ParallelRetrievalExecutor(retrievers, max_concurrency=max(1, config.concurrency))
        top_k = _top_k(config) * 2  # breadth for rerank + context
        results = await executor.execute(qres.query, qres.variants, top_k)
        timings["merge_ms"] = 0.0
        return list(results)

    def _fuse(self, results: list[RetrievalResult], config: PipelineConfig) -> list[RetrievalHit]:
        if not results:
            return []
        fusion = self.services.fusion
        assert fusion is not None  # validated in __init__
        top_k = _rerank_cap(config)
        fused = fusion.fuse(results, top_k, dedup=True)
        return list(fused.hits)

    async def _rerank(
        self, query: Query, hits: list[RetrievalHit], config: PipelineConfig
    ) -> list[RetrievalHit]:
        reranker = self.services.reranker
        final_top_k = _rerank_top_k(config)
        if not config.rerank.enable or reranker is None:
            return hits[:final_top_k]
        candidates = hits[: max(_rerank_cap(config), 0)]
        try:
            return await reranker.rerank(query, candidates, top_k=final_top_k)
        except RerankError:
            self.services.observer.record("rerank.unavailable", {})
            return hits[:final_top_k]  # fall back to retrieval scores

    async def _build_context(
        self, query: Query, hits: list[RetrievalHit], config: PipelineConfig
    ) -> Context:
        builder = self.services.context_builder
        assert builder is not None  # validated in __init__
        return await builder.build(query, hits, _context_budget(config))

    def _build_request(
        self, query: Query, rendered: str, config: PipelineConfig, stream: bool
    ) -> GenerationRequest:
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=_render_prompt(rendered, query.text)),
        ]
        return GenerationRequest(
            messages=messages,
            model=_generation_model(config),
            temperature=_temperature(config),
            max_tokens=_max_tokens(config),
            stream=stream,
            metadata={"query_id": query.id},
        )

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        generator = self.services.generator
        assert generator is not None  # validated in __init__
        last_exc: Exception | None = None
        try:
            return await generator.generate(request)
        except ProviderError as exc:
            if not isinstance(exc, (ProviderUnavailableError, RateLimitError)):
                raise  # non-transient (e.g. auth): do not fall back.
            self.services.observer.record("generate.fallback", {"reason": exc.code})
            last_exc = exc
        for fb in self.services.fallback_generators:
            try:
                return await fb.generate(request)
            except ProviderError as exc:
                last_exc = exc
                continue
        assert last_exc is not None  # primary was transient; some error exists to re-raise
        raise last_exc

    def _build_result(
        self,
        result: GenerationResult,
        citations: list[Citation],
        context: Context | None,
        timings: dict[str, float],
        qid: str,
        cid: str,
        start: float,
        cached: bool,
    ) -> AskResult:
        timings["total_ms"] = (time.perf_counter() - start) * 1000.0
        metrics: dict[str, Any] = {}
        if result.usage is not None:
            metrics.update(result.usage.model_dump())
        metrics["cached"] = cached
        metrics["correlation_id"] = cid
        return AskResult(
            answer=result.text,
            citations=citations,
            context=context,
            timings_ms=timings,
            query_id=qid,
            metrics=metrics,
        )

    # -- streaming ----------------------------------------------------------

    def _stream(
        self, query_text: str, config: PipelineConfig, qid: str, cid: str
    ) -> AsyncIterator[str]:
        """Return an async iterator of generation deltas for the same pipeline."""
        from rag_observe import set_correlation_id

        async def _gen() -> AsyncIterator[str]:
            set_correlation_id(cid)
            try:
                async with asyncio.timeout(config.timeout_s):
                    request, _context, _citations = await self._prepare(query_text, config, qid, {})
                    generator = self.services.generator
                    assert generator is not None
                    async for delta in generator.stream(request):
                        yield delta
            except TimeoutError as exc:
                raise OperationTimeout(
                    "stream exceeded configured timeout",
                    details={"timeout_s": config.timeout_s, "query_id": qid},
                ) from exc
            finally:
                set_correlation_id(None)
                self.services.observer.record("ask.stream.end", {"query_id": qid})

        return _gen()

    # -- timing helper ------------------------------------------------------

    async def _timed(self, stage: str, awaitable: Any, timings: dict[str, float]) -> Any:
        start = time.perf_counter()
        try:
            return await awaitable
        finally:
            timings[stage] = (time.perf_counter() - start) * 1000.0
