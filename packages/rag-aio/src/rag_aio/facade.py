"""High-level RAG facade: config-driven service composition and convenience API.

:class:`RAG` is the single entry point for most users.  It wires together the
contract-bound components from every ``rag-*`` package into an
:class:`~rag_orchestrator.Orchestrator` and exposes a small, opinionated surface
(``ingest``, ``ask``, ``app``, async context manager).

Heavy backends (``fastembed``, ``qdrant-client``, ``rag_doc_handler`` parsers,
``numpy``, …) are imported **lazily inside** :func:`_build_mock_services` so that
``import rag_aio`` stays dependency-light.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from rag_core.documents import Document
from rag_core.generation import GenerationRequest, GenerationResult, Usage
from rag_core.pipeline_config import PipelineConfig

from rag_aio.config import RAGConfig

if TYPE_CHECKING:
    from fastapi import FastAPI
    from rag_orchestrator import AskResult, OrchestratorServices

__all__ = ["RAG", "build_services"]


# --------------------------------------------------------------------------- #
# Lazy orchestrator access
# --------------------------------------------------------------------------- #


def _require_orchestrator() -> ModuleType:
    """Import ``rag-orchestrator`` on first use with an actionable error.

    The base ``rag-aio`` install is dependency-light (``rag-core`` + CLI/service
    deps); the pipeline machinery lives behind the ``rag-orchestrator`` extra,
    and the full stack behind ``all``.
    """
    try:
        return importlib.import_module("rag_orchestrator")
    except ImportError as exc:
        msg = (
            "rag-aio needs the 'rag-orchestrator' package for this feature. "
            'Install it with: pip install "rag-aio[all]" '
            'or pip install "rag-aio[rag-orchestrator]".'
        )
        raise ImportError(msg) from exc


# --------------------------------------------------------------------------- #
# Offline / stub generator
# --------------------------------------------------------------------------- #


class _StubGenerator:
    """Deterministic generator with **no HTTP backend**.

    Used by the mock profile so that ``RAG.from_config(RAGConfig.mock())`` works
    fully offline.  The answer is derived from the retrieved context that the
    orchestrator embeds in the user message; when no context is available a
    fixed fallback string is returned.
    """

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self._answer_from(request),
            model="stub",
            finish_reason="stop",
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        return self._stream(request)

    async def _stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        answer = self._answer_from(request)
        for word in answer.split(" "):
            yield word + " "

    @staticmethod
    def _answer_from(request: GenerationRequest) -> str:
        """Extract the retrieved context from the rendered prompt as the answer."""
        for msg in reversed(request.messages):
            if msg.role == "user":
                content = msg.content
                sep = "\n\nQuestion:"
                idx = content.rfind(sep)
                if idx >= 0:
                    context_text = content[:idx].strip()
                    if context_text:
                        return context_text
                return "Based on the provided context."
        return "Based on the provided context."


# --------------------------------------------------------------------------- #
# Service composition
# --------------------------------------------------------------------------- #


def build_services(config: RAGConfig) -> OrchestratorServices:
    """Wire :class:`RAGConfig` into an :class:`OrchestratorServices` bag.

    Selects the offline (mock) wiring when ``config.embedder.backend == "mock"``,
    otherwise delegates to :func:`rag_orchestrator.load_local_services`.
    """
    if config.embedder.backend == "mock":
        return _build_mock_services(config)

    orchestrator = _require_orchestrator()
    from rag_observe import timed

    with timed("build_services", {"backend": "fastembed"}):
        return orchestrator.load_local_services(
            qdrant_path=config.storage.qdrant_path,
            db_url=config.storage.db_url,
            lm_studio_url=config.generation.base_url,
        )


def _build_mock_services(config: RAGConfig) -> OrchestratorServices:
    """Build a fully-offline :class:`OrchestratorServices` bag.

    Only pure-Python / dependency-light implementations are used:
    :class:`MockEmbedder`, :class:`MockSparseEmbedder`,
    :class:`InMemoryVectorStore`, :class:`MemoryCache`,
    :class:`HeuristicReranker`, and :class:`_StubGenerator`.
    """
    # All heavy imports live inside this function so ``import rag_aio`` stays light.
    orchestrator = _require_orchestrator()
    OrchestratorServices = orchestrator.OrchestratorServices
    NoOpObserver = orchestrator.NoOpObserver

    from rag_cache.backends.memory import MemoryCache
    from rag_cache.stats import InstrumentedCache
    from rag_context.builder import ContextBuilderImpl
    from rag_context.config import ContextConfig
    from rag_context.tokenizer import ContextTokenizer, WhitespaceCounter
    from rag_db_handler.memory_store import InMemoryVectorStore
    from rag_doc_handler import default_pipeline
    from rag_embedder.chunking.base import ChunkerConfig
    from rag_embedder.chunking.registry import create_chunker
    from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
    from rag_embedder.pipeline import EmbeddingPipeline
    from rag_embedder.tokenization import SimpleTokenizer
    from rag_llm_provider.config import ProvidersConfig
    from rag_llm_provider.registry import ProviderRegistry
    from rag_llm_provider.routing import ModelRouter
    from rag_query.config import QueryConfig
    from rag_query.engine import QueryEngine
    from rag_rerank.config import RerankConfig
    from rag_rerank.local import HeuristicReranker
    from rag_rerank.pipeline import RerankPipeline
    from rag_retrieval.config import RetrievalConfig
    from rag_retrieval.dense import DenseRetriever
    from rag_retrieval.fusion import ReciprocalRankFusion
    from rag_retrieval.hybrid import HybridRetriever

    dim = config.embedder.dim
    store = InMemoryVectorStore(vector_size=dim)
    embedder = MockEmbedder(dim=dim)
    sparse_embedder = MockSparseEmbedder()

    chunker = create_chunker(ChunkerConfig(strategy="recursive"), SimpleTokenizer())
    ingestion = default_pipeline()
    embedding_pipeline = EmbeddingPipeline(
        chunker=chunker,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        store=store,
    )

    ret_cfg = RetrievalConfig(strategies=["dense"], top_k=10, candidate_k=50)
    hybrid = HybridRetriever(
        retrievers=[DenseRetriever(store, embedder, ret_cfg)],
        fusion=ReciprocalRankFusion(),
        config=ret_cfg,
    )
    fusion = ReciprocalRankFusion()

    reranker = RerankPipeline(
        HeuristicReranker(RerankConfig(max_candidates=50, top_k=10)),
        RerankConfig(),
    )
    ctx_builder = ContextBuilderImpl(
        ContextTokenizer(WhitespaceCounter()),
        ContextConfig(token_budget=2048, reserve_for_answer=64),
    )
    query_engine = QueryEngine(QueryConfig())

    providers_config = ProvidersConfig.local_default()
    registry = ProviderRegistry(providers_config)

    return OrchestratorServices(
        query_engine=query_engine,
        retrievers=[hybrid],
        hybrid_retriever=None,
        fusion=fusion,
        reranker=reranker,
        context_builder=ctx_builder,
        generator=_StubGenerator(),
        fallback_generators=[],
        model_router=ModelRouter(registry),
        provider_registry=registry,
        loader=ingestion.loader,
        parser_registry=ingestion.registry,
        ingestion_pipeline=ingestion,
        chunker=chunker,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        vector_store=store,
        embedding_pipeline=embedding_pipeline,
        cache=InstrumentedCache(MemoryCache(max_items=4096)),
        observer=NoOpObserver(),
    )


# --------------------------------------------------------------------------- #
# High-level facade
# --------------------------------------------------------------------------- #


class RAG:
    """Single-shot facade over :class:`~rag_orchestrator.Orchestrator`.

    Typical usage::

        async with RAG.from_config(RAGConfig.mock()) as rag:
            await rag.ingest("doc.md")
            result = await rag.ask("What is this about?")
    """

    def __init__(
        self,
        services: OrchestratorServices,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        orchestrator = _require_orchestrator()
        self.services = services
        self._pipeline_config = pipeline_config
        self._orchestrator = orchestrator.Orchestrator(services, pipeline_config)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(cls, config: RAGConfig | str | Path) -> RAG:
        """Build services from *config* and return a ready :class:`RAG`.

        *config* is a :class:`RAGConfig` or a path to a TOML config file, so the
        README quickstart ``RAG.from_config("config.toml")`` works as written.
        """
        if not isinstance(config, RAGConfig):
            config = RAGConfig.from_file(config)
        services = build_services(config)
        return cls(services, config.pipeline)

    # -- data ---------------------------------------------------------------

    async def ingest(self, path: str | Path) -> Document:
        """Ingest a file through the orchestrator's stage-cached pipeline."""
        orchestrator = _require_orchestrator()
        cfg = self._pipeline_config or PipelineConfig.local_default()
        document: Document = await orchestrator.ingest(self.services, str(path), config=cfg)
        return document

    async def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> list[Document]:
        """Ingest every supported file under *directory*."""
        orchestrator = _require_orchestrator()
        cfg = self._pipeline_config or PipelineConfig.local_default()
        documents: list[Document] = await orchestrator.ingest_directory(
            self.services, str(directory), recursive, config=cfg
        )
        return documents

    # -- query --------------------------------------------------------------

    async def ask(
        self,
        query: str,
        *,
        stream: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> AskResult | AsyncIterator[str]:
        """Run the configured RAG pipeline for *query*."""
        result: AskResult | AsyncIterator[str] = await self._orchestrator.ask(
            query, stream=stream, overrides=overrides
        )
        return result

    # -- web app -----------------------------------------------------------

    @property
    def app(self) -> FastAPI:
        """A FastAPI application backed by this RAG's services."""
        from rag_orchestrator.app.app import create_app

        return create_app(self.services, self._pipeline_config)

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> RAG:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        await self._close()

    async def _close(self) -> None:
        """Best-effort cleanup of closable backend resources."""
        for store in (self.services.vector_store, self.services.document_store):
            if store is None:
                continue
            close_fn = getattr(store, "close", None)
            if callable(close_fn):
                outcome = close_fn()
                if inspect.isawaitable(outcome):
                    await outcome
