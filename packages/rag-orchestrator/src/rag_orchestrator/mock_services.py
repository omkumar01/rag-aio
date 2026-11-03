"""Fully-offline mock service wiring for the orchestrator.

:class:`build_mock_services` assembles an :class:`~rag_orchestrator.OrchestratorServices`
bag in which every component is a pure-Python, dependency-light implementation:
:class:`~rag_embedder.embedding.MockEmbedder`,
:class:`~rag_embedder.embedding.MockSparseEmbedder`,
:class:`~rag_db_handler.memory_store.InMemoryVectorStore`,
:class:`~rag_cache.backends.memory.MemoryCache`,
:class:`~rag_rerank.local.HeuristicReranker`, and :class:`StubGenerator`.

This is the same wiring the ``rag-aio`` facade selects for
``RAGConfig.mock()``; it lives here so consumers depending on
``rag-aio-orchestrator`` directly (without the facade) can build the offline
stack with a single call:

    from rag_orchestrator import build_mock_services

    services = build_mock_services()
    orchestrator = Orchestrator(services)

Heavy sibling packages are imported **lazily inside** :func:`build_mock_services`
so ``import rag_orchestrator`` stays dependency-light.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rag_core.generation import GenerationRequest, GenerationResult, Usage

if TYPE_CHECKING:
    from .services import OrchestratorServices

__all__ = ["StubGenerator", "build_mock_services"]

DEFAULT_VECTOR_SIZE = 384
"""Default embedding dimensionality (matches ``RAGConfig.embedder.dim``)."""

DEFAULT_CACHE_MAX_ITEMS = 4096
"""Default cache capacity (matches the facade's historic mock wiring)."""


class StubGenerator:
    """Deterministic generator with **no HTTP backend**.

    Used by the mock profile so that offline RAG stacks work fully offline.
    The answer is derived from the retrieved context that the orchestrator
    embeds in the user message; when no context is available a fixed fallback
    string is returned.
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


def build_mock_services(
    *,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    cache_max_items: int = DEFAULT_CACHE_MAX_ITEMS,
) -> OrchestratorServices:
    """Build a fully-offline :class:`~rag_orchestrator.OrchestratorServices` bag.

    Args:
        vector_size: Embedding dimensionality for the mock embedder and the
            in-memory vector store (facade maps ``RAGConfig.embedder.dim``).
        cache_max_items: Capacity of the instrumented memory cache.

    Only pure-Python / dependency-light implementations are used:
    :class:`~rag_embedder.embedding.MockEmbedder`,
    :class:`~rag_embedder.embedding.MockSparseEmbedder`,
    :class:`~rag_db_handler.memory_store.InMemoryVectorStore`,
    :class:`~rag_cache.backends.memory.MemoryCache`,
    :class:`~rag_rerank.local.HeuristicReranker`, and :class:`StubGenerator`.
    The bag intentionally leaves ``document_store`` unset.
    """
    # All heavy imports live inside this function so ``import rag_orchestrator`` stays light.
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

    from .services import NoOpObserver, OrchestratorServices

    dim = vector_size
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
        generator=StubGenerator(),
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
        cache=InstrumentedCache(MemoryCache(max_items=cache_max_items)),
        observer=NoOpObserver(),
    )
