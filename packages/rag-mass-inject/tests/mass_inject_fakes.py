"""Shared offline fakes for rag-mass-inject tests.

Build an :class:`~rag_orchestrator.services.OrchestratorServices` bag wired
entirely with deterministic, download-free components so every test path can
run without network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_core import (
    GenerationRequest,
    GenerationResult,
    Usage,
)
from rag_orchestrator.services import NoOpObserver, OrchestratorServices


class _StubGenerator:
    """Deterministic, offline :class:`~rag_core.protocols.Generator` stub.

    Implements both ``generate`` and ``stream`` so it satisfies the full
    :class:`Generator` protocol expected by :class:`OrchestratorServices`.
    """

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

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        self.requests.append(request)

        async def _gen() -> AsyncIterator[str]:
            for word in self.answer.split(" "):
                yield word + " "

        return _gen()


def make_services(tmp_path: Any) -> OrchestratorServices:
    """Build an offline services bag for mass-inject tests.

    All components are deterministic and require no network or heavy models:
    ``MockEmbedder(dim=8)``, ``MockSparseEmbedder``,
    ``InMemoryVectorStore(vector_size=8)``, ``MemoryCache``, and the default
    ingestion / embedding pipelines.
    """
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

    store = InMemoryVectorStore(vector_size=8)
    chunker = create_chunker(ChunkerConfig(strategy="recursive"), SimpleTokenizer())
    embedder = MockEmbedder(dim=8)
    sparse = MockSparseEmbedder()
    cache = InstrumentedCache(MemoryCache(max_items=128))
    ingestion_pipeline = default_pipeline()

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
    registry = ProviderRegistry(ProvidersConfig.local_default())
    generator = _StubGenerator()

    return OrchestratorServices(
        query_engine=query_engine,
        retrievers=[hybrid],  # type: ignore[list-item]
        fusion=fusion,
        reranker=reranker,
        context_builder=ctx_builder,
        generator=generator,  # type: ignore[arg-type]
        fallback_generators=[],
        model_router=ModelRouter(registry),
        provider_registry=registry,
        loader=ingestion_pipeline.loader,
        parser_registry=ingestion_pipeline.registry,
        ingestion_pipeline=ingestion_pipeline,
        chunker=chunker,
        embedder=embedder,
        sparse_embedder=sparse,
        vector_store=store,
        embedding_pipeline=EmbeddingPipeline(
            chunker=chunker, embedder=embedder, sparse_embedder=sparse, store=store
        ),
        cache=cache,
        observer=NoOpObserver(),
    )
