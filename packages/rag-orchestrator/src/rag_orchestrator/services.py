"""Component wiring bag for the orchestrator.

:class:`OrchestratorServices` is an intentionally thin, typed container — *a bag*
of already-constructed components. It holds **no behavior** of its own: every
field is a component reference typed against the ``rag_core`` Protocol contracts
(or a lightweight concrete type imported only under ``TYPE_CHECKING``). This
keeps :mod:`rag_orchestrator` importable without pulling in heavy backends
(numpy/fastembed/qdrant/etc.) — those are imported lazily inside
:func:`load_local_services`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from rag_core.protocols import (
    Cache,
    Chunker,
    ContextBuilder,
    DocumentStore,
    Embedder,
    Generator,
    HybridRetriever,
    Observer,
    Reranker,
    Retriever,
    SparseEmbedder,
    VectorStore,
)

if TYPE_CHECKING:
    from rag_cache.backends.memory import MemoryCache
    from rag_doc_handler.loader import SourceLoader
    from rag_doc_handler.parsers.base import ParserRegistry
    from rag_doc_handler.pipeline import IngestionPipeline
    from rag_embedder.pipeline import EmbeddingPipeline
    from rag_llm_provider.registry import ProviderRegistry
    from rag_llm_provider.routing import ModelRouter
    from rag_query.engine import QueryEngine
    from rag_retrieval.fusion import FusionStrategy

__all__ = ["NoOpObserver", "OrchestratorServices", "load_local_services"]


class NoOpObserver:
    """A no-op :class:`rag_core.protocols.Observer` used as a safe default."""

    def record(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Discard the event (no-op)."""
        return None


class _StreamableProvider(Protocol):
    """Structural view of an LLM provider that also supports streaming.

    The ``rag_core.protocols.LLMProvider`` contract only declares ``complete``;
    concrete adapters additionally expose ``stream``. We declare the streaming
    shape locally so the adapter can forward streaming calls without reaching
    into provider internals.
    """

    async def complete(self, request: Any) -> Any: ...

    def stream(self, request: Any) -> AsyncIterator[str]: ...


class _ServiceGenerator:
    """Adapter turning a :class:`rag_generation.GenerationService` into a
    :class:`rag_core.protocols.Generator`.

    ``generate`` is delegated to the service verbatim, so provider fallback
    (router + fallback_names) is reused for free. ``stream`` resolves the
    effective provider and forwards to its native ``stream``.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    async def generate(self, request: Any) -> Any:
        return await self._service.generate(request)

    def stream(self, request: Any) -> AsyncIterator[str]:
        provider = self._resolve_provider(request)
        streamable = cast("_StreamableProvider", provider)
        return streamable.stream(request)

    def _resolve_provider(self, request: Any) -> Any:
        service = self._service
        providers: dict[str, Any] = getattr(service, "providers", {}) or {}
        name: str | None = None
        router = getattr(service, "router", None)
        if router is not None:
            try:
                name = router(request)
            except Exception:
                name = None
        if name is None or name not in providers:
            name = next(iter(providers), None)
        provider = providers.get(name) if name else None
        if provider is None:
            first = next(iter(providers.values()), None)
            if first is None:
                msg = "no provider available to stream from"
                raise RuntimeError(msg)
            return first
        return provider


@dataclass
class OrchestratorServices:
    """Typed bag of wired RAG components (no behavior).

    Every field has a default so test fixtures can construct a partial bag with
    only the components a given path exercises. The orchestrator validates the
    components it actually needs at construction time.
    """

    query_engine: QueryEngine | None = None
    retrievers: list[Retriever] = field(default_factory=list)
    fusion: FusionStrategy | None = None
    hybrid_retriever: HybridRetriever | None = None
    reranker: Reranker | None = None
    context_builder: ContextBuilder | None = None
    generator: Generator | None = None
    fallback_generators: list[Generator] = field(default_factory=list)
    model_router: ModelRouter | None = None
    provider_registry: ProviderRegistry | None = None
    loader: SourceLoader | None = None
    parser_registry: ParserRegistry | None = None
    ingestion_pipeline: IngestionPipeline | None = None
    chunker: Chunker | None = None
    embedder: Embedder | None = None
    sparse_embedder: SparseEmbedder | None = None
    vector_store: VectorStore | None = None
    document_store: DocumentStore | None = None
    embedding_pipeline: EmbeddingPipeline | None = None
    cache: Cache | None = None
    observer: Observer = field(default_factory=NoOpObserver)

    def ensure_retrievers(self) -> list[Retriever]:
        """Return the effective retriever list (hybrid first if present)."""
        if self.hybrid_retriever is not None:
            return [self.hybrid_retriever, *self.retrievers]
        return list(self.retrievers)


def _memory_cache() -> MemoryCache:
    from rag_cache.backends.memory import MemoryCache

    return MemoryCache(max_items=4096)


def load_local_services(
    *,
    qdrant_path: str = "./data/qdrant",
    db_url: str = "sqlite+aiosqlite:///./data/rag.db",
    lm_studio_url: str = "http://localhost:1234/v1",
) -> OrchestratorServices:
    """Wire the ``local_default`` profile against local backends.

    Heavy backends (qdrant_client, fastembed, bm25s, httpx-based providers) are
    imported *lazily inside this function* so ``import rag_orchestrator`` stays
    cheap and dependency-light. Callers must have the relevant extras installed
    (``qdrant-client``, ``fastembed``, ``rag-generation``, etc.).

    Backends:
      * vectors — on-disk Qdrant (local mode).
      * documents — async-sqlite document store.
      * cache — in-process memory cache.
      * embedder — FastEmbed dense + FastEmbed sparse (BM25-style).
      * chunking — recursive chunker.
      * retrieval — dense + sparse hybrid with RRF.
      * rerank — heuristic Jaccard.
      * context — token-budgeted builder (whitespace tokenizer).
      * generation — LM Studio (OpenAI-compatible) via GenerationService.
    """
    # Local imports keep the module graph light until this is called.
    from rag_cache.backends.memory import MemoryCache
    from rag_cache.stats import InstrumentedCache
    from rag_context.builder import ContextBuilderImpl
    from rag_context.config import ContextConfig
    from rag_context.tokenizer import ContextTokenizer, WhitespaceCounter
    from rag_db_handler.config import (
        KeyValueStoreConfig,
        SQLStoreConfig,
        VectorStoreConfig,
        create_kv_store,
        create_sql_store,
        create_vector_store,
    )
    from rag_doc_handler import default_pipeline
    from rag_embedder.chunking.base import ChunkerConfig
    from rag_embedder.chunking.registry import create_chunker
    from rag_embedder.embedding import FastEmbedDense, FastEmbedSparse
    from rag_embedder.pipeline import EmbeddingPipeline
    from rag_embedder.tokenization import SimpleTokenizer
    from rag_generation.generation import GenerationService
    from rag_generation.openai_compatible import OpenAICompatibleProvider
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

    dense = FastEmbedDense(model_name="BAAI/bge-small-en-v1.5")
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    store = create_vector_store(VectorStoreConfig(mode="local", path=qdrant_path))
    doc_store = create_sql_store(SQLStoreConfig(url=db_url))
    _kv = create_kv_store(KeyValueStoreConfig(backend="memory"))

    chunker = create_chunker(ChunkerConfig(strategy="recursive"), SimpleTokenizer())
    embedding_pipeline = EmbeddingPipeline(
        chunker=chunker, embedder=dense, sparse_embedder=sparse, store=store
    )

    ret_cfg = RetrievalConfig(strategies=["dense"], top_k=10, candidate_k=50)
    # Dense-only hybrid wiring: QdrantVectorStore does not implement the
    # SparseSearchStore protocol (search_sparse), so sparse retrieval would be
    # a lie at the type level. BM25-in-process is the documented extension
    # point once a corpus provider is wired.
    hybrid = HybridRetriever(
        retrievers=[
            DenseRetriever(store, dense, ret_cfg),
        ],
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

    provider = OpenAICompatibleProvider(base_url=lm_studio_url)
    gen_service = GenerationService(providers={"lm_studio": provider}, fallback_names=[])
    generator = _ServiceGenerator(gen_service)

    pipeline = default_pipeline()
    providers_config = ProvidersConfig.local_default()
    registry = ProviderRegistry(providers_config)

    return OrchestratorServices(
        query_engine=query_engine,
        retrievers=[hybrid],
        hybrid_retriever=None,
        fusion=fusion,
        reranker=reranker,
        context_builder=ctx_builder,
        generator=generator,
        fallback_generators=[],
        model_router=ModelRouter(registry),
        provider_registry=registry,
        loader=pipeline.loader,
        parser_registry=pipeline.registry,
        ingestion_pipeline=pipeline,
        chunker=chunker,
        embedder=dense,
        sparse_embedder=sparse,
        vector_store=store,
        document_store=doc_store,
        embedding_pipeline=embedding_pipeline,
        cache=InstrumentedCache(MemoryCache(max_items=4096)),
        observer=NoOpObserver(),
    )


__all__ = ["NoOpObserver", "OrchestratorServices", "load_local_services"]
