"""Config-driven service construction for the web example.

The workspace's :func:`rag_orchestrator.load_local_services` hardcodes the
embedding models (BAAI/bge-small + Qdrant/bm25), a heuristic reranker, no OCR,
and an LM-Studio-only generator — none of the settings page's model choices
would take effect. This module wires every pool's service bag straight from
the resolved settings instead:

- **embedding**: mock (offline), FastEmbed with the user's dense/sparse model
  ids and dimension, or a remote OpenAI-compatible ``/embeddings`` endpoint
  (:class:`rag_embedder.OpenAICompatibleEmbedder` — LM Studio, Ollama, vLLM,
  OpenAI, ...) with ``api_key_ref`` auth;
- **vector store**: local Qdrant path or server URL + ``api_key_ref``, sized
  to the embedder dimension;
- **rerank**: heuristic, local cross-encoder, or remote OpenAI-compatible
  rerank endpoint (model + api_key_ref);
- **OCR**: optional VLM escalation wired into the PDF parser's OCR fallback;
- **generation**: provider base URL + api_key_ref (model/temperature flow in
  via pipeline overrides set by :class:`~web_example.services.rag_factory`).

Heavy imports (fastembed, qdrant_client, PIL) stay lazy inside functions.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field
from rag_aio.config import RAGConfig
from rag_core.documents import PageBlock

__all__ = [
    "EmbedderExtras",
    "OcrSettings",
    "RerankSettings",
    "ResolvedSettings",
    "StorageExtras",
    "build_pool_services",
]

_DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class StorageExtras(BaseModel):
    """Web-only storage options that RAGConfig's schema does not carry."""

    qdrant_url: str | None = None
    qdrant_api_key_ref: str | None = None


class EmbedderExtras(BaseModel):
    """Web-only embedder options for the remote backend (RAGConfig has no fields)."""

    base_url: str | None = None
    api_key_ref: str | None = None


class RerankSettings(BaseModel):
    """Rerank stage model selection (web-only; not part of RAGConfig)."""

    strategy: Literal["heuristic", "cross_encoder", "remote"] = "heuristic"
    model: str | None = None
    base_url: str | None = None
    api_key_ref: str | None = None
    top_k: int | None = Field(default=None, ge=1)


class OcrSettings(BaseModel):
    """OCR escalation for scanned pages (web-only; not part of RAGConfig)."""

    engine: Literal["none", "vlm"] = "none"
    base_url: str = "http://localhost:1234/v1"
    model: str = "glm-ocr"
    api_key_ref: str | None = None


class ResolvedSettings(BaseModel):
    """Everything the service factory needs for one pool (global + overrides)."""

    config: RAGConfig
    storage_extras: StorageExtras = Field(default_factory=StorageExtras)
    embedder_extras: EmbedderExtras = Field(default_factory=EmbedderExtras)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


def build_pool_services(resolved: ResolvedSettings) -> Any:
    """Build the base service bag for *resolved* settings.

    The mock profile stays fully offline; the fastembed and remote (OpenAI-
    compatible ``/embeddings``) profiles wire real backends from the config.
    Rerank and OCR settings apply to all of them.
    """
    if resolved.config.embedder.backend == "mock":
        from rag_aio.facade import build_services

        services = build_services(resolved.config)
    else:
        services = _build_real_services(resolved)
    _apply_rerank(services, resolved.rerank)
    _apply_ocr(services, resolved.ocr)
    return services


def _build_embedders(embedder_cfg: Any, extras: EmbedderExtras) -> tuple[Any, Any]:
    """Dense + sparse embedders for the configured backend.

    ``remote`` uses :class:`rag_embedder.OpenAICompatibleEmbedder` (the same
    instance embeds documents at index time and queries at search time, as the
    package README requires). Sparse indexing keeps FastEmbed when available
    and falls back to the deterministic mock sparse embedder otherwise —
    remote providers do not offer a sparse endpoint.
    """
    from rag_embedder.embedding import MockSparseEmbedder

    if embedder_cfg.backend == "remote":
        from rag_embedder import OpenAICompatibleEmbedder

        dense: Any = OpenAICompatibleEmbedder(
            base_url=extras.base_url or "http://localhost:1234/v1",
            model=embedder_cfg.dense_model or "",
            api_key_ref=extras.api_key_ref or None,
            normalize=True,
            dim=embedder_cfg.dim,
        )
        try:
            from rag_embedder.embedding import FastEmbedSparse

            sparse: Any = FastEmbedSparse(
                model_name=embedder_cfg.sparse_model or _DEFAULT_SPARSE_MODEL
            )
        except ImportError:
            sparse = MockSparseEmbedder()
        return dense, sparse

    from rag_embedder.embedding import FastEmbedDense, FastEmbedSparse

    dense = FastEmbedDense(model_name=embedder_cfg.dense_model or _DEFAULT_DENSE_MODEL)
    sparse = FastEmbedSparse(model_name=embedder_cfg.sparse_model or _DEFAULT_SPARSE_MODEL)
    return dense, sparse


def _build_real_services(resolved: ResolvedSettings) -> Any:
    from rag_cache.backends.memory import MemoryCache
    from rag_cache.backends.sqlite import SQLiteCache
    from rag_cache.stats import InstrumentedCache
    from rag_context.builder import ContextBuilderImpl
    from rag_context.config import ContextConfig
    from rag_context.tokenizer import ContextTokenizer, WhitespaceCounter
    from rag_db_handler.config import (
        SQLStoreConfig,
        VectorStoreConfig,
        create_sql_store,
        create_vector_store,
    )
    from rag_doc_handler import default_pipeline
    from rag_embedder.chunking.base import ChunkerConfig
    from rag_embedder.chunking.registry import create_chunker
    from rag_embedder.pipeline import EmbeddingPipeline
    from rag_embedder.tokenization import SimpleTokenizer
    from rag_generation.generation import GenerationService
    from rag_generation.openai_compatible import OpenAICompatibleProvider
    from rag_orchestrator.services import NoOpObserver, OrchestratorServices
    from rag_query.config import QueryConfig
    from rag_query.engine import QueryEngine
    from rag_retrieval.config import RetrievalConfig
    from rag_retrieval.dense import DenseRetriever
    from rag_retrieval.fusion import ReciprocalRankFusion
    from rag_retrieval.hybrid import HybridRetriever

    config = resolved.config
    embedder_cfg = config.embedder
    dense, sparse = _build_embedders(embedder_cfg, resolved.embedder_extras)

    vector_cfg = VectorStoreConfig(
        mode="local",
        path=config.storage.qdrant_path,
        vector_size=embedder_cfg.dim,
        collection=_collection_for(embedder_cfg),
    )
    if resolved.storage_extras.qdrant_url:
        vector_cfg = VectorStoreConfig(
            mode="server",
            url=resolved.storage_extras.qdrant_url,
            api_key_ref=resolved.storage_extras.qdrant_api_key_ref or None,
            vector_size=embedder_cfg.dim,
            collection=_collection_for(embedder_cfg),
        )
    store = create_vector_store(vector_cfg)
    doc_store = create_sql_store(SQLStoreConfig(url=config.storage.db_url))

    if config.storage.cache_backend == "sqlite":
        cache = InstrumentedCache(SQLiteCache(path=config.storage.cache_path or "./data/cache"))
    else:
        cache = InstrumentedCache(MemoryCache(max_items=4096))

    chunker = create_chunker(ChunkerConfig(strategy="recursive"), SimpleTokenizer())
    pipeline = default_pipeline()
    embedding_pipeline = EmbeddingPipeline(
        chunker=chunker, embedder=dense, sparse_embedder=sparse, store=store
    )

    retrieval_overrides = dict(config.pipeline.retrieval.overrides)
    ret_cfg = RetrievalConfig(
        strategies=["dense"],
        top_k=int(retrieval_overrides.get("top_k", 10)),
        candidate_k=int(retrieval_overrides.get("candidate_k", 50)),
    )
    hybrid = HybridRetriever(
        retrievers=[DenseRetriever(store, dense, ret_cfg)],
        fusion=ReciprocalRankFusion(),
        config=ret_cfg,
    )

    context_overrides = dict(config.pipeline.context.overrides)
    ctx_builder = ContextBuilderImpl(
        ContextTokenizer(WhitespaceCounter()),
        ContextConfig(
            token_budget=int(context_overrides.get("token_budget", 2048)),
            reserve_for_answer=64,
        ),
    )
    query_engine = QueryEngine(QueryConfig())

    gen_cfg = config.generation
    api_key_ref = gen_cfg.api_key_ref or None

    def _api_key() -> str | None:
        return os.environ.get(api_key_ref) if api_key_ref else None

    provider = OpenAICompatibleProvider(base_url=gen_cfg.base_url, api_key_provider=_api_key)

    gen_service = GenerationService(providers={"configured": provider}, fallback_names=[])

    return OrchestratorServices(
        query_engine=query_engine,
        retrievers=[hybrid],
        hybrid_retriever=None,
        fusion=ReciprocalRankFusion(),
        reranker=None,  # applied by _apply_rerank
        context_builder=ctx_builder,
        generator=_StreamGenerator(gen_service),
        fallback_generators=[],
        model_router=None,
        provider_registry=None,
        loader=pipeline.loader,
        parser_registry=pipeline.registry,
        ingestion_pipeline=pipeline,
        chunker=chunker,
        embedder=dense,
        sparse_embedder=sparse,
        vector_store=store,
        document_store=doc_store,
        embedding_pipeline=embedding_pipeline,
        cache=_embedder_scoped_cache(cache, embedder_cfg),
        observer=NoOpObserver(),
    )


def _collection_for(embedder_cfg: Any) -> str:
    """A dimension-scoped Qdrant collection name.

    ``ensure_collection`` cannot resize an existing collection, so switching to
    an embedder with a different dimension must target a *different* collection
    — otherwise every query dies with a numpy "shapes not aligned" error. Old
    collections are left intact; documents re-ingested after the switch land in
    the new one.
    """
    return f"rag_chunks_d{embedder_cfg.dim}"


def _embedder_scoped_cache(cache: Any, embedder_cfg: Any) -> Any:
    """Namespace the stage caches by embedder identity.

    The orchestrator's ingestion cache is keyed by *source*: without namespacing,
    re-ingesting a document after switching embedding models would return the
    cached document and skip re-embedding entirely, leaving the new collection
    empty. Keying by backend + model + dim makes every cache entry valid only
    for the embedder that produced it.
    """
    from rag_cache import NamespacedCache

    namespace = (
        f"embedder:{embedder_cfg.backend}:{embedder_cfg.dense_model or ''}:{embedder_cfg.dim}"
    )
    return NamespacedCache(cache, namespace=namespace)


class _StreamGenerator:
    """Generator adapter over a GenerationService (streaming via provider)."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def generate(self, request: Any) -> Any:
        return await self._service.generate(request)

    def stream(self, request: Any) -> Any:
        providers: dict[str, Any] = getattr(self._service, "providers", {}) or {}
        provider = next(iter(providers.values()), None)
        if provider is None:
            msg = "no generation provider configured"
            raise RuntimeError(msg)
        return provider.stream(request)


# -- rerank ------------------------------------------------------------------


def _apply_rerank(services: Any, settings: RerankSettings) -> None:
    """Replace the service bag's reranker per the configured strategy."""
    from rag_rerank.config import RerankConfig
    from rag_rerank.local import CrossEncoderReranker, HeuristicReranker
    from rag_rerank.pipeline import RerankPipeline
    from rag_rerank.remote import RemoteReranker

    top_k = settings.top_k
    rerank_cfg = RerankConfig(top_k=top_k, max_candidates=50)
    if settings.strategy == "remote":
        base_url = settings.base_url or "http://localhost:1234/v1"
        model = settings.model or "rerank"
        inner: Any = RemoteReranker(
            base_url=base_url,
            model=model,
            api_key_ref=settings.api_key_ref,
            config=rerank_cfg,
        )
    elif settings.strategy == "cross_encoder":
        inner = CrossEncoderReranker(
            model_name=settings.model or "cross-encoder/ms-marco-MiniLM-L-6-v2", config=rerank_cfg
        )
    else:
        inner = HeuristicReranker(rerank_cfg)
    services.reranker = RerankPipeline(inner, rerank_cfg)


# -- OCR -----------------------------------------------------------------


def _apply_ocr(services: Any, settings: OcrSettings) -> None:
    """Wire VLM OCR escalation into the PDF parser when configured."""
    if settings.engine != "vlm":
        return
    registry = _ocr_registry(settings)
    services.parser_registry = registry
    if services.ingestion_pipeline is not None:
        from rag_doc_handler.pipeline import IngestionPipeline

        services.ingestion_pipeline = IngestionPipeline(
            registry=registry, loader=services.ingestion_pipeline.loader
        )


def _ocr_registry(settings: OcrSettings) -> Any:
    """Default parser registry with an OCR-escalating PDF parser."""
    from rag_doc_handler import default_registry
    from rag_doc_handler.parsers.base import ParserRegistry
    from rag_doc_handler.parsers.pdf import PDFParser

    base = default_registry()
    registry = ParserRegistry()
    for ext in (".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".xlsx", ".pptx", ".eml"):
        parser = base.get_or_none(ext)
        if parser is not None:
            registry.register(parser)
    registry.register(PDFParser(ocr_fallback=_VlmOcrFallback(settings)))
    return registry


class _VlmOcrFallback:
    """PDF OCR fallback: mechanical engines first, VLM escalation on low confidence."""

    def __init__(self, settings: OcrSettings) -> None:
        from rag_ocr.routing import OCRRouter, OCRRouterConfig
        from rag_ocr.semantic import VLMSemanticExtractor

        engines: list[Any] = []
        try:
            from rag_ocr.engines import RapidOCREngine

            engines.append(RapidOCREngine())
        except Exception:
            # rapidocr (optional extra) is not installed — escalate straight to VLM.
            from rag_ocr.engines import NullOCREngine

            engines.append(NullOCREngine())
        semantic = VLMSemanticExtractor(
            base_url=settings.base_url,
            model=settings.model,
            api_key_ref=settings.api_key_ref,
        )
        self._router = OCRRouter(engines=engines, semantic=semantic, config=OCRRouterConfig())

    async def __call__(self, document_id: str, page_number: int, image: bytes) -> list[PageBlock]:
        from rag_ocr.pipeline import regions_to_blocks

        page_result = await self._router.process_page(document_id, page_number, image)
        page_id = f"{document_id}:p{page_number}"
        return regions_to_blocks(page_result, page_id)
