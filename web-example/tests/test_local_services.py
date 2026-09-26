"""Unit tests for the config-driven service factory (local_services)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rag_aio.config import EmbedderConfig, RAGConfig
from rag_rerank.local import CrossEncoderReranker, HeuristicReranker
from rag_rerank.remote import RemoteReranker

from web_example.services.local_services import (
    EmbedderExtras,
    OcrSettings,
    RerankSettings,
    ResolvedSettings,
    StorageExtras,
    build_pool_services,
)


def _resolved(**kwargs: Any) -> ResolvedSettings:
    return ResolvedSettings(config=RAGConfig.mock(), **kwargs)


def test_mock_profile_default_reranker() -> None:
    services = build_pool_services(_resolved())
    assert services.reranker is not None
    assert isinstance(services.reranker._reranker, HeuristicReranker)
    assert services.vector_store is not None


def test_rerank_remote_strategy_wires_remote_reranker() -> None:
    services = build_pool_services(
        _resolved(
            rerank=RerankSettings(
                strategy="remote",
                base_url="http://localhost:1234/v1",
                model="bge-reranker-base",
                api_key_ref="RERANK_API_KEY",
                top_k=5,
            )
        )
    )
    inner = services.reranker._reranker
    assert isinstance(inner, RemoteReranker)
    assert services.reranker._config.top_k == 5


def test_rerank_cross_encoder_strategy() -> None:
    services = build_pool_services(
        _resolved(
            rerank=RerankSettings(strategy="cross_encoder", model="cross-encoder/custom-model")
        )
    )
    inner = services.reranker._reranker
    assert isinstance(inner, CrossEncoderReranker)
    assert inner._model_name == "cross-encoder/custom-model"


def test_ocr_vlm_registry_uses_escalating_pdf_parser() -> None:
    services = build_pool_services(
        _resolved(
            ocr=OcrSettings(engine="vlm", base_url="http://localhost:1234/v1", model="my-ocr")
        )
    )
    parser = services.parser_registry.get_or_none(".pdf")
    assert parser is not None and parser.__class__.__name__ == "PDFParser"
    assert parser._ocr is not None
    # Other parsers are preserved in the OCR registry.
    assert services.parser_registry.get_or_none(".txt") is not None


def test_ocr_none_keeps_plain_pdf_parser() -> None:
    services = build_pool_services(_resolved(ocr=OcrSettings(engine="none")))
    parser = services.parser_registry.get_or_none(".pdf")
    assert parser is not None
    assert parser._ocr is None


def test_real_profile_uses_configured_models(tmp_path: Path) -> None:
    pytest.importorskip("fastembed")
    config = RAGConfig(
        embedder=EmbedderConfig(backend="fastembed", dense_model="BAAI/bge-small-en-v1.5", dim=384),
    )
    config.storage = config.storage.model_copy(
        update={
            "qdrant_path": str(tmp_path / "qdrant"),
            "db_url": f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
        }
    )
    resolved = ResolvedSettings(
        config=config,
        storage_extras=StorageExtras(),
        rerank=RerankSettings(top_k=4),
    )
    services = build_pool_services(resolved)
    assert services.embedder is not None
    assert services.embedder._model_name == "BAAI/bge-small-en-v1.5"
    assert services.sparse_embedder is not None
    assert services.vector_store is not None
    assert services.vector_store.collection == "rag_chunks_d384"
    assert services.reranker._config.top_k == 4
    # Retrieval config honors pipeline overrides.
    hybrid = services.retrievers[0]
    assert hybrid._config.top_k == 10


def test_collection_name_follows_embedder_dimension(tmp_path: Path) -> None:
    """Switching to a 1024-dim embedder targets a different collection — an
    existing 384-dim collection can never be queried with 1024-dim vectors."""
    from rag_embedder import OpenAICompatibleEmbedder

    config = RAGConfig(
        embedder=EmbedderConfig(backend="remote", dense_model="bge-m3", dim=1024),
    )
    config.storage = config.storage.model_copy(
        update={
            "qdrant_path": str(tmp_path / "qdrant"),
            "db_url": f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
        }
    )
    services = build_pool_services(
        ResolvedSettings(
            config=config,
            embedder_extras=EmbedderExtras(base_url="http://localhost:1234/v1"),
        )
    )
    assert isinstance(services.embedder, OpenAICompatibleEmbedder)
    assert services.vector_store.collection == "rag_chunks_d1024"


def test_cache_is_namespaced_per_embedder(tmp_path: Path) -> None:
    """The ingestion stage cache is keyed by source; without a per-embedder
    namespace, re-ingesting after a model switch would return the cached
    document and skip re-embedding entirely."""
    from rag_cache import NamespacedCache

    config = RAGConfig(
        embedder=EmbedderConfig(backend="fastembed", dense_model="BAAI/bge-small-en-v1.5", dim=384),
    )
    config.storage = config.storage.model_copy(
        update={
            "qdrant_path": str(tmp_path / "qdrant"),
            "db_url": f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
        }
    )
    services = build_pool_services(ResolvedSettings(config=config, storage_extras=StorageExtras()))
    assert isinstance(services.cache, NamespacedCache)
    assert "384" in services.cache._namespace
    assert "BAAI/bge-small-en-v1.5" in services.cache._namespace


def test_remote_profile_builds_openai_compatible_embedder(tmp_path: Path) -> None:
    from rag_embedder import OpenAICompatibleEmbedder

    config = RAGConfig(
        embedder=EmbedderConfig(
            backend="remote",
            dense_model="text-embedding-nomic-embed-text-v1.5",
            dim=768,
        ),
    )
    config.storage = config.storage.model_copy(
        update={
            "qdrant_path": str(tmp_path / "qdrant"),
            "db_url": f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
        }
    )
    resolved = ResolvedSettings(
        config=config,
        embedder_extras=EmbedderExtras(
            base_url="http://localhost:1234/v1", api_key_ref="EMBEDDING_API_KEY"
        ),
    )
    services = build_pool_services(resolved)
    embedder = services.embedder
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder._model == "text-embedding-nomic-embed-text-v1.5"
    assert embedder._base_url == "http://localhost:1234/v1"
    assert embedder._api_key_ref == "EMBEDDING_API_KEY"
    assert embedder._dim == 768
    assert embedder.normalize is True
    # The same embedder instance serves ingestion and query-time retrieval.
    dense_retriever = services.retrievers[0]._retrievers[0]
    assert dense_retriever._embedder is embedder
    assert services.vector_store is not None
