"""Tests for stage-cached ingestion (offline: mock embedder + memory vector store)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import fakes  # noqa: F401  (conftest puts tests dir on sys.path)
import pytest
from rag_core.errors import ConfigError, IngestionError
from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
from rag_embedder.pipeline import EmbeddingPipeline
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.ingestion import ingest, ingest_directory
from rag_orchestrator.services import OrchestratorServices


def _write_md(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def make_ingest_services(tmp_path) -> OrchestratorServices:
    """Services bag wired for fully-offline ingestion."""
    from rag_cache.backends.memory import MemoryCache
    from rag_db_handler.memory_store import InMemoryVectorStore
    from rag_doc_handler import default_pipeline
    from rag_embedder.chunking.base import ChunkerConfig
    from rag_embedder.chunking.registry import create_chunker
    from rag_embedder.tokenization import SimpleTokenizer

    chunker = create_chunker(ChunkerConfig(strategy="fixed", chunk_size=64), SimpleTokenizer())
    embedder = MockEmbedder(dim=8)
    store = InMemoryVectorStore()
    pipeline = default_pipeline()
    return OrchestratorServices(
        ingestion_pipeline=pipeline,
        embedding_pipeline=EmbeddingPipeline(
            chunker=chunker, embedder=embedder, sparse_embedder=MockSparseEmbedder(), store=store
        ),
        vector_store=store,
        cache=MemoryCache(max_items=128),
    )


def test_ingest_requires_wired_components() -> None:
    with pytest.raises(ConfigError):
        import asyncio

        asyncio.run(ingest(OrchestratorServices(), "x.txt"))


async def test_ingest_missing_embedding_pipeline_raises(tmp_path) -> None:
    source = _write_md(tmp_path, "doc.md", "content " * 50)
    services = make_ingest_services(tmp_path)
    services.embedding_pipeline = None  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="embedding_pipeline is required"):
        await ingest(services, source, config=PipelineConfig())


async def test_ingest_missing_vector_store_raises(tmp_path) -> None:
    source = _write_md(tmp_path, "doc.md", "content " * 50)
    services = make_ingest_services(tmp_path)
    services.vector_store = None  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="vector_store is required"):
        await ingest(services, source, config=PipelineConfig())


async def test_ingest_indexing_failure_raises_ingestion_error(tmp_path) -> None:
    """When the embedding pipeline fails, ingest raises IngestionError."""

    class FailingPipeline:
        async def process(self, document: object, store: object) -> None:
            raise RuntimeError("embedding backend crashed")

    source = _write_md(tmp_path, "doc.md", "content " * 50)
    services = make_ingest_services(tmp_path)
    services.embedding_pipeline = FailingPipeline()  # type: ignore[assignment]
    with pytest.raises(IngestionError, match="indexing failed"):
        await ingest(services, source, config=PipelineConfig())


async def test_ingest_directory_not_found_raises(tmp_path) -> None:
    services = make_ingest_services(tmp_path)
    with pytest.raises(IngestionError, match="not a directory"):
        await ingest_directory(services, tmp_path / "nonexistent", config=PipelineConfig())


async def test_ingest_single_file(tmp_path) -> None:
    source = _write_md(tmp_path, "auth.md", "# Auth\n\n" + "Requirements. " * 40)
    services = make_ingest_services(tmp_path)
    doc = await ingest(services, source, config=PipelineConfig())
    assert doc.content_hash
    assert "Requirements" in doc.text


async def test_ingest_cached_second_call_returns_same_document(tmp_path) -> None:
    source = _write_md(tmp_path, "doc.md", "content " * 50)
    services = make_ingest_services(tmp_path)
    first = await ingest(services, source, config=PipelineConfig())
    second = await ingest(services, source, config=PipelineConfig())
    assert first.id == second.id
    assert first.content_hash == second.content_hash


async def test_ingest_unsupported_format_raises(tmp_path) -> None:
    source = _write_md(tmp_path, "data.binary", "\x00\x01\x02not-a-doc")
    services = make_ingest_services(tmp_path)
    with pytest.raises(IngestionError):
        await ingest(services, source, config=PipelineConfig())


async def test_ingest_non_rag_error_wrapped_as_ingestion_error(tmp_path) -> None:
    """A generic exception during load/parse/dedup is wrapped as IngestionError."""

    class BrokenPipeline:
        async def ingest(self, source: str, dedup: object | None = None):
            raise RuntimeError("parser crashed")

    source = _write_md(tmp_path, "doc.md", "content " * 50)
    services = make_ingest_services(tmp_path)
    services.ingestion_pipeline = BrokenPipeline()  # type: ignore[assignment]
    with pytest.raises(IngestionError, match="failed to ingest"):
        await ingest(services, source, config=PipelineConfig())


async def test_ingest_directory_skips_unsupported(tmp_path) -> None:
    _write_md(tmp_path, "a.md", "alpha content " * 20)
    _write_md(tmp_path, "b.md", "beta content " * 20)
    _write_md(tmp_path, "skip.binary", "\x00\x01\x02")
    services = make_ingest_services(tmp_path)
    docs = await ingest_directory(services, tmp_path, config=PipelineConfig())
    assert len(docs) == 2


async def test_ingest_modified_file_not_served_from_stale_cache(tmp_path) -> None:
    """Re-ingesting an edited file must reflect the new content, not the cache."""
    path = tmp_path / "doc.md"
    path.write_text("old content " * 60, encoding="utf-8")
    services = make_ingest_services(tmp_path)
    first = await ingest(services, str(path), config=PipelineConfig())

    # Rewrite with different (shorter) content and force a distinct mtime so the
    # fingerprint (mtime_ns:size) cannot accidentally match.
    path.write_text("brand new content", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_mtime_ns + 10**9, stat.st_mtime_ns + 10**9))

    second = await ingest(services, str(path), config=PipelineConfig())
    assert second.content_hash != first.content_hash
    assert "brand new content" in second.text


async def test_ingest_non_file_source_hits_cache_on_second_call(tmp_path) -> None:
    """Non-file sources keep the old URI-only keying: second call is a cache hit."""
    from rag_core.documents import Document

    class OnceOnlyPipeline:
        """Succeeds once; any second call means the cache was bypassed."""

        def __init__(self) -> None:
            self.calls = 0

        async def ingest(self, source: str, dedup: object | None = None):
            self.calls += 1
            assert self.calls == 1, "second ingest must be served from cache"
            return Document(source_uri=source, text="memorized"), True

    services = make_ingest_services(tmp_path)
    services.ingestion_pipeline = OnceOnlyPipeline()  # type: ignore[assignment]
    first = await ingest(services, "mem://doc-1", config=PipelineConfig())
    second = await ingest(services, "mem://doc-1", config=PipelineConfig())
    assert first.id == second.id
    assert first.content_hash == second.content_hash


def test_ingestion_cache_key_fingerprint_follows_file_mtime() -> None:
    """Local files are keyed by (mtime_ns, size); other sources are not."""
    from rag_orchestrator.config import PipelineConfig
    from rag_orchestrator.ingestion import _ingestion_cache_key

    fd, name = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("fingerprint probe")
    path = Path(name)
    try:
        key_before = _ingestion_cache_key(str(path), PipelineConfig())
        stat = path.stat()
        os.utime(path, ns=(stat.st_mtime_ns + 10**9, stat.st_mtime_ns + 10**9))
        key_after = _ingestion_cache_key(str(path), PipelineConfig())
        assert key_before != key_after

        # Unchanged file: stable key across calls.
        assert _ingestion_cache_key(str(path), PipelineConfig()) == key_after

        # Non-file source: key is deterministic and independent of any mtime.
        mem_key = _ingestion_cache_key("mem://doc-1", PipelineConfig())
        assert mem_key == _ingestion_cache_key("mem://doc-1", PipelineConfig())
    finally:
        path.unlink(missing_ok=True)
