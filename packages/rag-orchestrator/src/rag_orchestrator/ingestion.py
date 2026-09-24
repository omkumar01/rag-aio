"""Stage-cached ingestion: load -> parse -> chunk -> embed -> index.

This is the orchestration glue between the doc-handler ingestion pipeline
(load + parse + dedup) and the embedder indexing pipeline (chunk -> embed ->
index). A per-stage cache (keyed by content hash + config hash) lets repeated
ingests of unchanged documents skip the expensive embed/index step.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.documents import Document
from rag_core.errors import ConfigError, IngestionError, RagError, UnsupportedFormatError
from rag_core.ids import config_hash
from rag_core.serde import from_json, to_json

from .config import PipelineConfig
from .services import OrchestratorServices

if TYPE_CHECKING:
    from rag_cache.backends.memory import MemoryCache  # noqa: F401  (hint only)

__all__ = ["ingest", "ingest_directory"]


def _ingestion_cache_key(source: str, config: PipelineConfig) -> str:
    """Deterministic cache key for a source + ingestion config.

    Keyed by source URI (not content hash): the content hash is unknown until
    after parsing, and the read probe must match the write key. Content
    identity is handled separately by dedup.
    """
    from rag_cache.keys import cache_key

    stage_cfg = config_hash(config.ingestion.model_dump(mode="json"))
    return cache_key("ingest", source=source, config_hash=stage_cfg)


async def ingest(
    services: OrchestratorServices,
    source: str,
    dedup: Any | None = None,
    *,
    config: PipelineConfig | None = None,
) -> Document:
    """Ingest a single source through the configured pipeline.

    Steps:
      1. ``cache get`` — if a cached indexed document exists for this
         content_hash + ingestion config, return it (unless ``cache_reads`` is
         disabled).
      2. load + parse + dedup via :class:`~rag_doc_handler.pipeline.IngestionPipeline`.
      3. chunk -> embed -> index via
         :class:`~rag_embedder.pipeline.EmbeddingPipeline`.
      4. persist the canonical document (if a document store is wired).
      5. ``cache set`` the resulting document.

    Returns the canonical :class:`~rag_core.documents.Document`.
    """
    cfg = config or PipelineConfig.local_default()
    pipeline = services.ingestion_pipeline
    if pipeline is None:
        msg = "ingestion_pipeline is required: wire one via load_local_services()"
        raise ConfigError(msg)
    embeddings = services.embedding_pipeline
    if embeddings is None:
        msg = "embedding_pipeline is required: wire one via load_local_services()"
        raise ConfigError(msg)
    store = services.vector_store
    if store is None:
        msg = "vector_store is required: wire one via load_local_services()"
        raise ConfigError(msg)

    cache = services.cache

    # 1. short-circuit from cache (source + ingestion config).
    cache_key = _ingestion_cache_key(source, cfg)
    if cache is not None and cfg.cache_reads:
        cached = await cache.get(cache_key)
        if cached is not None:
            return from_json(cached.decode(), Document)

    start = time.perf_counter()
    # 2. load -> parse -> dedup.
    try:
        document, is_new = await pipeline.ingest(source, dedup=dedup)
    except RagError:
        raise
    except Exception as exc:
        msg = f"failed to ingest {source!r}: {exc}"
        raise IngestionError(msg) from exc

    # If dedup says it is not new and we have a cached document, reuse it.
    if not is_new and cache is not None and cfg.cache_reads:
        existing = await cache.get(cache_key)
        if existing is not None:
            return from_json(existing.decode(), Document)
        return document

    # 3. chunk -> embed -> index.
    try:
        outcome = await embeddings.process(document, store)
    except Exception as exc:
        msg = f"indexing failed for {document.id}: {exc}"
        raise IngestionError(msg) from exc

    # 4. persist canonical document metadata.
    if services.document_store is not None:
        try:
            await services.document_store.put(document)
        except Exception as exc:
            services.observer.record("document_store.put_failed", {"error": str(exc)})

    took_ms = (time.perf_counter() - start) * 1000.0
    services.observer.record(
        "ingest.done",
        {
            "source": source,
            "document_id": document.id,
            "chunks_indexed": outcome.chunks_indexed,
            "took_ms": took_ms,
        },
    )

    # 5. cache the result.
    if cache is not None and cfg.cache_writes:
        await cache.set(cache_key, to_json(document).encode())

    return document


async def ingest_directory(
    services: OrchestratorServices,
    directory: str | Path,
    recursive: bool = False,
    *,
    config: PipelineConfig | None = None,
    dedup: Any | None = None,
) -> list[Document]:
    """Ingest every supported file under ``directory``.

    Unsupported formats are skipped (logged). A shared :class:`DedupIndex` is
    used within the run so duplicate files are not re-indexed.
    """
    from rag_doc_handler.pipeline import DedupIndex  # lazy: keep import light.

    def _collect() -> list[Path]:
        base = Path(directory)
        if not base.is_dir():
            msg = f"not a directory: {directory}"
            raise IngestionError(msg)
        paths = sorted(base.rglob("*") if recursive else base.iterdir())
        return [p for p in paths if p.is_file()]

    files = await asyncio.to_thread(_collect)

    dedup_index = dedup if dedup is not None else DedupIndex()
    documents: list[Document] = []
    for path in files:
        try:
            document = await ingest(services, str(path), dedup=dedup_index, config=config)
        except UnsupportedFormatError as exc:
            services.observer.record(
                "ingest.skipped",
                {"source": str(path), "error": str(exc), "code": exc.code},
            )
            continue
        documents.append(document)
    return documents
