"""Per-user RAG engine construction over one shared, namespace-isolated store.

:class:`RagManager` lazily builds a base :class:`~rag_aio.facade.RAG` service
bag from a :class:`~rag_aio.config.RAGConfig` (mock profile by default — fully
offline) and hands every user a *view* of that bag whose vector store is a
:class:`~web_example.services.collections.CompositeVectorStore` scoped to
``user_{id}`` + ``global`` namespaces. Per-user retriever/pipeline components
are cloned so the namespace swap never leaks between users.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rag_aio.config import RAGConfig
from rag_aio.facade import RAG
from rag_core.documents import Document
from rag_core.ids import new_id
from rag_core.protocols import Retriever, VectorStore
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.services.collections import CompositeVectorStore
from web_example.services.local_services import ResolvedSettings, build_pool_services
from web_example.settings import AppSettings

if TYPE_CHECKING:
    from rag_orchestrator import OrchestratorServices

__all__ = ["ConfigProvider", "RagManager"]

logger = logging.getLogger(__name__)

#: Async callable ``(session, user_id) -> ResolvedSettings`` (wired by the
#: settings track); the default resolves the offline mock profile.
ConfigProvider = Callable[[AsyncSession, str], Awaitable[ResolvedSettings]]

_GLOBAL_NS = "global"
_GLOBAL_CACHE_KEY = "__global__"
_REBIND_MAX_DEPTH = 8


def _contains_value(node: Any, target: Any, depth: int = 0, seen: set[int] | None = None) -> bool:
    """True when *node* references *target* directly or transitively."""
    if node is target:
        return True
    if depth >= _REBIND_MAX_DEPTH:
        return False
    if node is None or isinstance(node, (str, bytes, bytearray, int, float, bool)):
        return False
    if seen is None:
        seen = set()
    marker = id(node)
    if marker in seen:
        return False
    seen.add(marker)
    if isinstance(node, (list, tuple, set, frozenset)):
        return any(_contains_value(item, target, depth + 1, seen) for item in node)
    if isinstance(node, dict):
        return any(_contains_value(item, target, depth + 1, seen) for item in node.values())
    attrs = getattr(node, "__dict__", None)
    if isinstance(attrs, dict):
        return any(_contains_value(item, target, depth + 1, seen) for item in attrs.values())
    return False


def _rebind(node: Any, target: Any, replacement: Any, depth: int = 0) -> Any:
    """Return a copy of *node* with every reference to *target* swapped.

    Containers and plain objects on the path to *target* are shallow-copied so
    the originals (shared between users) are never mutated.
    """
    if node is target:
        return replacement
    if depth >= _REBIND_MAX_DEPTH or not _contains_value(node, target):
        return node
    if isinstance(node, list):
        return [_rebind(item, target, replacement, depth + 1) for item in node]
    if isinstance(node, tuple):
        return tuple(_rebind(item, target, replacement, depth + 1) for item in node)
    if isinstance(node, dict):
        return {key: _rebind(item, target, replacement, depth + 1) for key, item in node.items()}
    attrs = getattr(node, "__dict__", None)
    if isinstance(attrs, dict):
        clone = copy.copy(node)
        for name, value in attrs.items():
            setattr(clone, name, _rebind(value, target, replacement, depth + 1))
        return clone
    return node


class RagManager:
    """Builds and caches per-user :class:`~rag_aio.facade.RAG` instances.

    The base service bag (embedder, chunker, generator, ...) is built once per
    resolved config and shared; every user gets their own composite vector
    store view plus cloned retriever/pipeline components over it.
    """

    def __init__(
        self,
        settings: AppSettings,
        *,
        config_provider: ConfigProvider | None = None,
    ) -> None:
        self.settings = settings
        self.config_provider: ConfigProvider | None = config_provider
        self._rag_cache: dict[str, RAG] = {}
        self._base_cache: dict[str, OrchestratorServices] = {}
        self._lock = asyncio.Lock()

    # -- config resolution ---------------------------------------------------

    async def _resolve_settings(self, session: AsyncSession, user_id: str) -> ResolvedSettings:
        provider = self.config_provider
        if provider is not None:
            return await provider(session, user_id)
        return ResolvedSettings(config=RAGConfig.mock())

    @staticmethod
    def _cache_key(resolved: ResolvedSettings) -> str:
        return hashlib.sha256(resolved.model_dump_json().encode("utf-8")).hexdigest()

    async def _base_services(
        self, resolved: ResolvedSettings
    ) -> tuple[OrchestratorServices, VectorStore]:
        key = self._cache_key(resolved)
        async with self._lock:
            services = self._base_cache.get(key)
            if services is None:
                services = cast("OrchestratorServices", build_pool_services(resolved))
                self._base_cache[key] = services
        base = services.vector_store
        if base is None:
            msg = "service factory produced no vector store"
            raise RuntimeError(msg)
        return services, base

    # -- instance construction -----------------------------------------------

    async def _build_rag(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        primary_ns: str,
        search_ns: tuple[str, ...],
    ) -> RAG:
        resolved = await self._resolve_settings(session, user_id)
        base_services, base_store = await self._base_services(resolved)
        composite = CompositeVectorStore(base_store, primary_ns, search_ns)
        services = self._user_services(base_services, base_store, composite)
        services.ingestion_pipeline = self._routed_ingestion_pipeline(base_services)
        return RAG(services, self._pipeline_config(resolved.config))

    @staticmethod
    def _pipeline_config(config: RAGConfig) -> Any:
        """Pipeline config with the generation settings applied as overrides.

        ``RAGConfig.generation`` is only metadata for :func:`build_services` —
        the orchestrator reads per-request values from the pipeline's
        generation-stage overrides, so model/temperature/max_tokens chosen on
        the settings page must be mirrored there.
        """
        overrides = dict(config.pipeline.generation.overrides)
        if config.generation.model:
            overrides["model"] = config.generation.model
        overrides["temperature"] = config.generation.temperature
        if config.generation.max_tokens:
            overrides["max_tokens"] = config.generation.max_tokens
        generation = config.pipeline.generation.model_copy(update={"overrides": overrides})
        return config.pipeline.model_copy(update={"generation": generation})

    @staticmethod
    def _user_services(
        base_services: OrchestratorServices,
        base_store: VectorStore,
        composite: CompositeVectorStore,
    ) -> OrchestratorServices:
        """Copy the shared bag and point every store reference at *composite*."""
        services = copy.copy(base_services)
        services.vector_store = composite
        services.retrievers = [
            cast(Retriever, _rebind(retriever, base_store, composite))
            for retriever in base_services.retrievers
        ]
        if services.embedding_pipeline is not None:
            services.embedding_pipeline = cast(
                Any, _rebind(services.embedding_pipeline, base_store, composite)
            )
        for name, value in vars(services).items():
            if name in {"vector_store", "retrievers", "embedding_pipeline"}:
                continue
            rebound = _rebind(value, base_store, composite)
            if rebound is not value:
                setattr(services, name, rebound)
        return services

    @staticmethod
    def _routed_ingestion_pipeline(base_services: OrchestratorServices) -> Any:
        """An ingestion pipeline that also loads http(s) sources.

        ``default_pipeline()`` is file-only; web/sitemap ingestion needs the
        scheme-routing loader. The parser registry (incl. optional parsers) is
        carried over from the base bag.
        """
        from rag_doc_handler import default_registry
        from rag_doc_handler.pipeline import IngestionPipeline

        from web_example.services.loader import MimeAwareRegistry, SchemeRouterLoader

        base_pipeline = base_services.ingestion_pipeline
        registry = getattr(base_pipeline, "registry", None) or default_registry()
        return IngestionPipeline(registry=MimeAwareRegistry(registry), loader=SchemeRouterLoader())

    # -- public API ----------------------------------------------------------

    async def get(self, session: AsyncSession, user_id: str) -> RAG:
        """Return the cached per-user RAG facade (built on first use)."""
        async with self._lock:
            cached = self._rag_cache.get(user_id)
        if cached is not None:
            return cached
        primary = f"user_{user_id}"
        rag = await self._build_rag(
            session, user_id=user_id, primary_ns=primary, search_ns=(primary, _GLOBAL_NS)
        )
        async with self._lock:
            existing = self._rag_cache.get(user_id)
            if existing is not None:
                return existing
            self._rag_cache[user_id] = rag
        return rag

    async def get_global(self, session: AsyncSession) -> RAG:
        """Return the admin-managed global-pool RAG facade."""
        async with self._lock:
            cached = self._rag_cache.get(_GLOBAL_CACHE_KEY)
        if cached is not None:
            return cached
        rag = await self._build_rag(
            session, user_id=_GLOBAL_CACHE_KEY, primary_ns=_GLOBAL_NS, search_ns=(_GLOBAL_NS,)
        )
        async with self._lock:
            existing = self._rag_cache.get(_GLOBAL_CACHE_KEY)
            if existing is not None:
                return existing
            self._rag_cache[_GLOBAL_CACHE_KEY] = rag
        return rag

    async def invalidate(self, user_id: str | None = None) -> None:
        """Drop cached RAG instances (all of them when *user_id* is None)."""
        async with self._lock:
            if user_id is None:
                self._rag_cache.clear()
                self._base_cache.clear()
            else:
                self._rag_cache.pop(user_id, None)

    async def count_chunks(self, session: AsyncSession, user_id: str) -> int:
        """Number of indexed chunks currently visible to *user_id*."""
        rag = await self.get(session, user_id)
        store = rag.services.vector_store
        if isinstance(store, CompositeVectorStore):
            return await store.count()
        return 0

    async def mass_ingestor(
        self,
        session: AsyncSession,
        user_id: str,
        *,
        global_pool: bool = False,
    ) -> Any:
        """A :class:`~rag_mass_inject.MassIngestor` bound to the target pool.

        Sources discovered by the mass pipeline (directories, sitemaps) are
        ingested through the pool's own services, so writes land in the
        ``user_{id}`` or ``global`` namespace exactly like uploads do.
        Checkpoints live under the data dir, one sqlite per pool.
        """
        from rag_mass_inject import MassIngestor
        from rag_mass_inject.config import MassInjectConfig

        if global_pool:
            rag = await self.get_global(session)
        else:
            rag = await self.get(session, user_id)
        pool = _GLOBAL_NS if global_pool else f"user_{user_id}"
        config = MassInjectConfig.local_default().model_copy(
            update={
                "checkpoint_dir": str(self.settings.data_dir / "checkpoints" / pool),
                "recursive": True,
            }
        )
        return MassIngestor(config, rag.services)

    async def ingest_bytes(
        self,
        session: AsyncSession,
        user_id: str,
        name: str,
        data: bytes,
        *,
        global_pool: bool = False,
    ) -> Document:
        """Persist uploaded bytes under the uploads dir and ingest them."""
        rag = await self.get_global(session) if global_pool else await self.get(session, user_id)
        target = self.settings.uploads_dir / (_GLOBAL_NS if global_pool else f"user_{user_id}")
        target.mkdir(parents=True, exist_ok=True)
        safe_name = Path(name).name or "upload.bin"
        path = target / f"{new_id()}_{safe_name}"
        path.write_bytes(data)
        return await rag.ingest(path)

    async def delete_document(
        self,
        session: AsyncSession,
        user_id: str,
        document_id: str,
        *,
        global_pool: bool = False,
    ) -> None:
        """Delete a document's vectors (best effort) and its registry row."""
        rag = await self.get_global(session) if global_pool else await self.get(session, user_id)
        store = rag.services.vector_store
        primary_ns: str | None = None
        base: Any = store
        if isinstance(store, CompositeVectorStore):
            primary_ns = store.primary_ns
            base = store.base
        probe = await self._probe_vector(rag)
        removed = await self._delete_document_vectors(base, document_id, primary_ns, probe)
        if not removed:
            logger.warning(
                "document %s: vector cleanup unsupported by %s; removing registry row only",
                document_id,
                type(base).__name__,
            )
        row = await session.get(Document, document_id)
        if row is not None:
            await session.delete(row)
            await session.commit()

    @staticmethod
    async def _probe_vector(rag: RAG) -> list[float] | None:
        """A correctly-dimensioned probe vector for filtered searches (best effort)."""
        embedder = rag.services.embedder
        if embedder is None:
            return None
        try:
            vectors = await embedder.embed([""])
            return [float(v) for v in vectors[0]]
        except Exception:
            return None

    @staticmethod
    async def _delete_document_vectors(
        base: Any, document_id: str, primary_ns: str | None, probe: list[float] | None
    ) -> bool:
        """Best-effort vector deletion. Returns False when unsupported."""
        for method in ("delete_by_document", "delete_document"):
            fn = getattr(base, method, None)
            if callable(fn):
                outcome = fn(document_id, namespace=primary_ns)
                if inspect.isawaitable(outcome):
                    await outcome
                return True
        delete_by_filter = getattr(base, "delete_by_filter", None)
        if callable(delete_by_filter):
            outcome = delete_by_filter({"document_id": document_id}, namespace=primary_ns)
            if inspect.isawaitable(outcome):
                await outcome
            return True
        # Last resort: collect chunk ids via a filtered probe search, then delete.
        try:
            counter = getattr(base, "count", None)
            total = 0
            if callable(counter):
                counted = counter(namespace=primary_ns)
                total = int(await counted) if inspect.isawaitable(counted) else int(counted)
            hits = await base.search(
                probe or [0.0],
                max(total, 1),
                filters={"document_id": document_id},
                namespace=primary_ns,
            )
        except Exception:
            return False
        if not hits:
            return True
        await base.delete([hit.chunk_id for hit in hits], namespace=primary_ns)
        return True
