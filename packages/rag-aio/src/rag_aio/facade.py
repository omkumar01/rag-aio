"""High-level RAG facade: config-driven service composition and convenience API.

:class:`RAG` is the single entry point for most users.  It wires together the
contract-bound components from every ``rag-*`` package into an
:class:`~rag_orchestrator.Orchestrator` and exposes a small, opinionated surface
(``ingest``, ``ask``, ``app``, async context manager).

Heavy backends (``fastembed``, ``qdrant-client``, ``rag_doc_handler`` parsers,
``numpy``, …) are imported **lazily inside** :func:`build_services` and the mock
wiring it delegates to (:func:`rag_orchestrator.build_mock_services`) so that
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

    Thin adapter over :func:`rag_orchestrator.build_mock_services`, where the
    mock wiring actually lives (member-only consumers can call it directly
    without this facade).  Only config-derived values are mapped through.
    """
    orchestrator = _require_orchestrator()
    return orchestrator.build_mock_services(vector_size=config.embedder.dim)


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
