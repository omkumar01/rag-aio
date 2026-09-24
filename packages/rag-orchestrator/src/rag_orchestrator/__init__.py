"""rag-orchestrator: strategy and workflow-management for rag-aio.

The orchestrator composes the contract-bound components of the other rag-aio
packages into declarative pipelines. It is dependency-light at import time:
heavy backends (fastembed, qdrant-client, the doc-handler parsers, fastapi) are
imported lazily — either inside :func:`load_local_services` or behind the
``app`` submodule — so ``import rag_orchestrator`` stays cheap.

Public API:
    Configuration: :class:`PipelineConfig`, :data:`pipeline_schema_version`,
        :class:`StageConfig` and its subclasses.
    Spec/registry: :class:`Pipeline`, :class:`PipelineRegistry`,
        :func:`default_pipeline`, :func:`default_registry`.
    Runtime: :class:`Orchestrator`, :class:`AskResult`.
    Wiring: :class:`OrchestratorServices`, :func:`load_local_services`.
    Ingestion: :func:`ingest`, :func:`ingest_directory`.
    App: :func:`create_app` (FastAPI, lazy — requires the ``fastapi`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import (
    EmbeddingStage,
    GenerationStage,
    IngestionStage,
    PipelineConfig,
    RerankStage,
    RetrievalStage,
    StageConfig,
    pipeline_schema_version,
)
from .ingest import ingest, ingest_directory
from .orchestrator import SYSTEM_PROMPT, AskResult, Orchestrator, OrchestratorConfigError
from .pipeline import (
    Pipeline,
    PipelineRegistry,
    default_pipeline,
    default_registry,
)
from .services import NoOpObserver, OrchestratorServices, load_local_services

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .config import PipelineConfig as _PipelineConfig
    from .services import OrchestratorServices as _Services

__version__ = "0.1.0"

__all__ = [
    "SYSTEM_PROMPT",
    "AskResult",
    "EmbeddingStage",
    "GenerationStage",
    "IngestionStage",
    "NoOpObserver",
    "Orchestrator",
    "OrchestratorConfigError",
    "OrchestratorServices",
    "Pipeline",
    "PipelineConfig",
    "PipelineRegistry",
    "RerankStage",
    "RetrievalStage",
    "StageConfig",
    "__version__",
    "default_pipeline",
    "default_registry",
    "ingest",
    "ingest_directory",
    "load_local_services",
    "pipeline_schema_version",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the FastAPI app surface so ``import rag_orchestrator`` stays
    cheap when the ``fastapi`` extra is not installed."""
    if name == "create_app":
        from .app.app import create_app

        return create_app
    if name == "app":
        from . import app as _app

        return _app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


if TYPE_CHECKING:

    def create_app(
        services: _Services, pipeline_config: _PipelineConfig | None = None
    ) -> FastAPI: ...
