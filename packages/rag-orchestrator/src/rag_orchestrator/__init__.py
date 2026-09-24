"""rag-orchestrator: strategy and workflow-management for rag-aio.

The orchestrator composes the contract-bound components of the other rag-aio
packages into declarative pipelines. It is dependency-light at import time:
only the configuration models (which need just ``rag-core``) are imported
eagerly — the pipeline, runtime, wiring, and ingestion surfaces are exposed
lazily (PEP 562) so ``import rag_orchestrator`` never pulls the sibling
``rag-*`` packages until their names are actually used.

Public API:
    Configuration (eager): :class:`PipelineConfig`, :data:`pipeline_schema_version`,
        :class:`StageConfig` and its subclasses.
    Spec/registry (lazy): :class:`Pipeline`, :class:`PipelineRegistry`,
        :func:`default_pipeline`, :func:`default_registry`.
    Runtime (lazy): :class:`Orchestrator`, :class:`AskResult`.
    Wiring (lazy): :class:`OrchestratorServices`, :func:`load_local_services`.
    Ingestion (lazy): :func:`ingest`, :func:`ingest_directory`.
    App (lazy): :func:`create_app` (FastAPI, requires the ``fastapi`` extra).
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

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .config import PipelineConfig as _PipelineConfig
    from .ingestion import ingest, ingest_directory
    from .orchestrator import (
        SYSTEM_PROMPT,
        AskResult,
        Orchestrator,
        OrchestratorConfigError,
    )
    from .pipeline import (
        Pipeline,
        PipelineRegistry,
        default_pipeline,
        default_registry,
    )
    from .services import (
        NoOpObserver,
        load_local_services,
    )
    from .services import (
        OrchestratorServices as _Services,
    )

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

# Names served lazily from the heavy submodules (everything but ``.config``).
_LAZY_ATTRS: dict[str, str] = {
    "ingest": ".ingestion",
    "ingest_directory": ".ingestion",
    "SYSTEM_PROMPT": ".orchestrator",
    "AskResult": ".orchestrator",
    "Orchestrator": ".orchestrator",
    "OrchestratorConfigError": ".orchestrator",
    "Pipeline": ".pipeline",
    "PipelineRegistry": ".pipeline",
    "default_pipeline": ".pipeline",
    "default_registry": ".pipeline",
    "NoOpObserver": ".services",
    "OrchestratorServices": ".services",
    "load_local_services": ".services",
}


def __getattr__(name: str) -> Any:
    """Lazily expose the heavy submodules so ``import rag_orchestrator`` only
    pulls the sibling ``rag-*`` packages when their names are actually used."""
    if name in _LAZY_ATTRS:
        import importlib

        value = getattr(importlib.import_module(_LAZY_ATTRS[name], __name__), name)
        # Importing the submodule binds the *module* under this name on the
        # package; cache the real object so it keeps shadowing the module,
        # matching the semantics of a plain ``from .sub import name``.
        globals()[name] = value
        return value
    if name == "create_app":
        from .app.app import create_app

        return create_app
    if name == "app":
        from . import app as _app

        return _app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | {"create_app", "app"})


if TYPE_CHECKING:

    def create_app(
        services: _Services, pipeline_config: _PipelineConfig | None = None
    ) -> FastAPI: ...
