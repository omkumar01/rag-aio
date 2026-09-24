"""rag-mass-inject: high-throughput bulk ingestion.

Provides :class:`MassIngestor` — a bounded, resumable batch ingestion engine
that sits on top of an :class:`~rag_orchestrator.services.OrchestratorServices`
bag, using the ingestion pipeline (load + parse) and embedding pipeline
(chunk + embed + index) with per-stage concurrency, checkpoint-based resume,
and dead-letter handling.
"""

from __future__ import annotations

from .config import MassInjectConfig
from .jobs import JobTracker
from .pipeline import MassIngestor
from .sources import (
    SUPPORTED_EXTENSIONS,
    discover_directory,
    discover_file_list,
    discover_sitemap,
    is_supported,
)
from .workers import BoundedQueue, StageWorker

__version__ = "0.1.1"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "BoundedQueue",
    "JobTracker",
    "MassIngestor",
    "MassInjectConfig",
    "StageWorker",
    "__version__",
    "discover_directory",
    "discover_file_list",
    "discover_sitemap",
    "is_supported",
]
