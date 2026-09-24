"""Compatibility shim for the pipeline configuration models.

The declarative pipeline spec moved to :mod:`rag_core.pipeline_config` so the
``rag-aio`` facade can validate pipeline configuration with only its
dependency-light base install. Every public name is re-exported unchanged, so
``from rag_orchestrator.config import PipelineConfig`` keeps working.
"""

from __future__ import annotations

from rag_core.pipeline_config import (
    EmbeddingStage,
    GenerationStage,
    IngestionStage,
    PipelineConfig,
    RerankStage,
    RetrievalStage,
    StageConfig,
    pipeline_schema_version,
)

__all__ = [
    "EmbeddingStage",
    "GenerationStage",
    "IngestionStage",
    "PipelineConfig",
    "RerankStage",
    "RetrievalStage",
    "StageConfig",
    "pipeline_schema_version",
]
