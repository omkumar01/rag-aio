"""Configuration models for the rag-aio facade.

All models use :class:`rag_core.base.RagBaseModel` (``extra="forbid"``) so that
configuration drift fails loudly.  Heavy backends are never imported here — the
config is a pure data description consumed by :func:`rag_aio.facade.build_services`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from rag_core.base import RagBaseModel
from rag_orchestrator.config import PipelineConfig

__all__ = ["EmbedderConfig", "GenerationConfig", "RAGConfig", "StorageConfig"]


class EmbedderConfig(RagBaseModel):
    """Embedding-backend selection and dimension hints."""

    backend: Literal["mock", "fastembed"] = "fastembed"
    dense_model: str | None = None
    sparse_model: str | None = None
    dim: int = 384


class StorageConfig(RagBaseModel):
    """Persistence-layer locations."""

    qdrant_path: str = "./data/qdrant"
    db_url: str = "sqlite+aiosqlite:///./data/rag.db"
    cache_backend: Literal["memory", "sqlite"] = "memory"
    cache_path: str | None = None


class GenerationConfig(RagBaseModel):
    """Generation provider settings.

    ``api_key_ref`` is an *environment-variable name* (secret-by-reference),
    never a credential value (ADR-0004/0006).
    """

    provider: str = "lm_studio"
    base_url: str = "http://localhost:1234/v1"
    model: str | None = None
    api_key_ref: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None


class RAGConfig(RagBaseModel):
    """Top-level facade configuration: pipeline + backends."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig.local_default)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> RAGConfig:
        """Load a :class:`RAGConfig` from a TOML file (Python 3.12 ``tomllib``)."""
        with open(path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        return cls.model_validate(data)

    @classmethod
    def mock(cls) -> RAGConfig:
        """Return a config that wires fully-offline mock backends.

        ``embedder.backend`` is ``"mock"`` and the pipeline's embedding policy is
        overridden to ``"mock"`` so downstream components know to use deterministic
        mock implementations.
        """
        pipeline = PipelineConfig.local_default()
        embedding = pipeline.embedding.model_copy(update={"policy": "mock"})
        pipeline = pipeline.model_copy(update={"embedding": embedding})
        return cls(
            pipeline=pipeline,
            embedder=EmbedderConfig(backend="mock"),
        )
