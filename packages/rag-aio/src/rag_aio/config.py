"""Configuration models for the rag-aio facade.

All models use :class:`rag_core.base.RagBaseModel` (``extra="forbid"``) so that
configuration drift fails loudly.  Heavy backends are never imported here — the
config is a pure data description consumed by :func:`rag_aio.facade.build_services`.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from rag_core.base import RagBaseModel
from rag_core.pipeline_config import PipelineConfig

__all__ = [
    "EmbedderConfig",
    "GenerationConfig",
    "RAGConfig",
    "StorageConfig",
    "env_overrides",
]


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in place (override wins)."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def env_overrides(prefix: str = "RAG_AIO_", env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect environment-variable overrides per the ADR-0006 env tier.

    Every variable whose name starts with ``prefix`` (default ``RAG_AIO_``)
    becomes an override:

    * The prefix is stripped; an empty remainder is skipped.
    * The remainder is split on ``__`` into a nested path, segments lowercased
      (``RAG_AIO_STORAGE__DB_URL`` -> ``{"storage": {"db_url": ...}}``).
    * Values are parsed with :func:`json.loads`; values that are not valid JSON
      (e.g. ``mock``, ``http://localhost:1234/v1``) are kept as raw strings.
      This means ``RAG_AIO_GENERATION__TEMPERATURE=0.2`` -> ``0.2`` (float) and
      ``RAG_AIO_ASK__STREAM=true`` -> ``True``.

    Only segments whose *top-level* key exists in :attr:`RAGConfig.model_fields`
    are kept; unknown top-level keys are dropped so that ``extra="forbid"``
    remains a loud-failure contract instead of being bypassed via the
    environment. Nested (non-top-level) segments are passed through untouched
    and validated later by :meth:`RAGConfig.model_validate`.

    Args:
        prefix: Variable-name prefix to match (case-sensitive).
        env: Mapping to read from; defaults to :data:`os.environ`. Pass an
            explicit mapping for deterministic tests.

    Returns:
        A nested dict suitable for deep-merging over TOML data.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    overrides: dict[str, Any] = {}
    for name, raw in source.items():
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        if not remainder:
            continue
        segments = [segment.lower() for segment in remainder.split("__")]
        if segments[0] not in RAGConfig.model_fields:
            continue
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = overrides
        for segment in segments[:-1]:
            child = node.get(segment)
            if not isinstance(child, dict):
                child = {}
                node[segment] = child
            node = child
        node[segments[-1]] = value
    return overrides


class EmbedderConfig(RagBaseModel):
    """Embedding-backend selection and dimension hints.

    ``backend="remote"`` selects an OpenAI-compatible ``/embeddings`` endpoint
    (see :class:`rag_embedder.OpenAICompatibleEmbedder`): ``dense_model`` is the
    remote model id and ``api_key_ref``/base_url live with the caller (the
    facade's :func:`build_services` does not wire remote embedders itself).
    """

    backend: Literal["mock", "fastembed", "remote"] = "fastembed"
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
        """Load a :class:`RAGConfig` from a TOML file (Python 3.12 ``tomllib``).

        ``RAG_AIO_``-prefixed environment variables (see :func:`env_overrides`)
        are deep-merged on top of the TOML data before validation, so the
        ADR-0006 precedence holds: built-in defaults -> TOML file -> environment
        (environment wins).
        """
        with open(path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        _deep_merge(data, env_overrides())
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
