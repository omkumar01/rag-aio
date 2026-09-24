"""Declarative pipeline configuration for ``rag-orchestrator``.

A :class:`PipelineConfig` is a *declarative spec only*: it holds no live
components. It is strictly validated (``extra="forbid"`` via
:class:`rag_core.base.RagBaseModel`) so pipeline drift fails loudly, and it is
versioned by :data:`pipeline_schema_version` so a future breaking shape can be
rejected rather than silently misinterpreted.

Component construction/compilation from a config happens in
:mod:`rag_orchestrator.services` (:func:`load_local_services`) and the runtime
in :mod:`rag_orchestrator.orchestrator`.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from rag_core.base import RagBaseModel

pipeline_schema_version = "1.0"
"""Monotonic schema version for :class:`PipelineConfig` (bump on breaking change)."""


class StageConfig(RagBaseModel):
    """Common shape shared by every pipeline stage.

    Attributes:
        enable: When ``False`` the stage is skipped at runtime.
        strategy: Named strategy to bind *within* the stage's domain
            (e.g. ``"rrf"`` for retrieval fusion, ``"recursive"`` for the
            chunker). This selects an algorithm, not a backend.
        policy: Named operational policy to bind (e.g. ``"local_fast"``).
            Subclasses constrain/retype this field.
        overrides: Per-stage config overrides merged into the bound component's
            own config object (e.g. ``top_k`` for retrieval).
    """

    enable: bool = True
    strategy: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class IngestionStage(StageConfig):
    """Ingestion stage."""

    policy: str | None = None
    """Named ingestion policy (e.g. ``"pymupdf"``, ``"fast"``)."""


class EmbeddingStage(StageConfig):
    """Embedding + chunking stage."""

    policy: str | None = None
    """Named embedding policy (e.g. ``"fastembed"``, ``"mock"``)."""


class RetrievalStage(StageConfig):
    """Retrieval stage."""

    policy: str | None = None
    """Named retrieval policy (e.g. ``"hybrid"``, ``"dense"``, ``"sparse"``, ``"rrf"``)."""


class RerankStage(StageConfig):
    """Reranking stage."""

    policy: str | None = None
    """Named rerank policy (e.g. ``"rerank"``, ``"none"``)."""


class ContextStage(StageConfig):
    """Context assembly stage."""

    policy: str | None = None
    """Named context policy (e.g. ``"relevance_first"``, ``"chronological"``)."""


class GenerationStage(StageConfig):
    """Generation stage."""

    policy: str | None = None
    """Named generation policy (e.g. ``"lm_studio"``, ``"openai"``)."""


class PipelineConfig(RagBaseModel):
    """Versioned, user-facing pipeline specification.

    The schema is intentionally flat: each stage is a :class:`StageConfig`
    subclass (with its typed ``policy`` field) plus shared execution budgets and
    concurrency. Stage-specific tunables are carried in each stage's
    ``overrides`` dict and applied to the bound component's config at wiring
    time.
    """

    schema_version: str = pipeline_schema_version
    name: str = "default"
    description: str | None = None
    ingestion: IngestionStage = Field(default_factory=IngestionStage)
    embedding: EmbeddingStage = Field(default_factory=EmbeddingStage)
    retrieval: RetrievalStage = Field(default_factory=RetrievalStage)
    rerank: RerankStage = Field(default_factory=RerankStage)
    context: ContextStage = Field(default_factory=ContextStage)
    generation: GenerationStage = Field(default_factory=GenerationStage)
    timeout_s: float | None = None
    """End-to-end budget for a single ``ask`` (seconds). ``None`` = no limit."""
    token_budget: int | None = None
    """Generation answer budget (tokens). Propagated to the context builder as
    the context token budget when no per-stage value is given."""
    cost_budget_usd: float | None = None
    """Soft spend cap for generation (USD); enforced via the usage callback."""
    concurrency: int = Field(default=4, ge=1)
    """Parallelism for independent retrieval branches."""
    cache_reads: bool = True
    cache_writes: bool = True

    @classmethod
    def local_default(cls) -> PipelineConfig:
        """The bundled local-development profile.

        A single-node, iteration-friendly profile that requires no cloud
        services (LM Studio for generation, on-disk Qdrant for vectors):

        * **ingestion** — PyMuPDF-backed PDF parsing (``strategy="pymupdf"``,
          ``policy="fast"``) with OCR enabled.
        * **embedding** — ``recursive`` chunker, dense via FastEmbed
          (``BAAI/bge-small-en-v1.5``) and learned-sparse (BM25-style) vectors.
        * **retrieval** — sparse (BM25) + dense (FastEmbed) hybrid fused with
          RRF (default ``k=60``); ``candidate_k=50``, ``top_k=10``.
        * **rerank** — heuristic Jaccard rerank over at most 50 candidates,
          ``top_k=10``.
        * **context** — ~2048 token budget, ``relevance_first`` ordering,
          numeric citations, a small ``reserve_for_answer``.
        * **generation** — LM Studio OpenAI-compatible endpoint
          (``http://localhost:1234/v1``), ``temperature=0.2``.
        """
        return cls(
            schema_version=pipeline_schema_version,
            name="local_fast",
            description="Local single-node profile: PyMuPDF + FastEmbed + "
            "Qdrant local + LM Studio.",
            ingestion=IngestionStage(policy="fast", strategy="pymupdf", overrides={"ocr": True}),
            embedding=EmbeddingStage(
                policy="fastembed",
                strategy="recursive",
                overrides={"dense": "BAAI/bge-small-en-v1.5", "sparse": "Qdrant/bm25"},
            ),
            retrieval=RetrievalStage(
                policy="hybrid",
                strategy="rrf",
                overrides={"mode": "hybrid", "top_k": 10, "candidate_k": 50},
            ),
            rerank=RerankStage(
                policy="rerank",
                strategy="heuristic",
                overrides={"max_candidates": 50, "top_k": 10},
            ),
            context=ContextStage(
                policy="relevance_first",
                strategy="relevance_first",
                overrides={"token_budget": 2048, "reserve_for_answer": 64},
            ),
            generation=GenerationStage(
                policy="lm_studio",
                strategy="openai_compatible",
                overrides={
                    "base_url": "http://localhost:1234/v1",
                    "model": "local-chat",
                    "temperature": 0.2,
                },
            ),
        )

    def merge_overrides(self, overrides: dict[str, Any] | None) -> PipelineConfig:
        """Return a copy with per-request ``overrides`` folded into stages.

        Recognized flat keys are routed to the appropriate stage's ``overrides``
        dict (or to a top-level budget). Unknown keys are written to the
        ``generation`` overrides so experimenters can pass provider-specific
        options without fighting validation.
        """
        if not overrides:
            return self
        copy = self.model_copy(deep=True)
        for key, value in overrides.items():
            if key in {"top_k", "candidate_k", "mode", "fusion", "dense_weight", "sparse_weight"}:
                copy.retrieval.overrides[key] = value
            elif key in {"max_candidates"}:
                copy.rerank.overrides[key] = value
            elif key == "token_budget":
                copy.token_budget = int(value)
                copy.context.overrides[key] = int(value)
            elif key in {"base_url", "model", "temperature", "max_tokens", "top_p"}:
                copy.generation.overrides[key] = value
            elif key == "timeout_s":
                copy.timeout_s = float(value)
            elif key == "cost_budget_usd":
                copy.cost_budget_usd = float(value)
            elif key in {"concurrency"}:
                copy.concurrency = int(value)
            else:
                # Unknown per-request override lands in generation overrides.
                copy.generation.overrides[key] = value
        return copy


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
