"""Evaluation models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from .base import RagBaseModel
from .ids import new_id


class MetricResult(RagBaseModel):
    """One computed metric (e.g. recall@10 = 0.75)."""

    name: str
    value: float
    k: int | None = Field(default=None, ge=1)
    details: dict[str, Any] = Field(default_factory=dict)


class QueryEvaluation(RagBaseModel):
    """Per-query metric breakdown."""

    query_id: str
    metrics: list[MetricResult] = Field(default_factory=list)
    failure_class: str | None = None


class EvaluationResult(RagBaseModel):
    """A reproducible evaluation run."""

    run_id: str = Field(default_factory=new_id)
    dataset: str
    config_hash: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    metrics: list[MetricResult] = Field(default_factory=list)
    per_query: list[QueryEvaluation] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
