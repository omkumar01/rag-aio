"""Job models for long-running operations (ADR-0007)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from .base import RagBaseModel
from .ids import new_id


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class PipelineJob(RagBaseModel):
    """Status record for a long-running pipeline operation."""

    job_id: str = Field(default_factory=new_id)
    kind: str
    status: JobStatus = JobStatus.queued
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: str | None = None
    total_items: int | None = Field(default=None, ge=0)
    processed_items: int | None = Field(default=None, ge=0)
    error: str | None = None
    error_code: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    checkpoints: dict[str, Any] = Field(default_factory=dict)
