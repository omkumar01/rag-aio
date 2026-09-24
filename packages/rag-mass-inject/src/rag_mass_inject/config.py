"""Configuration for the mass-injection batch pipeline.

A :class:`MassInjectConfig` controls how a :class:`~rag_mass_inject.pipeline.MassIngestor`
discovers, processes and persists sources. It is a strict Pydantic model
(``extra="forbid"`` via :class:`rag_core.base.RagBaseModel`) so configuration
drift fails loudly.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from rag_core.base import RagBaseModel


def _default_concurrency() -> dict[str, int]:
    """Default per-stage semaphore capacities."""
    return {"read": 4, "parse": 2, "ocr": 2, "chunk": 2, "embed": 4, "index": 4}


__all__ = ["MassInjectConfig"]


class MassInjectConfig(RagBaseModel):
    """Tuning knobs for bulk ingestion.

    Attributes:
        max_workers: Total concurrent files being processed at once.
        recursive: Whether to recurse into sub-directories when discovering
            files from a directory source.
        concurrency: Per-stage semaphore limits keyed by stage name
            (``read``, ``parse``, ``ocr``, ``chunk``, ``embed``, ``index``).
        checkpoint_dir: Directory for the SQLite job/checkpoint database.
        dead_letter_dir: Directory where dead-lettered files are copied.
        resume: When ``True``, files whose content hash already has a
            checkpoint from a prior run are skipped.
        timeout_s: Optional end-to-end timeout per ``wait()`` call, in seconds.
            ``None`` means no limit.
    """

    max_workers: int = Field(default=4, ge=1)
    recursive: bool = False
    concurrency: dict[str, int] = Field(default_factory=_default_concurrency)
    checkpoint_dir: str = "./data/mass-inject"
    dead_letter_dir: str = "./data/mass-inject/dead-letter"
    resume: bool = True
    timeout_s: float | None = None

    @classmethod
    def local_default(cls) -> MassInjectConfig:
        """Convenience profile suitable for local development."""
        return cls()

    def stage_limit(self, stage: str) -> int:
        """Return the configured concurrency limit for *stage*, defaulting to 1."""
        return self.concurrency.get(stage, 1)

    def dump_for_log(self) -> dict[str, Any]:
        """Return a log-safe summary (no secrets — there are none here)."""
        return self.model_dump(mode="json")
