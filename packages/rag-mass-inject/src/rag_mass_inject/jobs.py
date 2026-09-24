"""SQLite-backed job and checkpoint store for mass-injection runs.

A :class:`JobTracker` persists long-running job state (``PipelineJob`` rows),
per-file processing checkpoints (content-hash based dedup for resume), and
dead-letter records.  The ``create``/``_init_schema`` paths use the standard
library :mod:`sqlite3` so they can run from synchronous code (``submit``);
all other operations use :mod:`aiosqlite` so they compose with the async
file-processing pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from rag_core.ids import new_id
from rag_core.jobs import JobStatus, PipelineJob

__all__ = ["JobTracker"]

_DEFAULT_KIND = "mass_ingest"


def _now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp stored in SQLite, returning ``None`` if empty."""
    if not value:
        return None
    return datetime.fromisoformat(value)


class JobTracker:
    """SQLite job tracker with checkpoints and dead-letter support.

    Parameters
    ----------
    checkpoint_dir:
        Directory that will hold ``jobs.db``.  Created (with ``parents=True``)
        if it does not exist.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self._dir / "jobs.db")
        # Schema is created synchronously so it is ready before any async call.
        self._init_schema()

    # -- schema ----------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the jobs / checkpoints / dead_letters tables if absent."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id              TEXT PRIMARY KEY,
                    kind            TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'queued',
                    progress        REAL NOT NULL DEFAULT 0.0,
                    stage           TEXT,
                    total_items     INTEGER,
                    processed_items INTEGER,
                    error           TEXT,
                    error_code      TEXT,
                    created_at      TEXT NOT NULL,
                    started_at      TEXT,
                    finished_at     TEXT,
                    result_summary_json TEXT NOT NULL DEFAULT '{}',
                    source          TEXT
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id         TEXT NOT NULL,
                    source         TEXT NOT NULL,
                    content_hash   TEXT NOT NULL,
                    processed_at   TEXT NOT NULL,
                    PRIMARY KEY (job_id, content_hash)
                );
                CREATE TABLE IF NOT EXISTS dead_letters (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id         TEXT NOT NULL,
                    source         TEXT NOT NULL,
                    error          TEXT NOT NULL,
                    error_code     TEXT,
                    ts             TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    # -- synchronous create (called from sync submit) --------------------------

    def create(self, source: str, kind: str = _DEFAULT_KIND) -> str:
        """Insert a new queued job and return its id.

        Synchronous so it can be called from synchronous code (e.g.
        :meth:`~rag_mass_inject.pipeline.MassIngestor.submit`).
        """
        job_id = new_id()
        now = _now_iso()
        result_summary = {"source": source}
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, progress, stage,
                    total_items, processed_items, error, error_code,
                    created_at, started_at, finished_at,
                    result_summary_json, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    JobStatus.queued.value,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    now,
                    None,
                    None,
                    json.dumps(result_summary),
                    source,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return job_id

    # -- async helpers ---------------------------------------------------------

    async def get(self, job_id: str) -> PipelineJob | None:
        """Return the :class:`PipelineJob` for *job_id*, or ``None``."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        return self._row_to_job(row)

    async def list(self) -> list[PipelineJob]:
        """Return all jobs ordered by creation time (oldest first)."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at ASC")
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._row_to_job(row) for row in rows]

    async def update(self, job_id: str, **fields: object) -> None:
        """Partially update a job row.

        Recognised field names are mapped to SQLite columns.  ``datetime``
        values are serialised to ISO-8601; ``dict`` values (e.g.
        ``result_summary``) to JSON.
        """
        if not fields:
            return
        assignments: list[str] = []
        values: list[object] = []
        for key, value in fields.items():
            col = _FIELD_TO_COLUMN.get(key, key)
            assignments.append(f"{col} = ?")
            if key in ("created_at", "started_at", "finished_at"):
                values.append(_serialize_dt(value))
            elif key == "result_summary":
                values.append(json.dumps(value))
            elif key == "status" and isinstance(value, JobStatus):
                values.append(value.value)
            else:
                values.append(value)
        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?"
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(sql, values)
            await db.commit()

    async def checkpoint(self, job_id: str, source: str, content_hash: str) -> None:
        """Record that *source* (hash *content_hash*) was processed for *job_id*.

        ``INSERT OR IGNORE`` makes this safe under concurrent workers: the
        first writer wins and duplicates are silently dropped.
        """
        now = _now_iso()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO checkpoints (job_id, source, content_hash, processed_at) "
                "VALUES (?, ?, ?, ?)",
                (job_id, source, content_hash, now),
            )
            await db.commit()

    async def is_checkpointed(self, job_id: str, content_hash: str) -> bool:
        """Return ``True`` if *content_hash* was already checkpointed for *job_id*."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM checkpoints WHERE job_id = ? AND content_hash = ?",
                (job_id, content_hash),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def dead_letter(
        self,
        job_id: str,
        source: str,
        error: str,
        error_code: str | None = None,
    ) -> None:
        """Insert a dead-letter record for a file that could not be processed."""
        now = _now_iso()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO dead_letters (job_id, source, error, error_code, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, source, error, error_code, now),
            )
            await db.commit()

    # -- row mapping -----------------------------------------------------------

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> PipelineJob:
        """Reconstruct a :class:`PipelineJob` from a SQLite row."""
        result_summary: dict[str, object] = json.loads(row["result_summary_json"]) or {}
        # Merge the dedicated source column into result_summary for callers
        # that expect it on the PipelineJob model.
        source_val = row["source"]
        if source_val and "source" not in result_summary:
            result_summary["source"] = source_val
        return PipelineJob(
            job_id=row["id"],
            kind=row["kind"],
            status=JobStatus(row["status"]),
            progress=float(row["progress"]) if row["progress"] is not None else 0.0,
            stage=row["stage"],
            total_items=row["total_items"],
            processed_items=row["processed_items"],
            error=row["error"],
            error_code=row["error_code"],
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
            result_summary=result_summary,
        )


# Maps Python-side field names to SQLite column names for the ``update`` helper.
_FIELD_TO_COLUMN: dict[str, str] = {
    "job_id": "id",
    "result_summary": "result_summary_json",
}


def _serialize_dt(value: object) -> str | None:
    """Coerce a datetime (or ISO string) to a SQLite-friendly ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
