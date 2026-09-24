"""Tests for the SQLite-backed JobTracker."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rag_core.jobs import JobStatus, PipelineJob
from rag_mass_inject.jobs import JobTracker


@pytest.fixture
def tracker(tmp_path: Path) -> JobTracker:
    return JobTracker(str(tmp_path / "checkpoints"))


async def test_create_returns_job_id(tracker: JobTracker) -> None:
    job_id = tracker.create("/path/to/source.txt")
    assert isinstance(job_id, str)
    assert len(job_id) > 0


async def test_get_returns_pipeline_job(tracker: JobTracker) -> None:
    source = "/path/to/source.txt"
    job_id = tracker.create(source)
    job = await tracker.get(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert job.kind == "mass_ingest"
    assert job.status == JobStatus.queued
    assert job.progress == 0.0
    assert job.error is None
    assert job.result_summary["source"] == source


async def test_get_unknown_returns_none(tracker: JobTracker) -> None:
    assert await tracker.get("nonexistent_id") is None


async def test_create_with_custom_kind(tracker: JobTracker) -> None:
    job_id = tracker.create("/source", kind="custom_kind")
    job = await tracker.get(job_id)
    assert job is not None
    assert job.kind == "custom_kind"


async def test_update_status_and_progress(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    now = datetime.now(UTC)
    await tracker.update(
        job_id,
        status=JobStatus.running,
        started_at=now,
        progress=0.5,
        processed_items=2,
        total_items=4,
    )
    job = await tracker.get(job_id)
    assert job is not None
    assert job.status == JobStatus.running
    assert job.progress == 0.5
    assert job.processed_items == 2
    assert job.total_items == 4
    assert job.started_at is not None


async def test_update_error_fields(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    await tracker.update(job_id, status=JobStatus.failed, error="boom", error_code="E123")
    job = await tracker.get(job_id)
    assert job is not None
    assert job.status == JobStatus.failed
    assert job.error == "boom"
    assert job.error_code == "E123"


async def test_update_result_summary(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    await tracker.update(job_id, result_summary={"foo": "bar", "count": 3})
    job = await tracker.get(job_id)
    assert job is not None
    assert job.result_summary["foo"] == "bar"
    assert job.result_summary["count"] == 3


async def test_checkpoint_and_is_checkpointed(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    chash = "abc123"
    assert not await tracker.is_checkpointed(job_id, chash)
    await tracker.checkpoint(job_id, "/source/file1.txt", chash)
    assert await tracker.is_checkpointed(job_id, chash)


async def test_checkpoint_idempotent(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    chash = "dedupe"
    await tracker.checkpoint(job_id, "/source/a.txt", chash)
    await tracker.checkpoint(job_id, "/source/a.txt", chash)  # second is a no-op
    assert await tracker.is_checkpointed(job_id, chash)

    # Verify only one checkpoint row exists.
    import aiosqlite

    async with aiosqlite.connect(tracker._db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE job_id = ? AND content_hash = ?",
            (job_id, chash),
        )
        count = (await cursor.fetchone())[0]
        await cursor.close()
    assert count == 1


async def test_checkpoint_isolates_per_job(tracker: JobTracker) -> None:
    job1 = tracker.create("/source1")
    job2 = tracker.create("/source2")
    chash = "shared_hash"
    await tracker.checkpoint(job1, "/source1/a.txt", chash)

    assert await tracker.is_checkpointed(job1, chash)
    assert not await tracker.is_checkpointed(job2, chash)


async def test_dead_letter(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    await tracker.dead_letter(job_id, "/source/bad.bin", "cannot parse", "unsupported_format")

    import aiosqlite

    async with aiosqlite.connect(tracker._db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM dead_letters WHERE job_id = ?", (job_id,))
        rows = await cursor.fetchall()
        await cursor.close()
    assert len(rows) == 1
    assert rows[0]["source"] == "/source/bad.bin"
    assert rows[0]["error"] == "cannot parse"
    assert rows[0]["error_code"] == "unsupported_format"


async def test_dead_letter_without_code(tracker: JobTracker) -> None:
    job_id = tracker.create("/source")
    await tracker.dead_letter(job_id, "/source/bad.txt", "generic error")

    import aiosqlite

    async with aiosqlite.connect(tracker._db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM dead_letters WHERE job_id = ?", (job_id,))
        rows = await cursor.fetchall()
        await cursor.close()
    assert len(rows) == 1
    assert rows[0]["error_code"] is None


async def test_list_returns_all_jobs(tmp_path: Path) -> None:
    tracker = JobTracker(str(tmp_path / "ck"))
    id1 = tracker.create("/a")
    id2 = tracker.create("/b")

    jobs = await tracker.list()
    assert len(jobs) == 2
    ids = {j.job_id for j in jobs}
    assert {id1, id2} == ids
    assert all(j.status == JobStatus.queued for j in jobs)


async def test_get_returns_pipeline_job_model(tracker: JobTracker) -> None:
    job_id = tracker.create("/source.txt")
    job = await tracker.get(job_id)
    assert isinstance(job, PipelineJob)
    assert job.job_id == job_id
    assert job.created_at is not None


async def test_checkpoint_dir_created(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "newdir" / "sub"
    assert not ckpt_dir.exists()
    JobTracker(str(ckpt_dir))
    assert ckpt_dir.exists()
    assert (ckpt_dir / "jobs.db").exists()


async def test_job_tracker_persists_across_instances(tmp_path: Path) -> None:
    ckpt_dir = str(tmp_path / "ck")
    tracker1 = JobTracker(ckpt_dir)
    job_id = tracker1.create("/source")

    # Re-create a tracker pointing at the same DB file.
    tracker2 = JobTracker(ckpt_dir)
    job = await tracker2.get(job_id)
    assert job is not None
    assert job.result_summary["source"] == "/source"
