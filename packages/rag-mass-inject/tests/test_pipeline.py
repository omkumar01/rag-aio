"""Integration tests for MassIngestor with offline mock services."""

from __future__ import annotations

from pathlib import Path

import mass_inject_fakes
import pytest
from rag_core import Document
from rag_core.jobs import JobStatus
from rag_mass_inject.config import MassInjectConfig
from rag_mass_inject.pipeline import MassIngestor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, **overrides: object) -> MassInjectConfig:
    """Build a MassInjectConfig pointing at *tmp_path* for persistence."""
    defaults: dict[str, object] = {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "dead_letter_dir": str(tmp_path / "dead_letters"),
        "resume": True,
        "max_workers": 2,
    }
    defaults.update(overrides)
    return MassInjectConfig(**defaults)  # type: ignore[arg-type]


def _write_text(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# (a) submit + wait returns documents
# ---------------------------------------------------------------------------


async def test_submit_returns_job_id(tmp_path: Path) -> None:
    services = mass_inject_fakes.make_services(tmp_path)
    ingestor = MassIngestor(_config(tmp_path), services)
    job_id = ingestor.submit(str(tmp_path))
    assert isinstance(job_id, str)
    assert len(job_id) > 0


async def test_submit_and_wait_returns_documents(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc1.txt", "alpha content " * 30)
    _write_text(tmp_path, "doc2.txt", "beta content " * 30)
    _write_text(tmp_path, "doc3.txt", "gamma content " * 30)

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)
    for d in docs:
        assert d.content_hash


async def test_status_after_completion(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc.txt", "content " * 30)

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    await ingestor.wait(job_id)

    job = await ingestor.status(job_id)
    assert job.status == JobStatus.completed
    assert job.total_items == 1
    assert job.processed_items == 1
    assert job.progress == 1.0


# ---------------------------------------------------------------------------
# (b) dedup skip (same file ingested twice — second skipped)
# ---------------------------------------------------------------------------


async def test_dedup_skips_duplicate_within_run(tmp_path: Path) -> None:
    """A file-list with the same path twice → second is deduped (is_new=False)."""
    doc = _write_text(tmp_path, "doc.txt", "dedup me " * 30)
    list_file = tmp_path / "inputs.list"
    list_file.write_text(f"{doc}\n{doc}\n", encoding="utf-8")

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(list_file))
    docs = await ingestor.wait(job_id)

    assert len(docs) == 1


async def test_dedup_two_files_same_content_different_paths(tmp_path: Path) -> None:
    """Two files with identical content but different paths → both processed.

    Because ``Document.content_hash`` includes the source_uri, distinct paths
    produce distinct hashes and both documents are ingested (the DedupIndex
    only catches exact source-level duplicates within one run).
    """
    _write_text(tmp_path, "a.txt", "same content " * 20)
    _write_text(tmp_path, "b.txt", "same content " * 20)

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    assert len(docs) == 2


# ---------------------------------------------------------------------------
# (c) resume (after a run completes, re-running skips checkpointed files)
# ---------------------------------------------------------------------------


async def test_resume_skips_checkpointed_files(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc1.txt", "first doc " * 30)
    _write_text(tmp_path, "doc2.txt", "second doc " * 30)

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))

    docs1 = await ingestor.wait(job_id)
    assert len(docs1) == 2

    # Second run: same job_id, resume=True → all files checkpointed → empty.
    docs2 = await ingestor.wait(job_id)
    assert docs2 == []


async def test_resume_disabled_reprocesses_all(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc1.txt", "first doc " * 30)

    config = _config(tmp_path, resume=False)
    ingestor = MassIngestor(config, mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))

    docs1 = await ingestor.wait(job_id)
    assert len(docs1) == 1

    # With resume=False, the same file is reprocessed (checkpointed or not).
    docs2 = await ingestor.wait(job_id)
    assert len(docs2) == 1


# ---------------------------------------------------------------------------
# (d) dead-letter (a .bin poison file → copied to dead_letter_dir, not in results)
# ---------------------------------------------------------------------------


async def test_dead_letter_poison_file(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc.txt", "good content " * 30)
    (tmp_path / "poison.bin").write_bytes(b"\x00\x01\x02\x03not-a-document")

    dl_dir = tmp_path / "dead_letters"
    config = _config(tmp_path, dead_letter_dir=str(dl_dir))
    ingestor = MassIngestor(config, mass_inject_fakes.make_services(tmp_path))

    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    # Only the .txt file is in results; .bin was dead-lettered.
    assert len(docs) == 1
    assert docs[0].source_uri.endswith("doc.txt")

    # The .bin file was copied to the dead-letter directory.
    assert (dl_dir / "poison.bin").exists()

    # Job completed (not failed) — dead-letters are non-fatal.
    job = await ingestor.status(job_id)
    assert job.status == JobStatus.completed


async def test_dead_letter_records_in_tracker(tmp_path: Path) -> None:
    (tmp_path / "poison.bin").write_bytes(b"\x00\x01\x02\x03")

    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    await ingestor.wait(job_id)

    # Verify a dead-letter record exists in the tracker.
    import aiosqlite

    db_path = str(tmp_path / "checkpoints" / "jobs.db")
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM dead_letters WHERE job_id = ?", (job_id,))
        rows = await cursor.fetchall()
        await cursor.close()
    assert len(rows) == 1
    assert "poison.bin" in rows[0]["source"]


# ---------------------------------------------------------------------------
# (e) recursive directory
# ---------------------------------------------------------------------------


async def test_recursive_directory(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc1.txt", "top level " * 30)
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_text(sub, "doc2.md", "nested level " * 30)
    deep = sub / "deep"
    deep.mkdir()
    _write_text(deep, "doc3.txt", "deep level " * 30)

    config = _config(tmp_path, recursive=True)
    ingestor = MassIngestor(config, mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    assert len(docs) == 3
    sources = sorted(d.source_uri for d in docs)
    assert any("doc1.txt" in s for s in sources)
    assert any("doc2.md" in s for s in sources)
    assert any("doc3.txt" in s for s in sources)


async def test_non_recursive_directory_skips_subdirs(tmp_path: Path) -> None:
    _write_text(tmp_path, "doc1.txt", "top level " * 30)
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_text(sub, "doc2.md", "nested level " * 30)

    ingestor = MassIngestor(
        _config(tmp_path, recursive=False), mass_inject_fakes.make_services(tmp_path)
    )
    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    assert len(docs) == 1
    assert docs[0].source_uri.endswith("doc1.txt")


# ---------------------------------------------------------------------------
# Job tracking edge cases
# ---------------------------------------------------------------------------


async def test_wait_unknown_job_raises(tmp_path: Path) -> None:
    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    with pytest.raises(Exception, match="not found"):
        await ingestor.wait("nonexistent_job_id")


async def test_zero_files_directory(tmp_path: Path) -> None:
    ingestor = MassIngestor(_config(tmp_path), mass_inject_fakes.make_services(tmp_path))
    job_id = ingestor.submit(str(tmp_path))
    docs = await ingestor.wait(job_id)

    assert docs == []
    job = await ingestor.status(job_id)
    assert job.status == JobStatus.completed
    assert job.total_items == 0


async def test_stage_worker_semaphores(tmp_path: Path) -> None:
    """Verify StageWorker correctly bounds concurrency per stage."""

    from rag_mass_inject.workers import StageWorker

    config = _config(tmp_path)
    worker = StageWorker(config)

    # Each stage should have a semaphore with the configured value.
    assert worker.semaphore("read") is not None
    assert worker.semaphore("read")._value == 4
    assert worker.semaphore("parse") is not None
    assert worker.semaphore("parse")._value == 2

    # Unknown stage → no semaphore (unbounded).
    assert worker.semaphore("unknown_stage") is None


async def test_bounded_queue_put_get_join(tmp_path: Path) -> None:
    """Verify BoundedQueue basic operations."""
    from rag_mass_inject.workers import BoundedQueue

    queue: BoundedQueue = BoundedQueue(maxsize=2)
    assert queue.empty()

    await queue.put("a")
    await queue.put("b")

    item1 = await queue.get()
    assert item1 == "a"
    assert queue.qsize() == 1

    item2 = await queue.get()
    assert item2 == "b"

    queue.task_done()
    queue.task_done()

    await queue.join()  # should not block
    assert queue.empty()
