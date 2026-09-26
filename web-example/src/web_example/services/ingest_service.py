"""Bulk-ingest service: UploadJob bookkeeping and the background runner.

Routes create an :class:`~web_example.db.UploadJob` row, then hand off to
:func:`spawn_run_job` which runs :func:`run_job` as a fire-and-forget task.

- ``files`` jobs ingest uploaded bytes one-by-one via ``RagManager.ingest_bytes``.
- ``directory`` and sitemap ``url`` jobs run through the real
  :class:`rag_mass_inject.MassIngestor` pipeline (bounded queue, checkpoints),
  bound to the target pool so writes land in the right vector namespace.
- A single-page ``url`` is ingested directly through the pool's pipeline.

The runner degrades gracefully when ``app.state.rag`` is not initialised.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol, cast

from rag_core.jobs import JobStatus
from rag_mass_inject import SUPPORTED_EXTENSIONS as _RAG_SUPPORTED_EXTENSIONS
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.db import UploadJob
from web_example.services.doc_service import register_document

__all__ = [
    "BACKGROUND_TASKS",
    "MAX_FILES_PER_JOB",
    "SUPPORTED_EXTENSIONS",
    "create_job",
    "get_job",
    "is_supported_name",
    "list_jobs",
    "run_job",
    "spawn_run_job",
    "update_job_progress",
]

#: Re-exported from rag-mass-inject (.pdf/.txt/.md/.html/.htm/.docx/.xlsx/.pptx/.eml).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_RAG_SUPPORTED_EXTENSIONS)

#: Hard cap on the number of files a single job may ingest.
MAX_FILES_PER_JOB = 500

#: Maximum length of the persisted per-error ``detail`` string.
_DETAIL_LIMIT = 2000

#: Seconds between MassIngestor status polls while a bulk job runs.
_POLL_INTERVAL_S = 1.0

#: MassIngestor PipelineJob status → UploadJob status.
_STATUS_MAP: dict[str, str] = {
    JobStatus.queued: "queued",
    JobStatus.running: "running",
    JobStatus.completed: "done",
    JobStatus.failed: "failed",
    JobStatus.cancelled: "failed",
}

#: Live background tasks; keeping references stops GC from cancelling them and
#: gives tests a handle to await job completion.
BACKGROUND_TASKS: set[asyncio.Task[None]] = set()

logger = logging.getLogger(__name__)


class _SupportsIngestBytes(Protocol):
    """Structural type for the per-user RAG engine hook (wired by app.state.rag)."""

    async def ingest_bytes(
        self,
        session: AsyncSession,
        user_id: str,
        name: str,
        data: bytes,
        *,
        global_pool: bool = False,
    ) -> object: ...


def is_supported_name(name: str) -> bool:
    """True when *name* carries one of the supported file extensions."""
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


async def create_job(
    session: AsyncSession, user_id: str, target: str, kind: str, total: int
) -> UploadJob:
    """Insert and commit a new UploadJob row in ``queued`` state."""
    job = UploadJob(user_id=user_id, target=target, kind=kind, total=total, status="queued")
    session.add(job)
    await session.commit()
    return job


async def update_job_progress(
    session: AsyncSession,
    job_id: str,
    *,
    total: int | None = None,
    processed: int | None = None,
    failed: int | None = None,
    status: str | None = None,
    detail: str | None = None,
) -> UploadJob | None:
    """Patch selected columns of one job (primary-key lookup, then setattr)."""
    job = await session.get(UploadJob, job_id)
    if job is None:
        return None
    if total is not None:
        job.total = total
    if processed is not None:
        job.processed = processed
    if failed is not None:
        job.failed = failed
    if status is not None:
        job.status = status
    if detail is not None:
        job.detail = detail
    await session.commit()
    return job


async def get_job(session: AsyncSession, user_id: str, job_id: str) -> UploadJob | None:
    """Fetch one job by primary key, enforcing ownership by *user_id*."""
    job = await session.get(UploadJob, job_id)
    if job is None or job.user_id != user_id:
        return None
    return job


async def list_jobs(session: AsyncSession, user_id: str, limit: int = 20) -> list[UploadJob]:
    """Return the user's most recent jobs, newest first (bound-param SQL)."""
    result = await session.execute(
        text("SELECT id FROM upload_jobs WHERE user_id = :uid ORDER BY created_at DESC LIMIT :lim"),
        {"uid": user_id, "lim": limit},
    )
    jobs: list[UploadJob] = []
    for row_id in result.scalars():
        job = await session.get(UploadJob, row_id)
        if job is not None:
            jobs.append(job)
    return jobs


def spawn_run_job(
    app: Any,
    job_id: str,
    user_id: str,
    target: str,
    *,
    files: list[tuple[str, bytes]] | None = None,
    directory: str | None = None,
    url: str | None = None,
) -> asyncio.Task[None]:
    """Start :func:`run_job` as a fire-and-forget background task."""
    task = asyncio.create_task(
        run_job(app, job_id, user_id, target, files=files, directory=directory, url=url)
    )
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return task


async def run_job(
    app: Any,
    job_id: str,
    user_id: str,
    target: str,
    files: list[tuple[str, bytes]] | None = None,
    directory: str | None = None,
    url: str | None = None,
) -> None:
    """Background worker for one UploadJob. Never raises."""
    try:
        await _run_job_inner(app, job_id, user_id, target, files, directory, url)
    except Exception:
        logger.exception("upload job %s crashed", job_id)
        try:
            maker: async_sessionmaker[AsyncSession] = app.state.sessionmaker
            async with maker() as session:
                await update_job_progress(
                    session, job_id, status="failed", detail="unexpected server error"
                )
        except Exception:
            logger.exception("could not mark upload job %s as failed", job_id)


async def _run_job_inner(
    app: Any,
    job_id: str,
    user_id: str,
    target: str,
    files: list[tuple[str, bytes]] | None,
    directory: str | None,
    url: str | None,
) -> None:
    maker: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    rag = cast("_SupportsIngestBytes | None", getattr(app.state, "rag", None))

    if directory is not None:
        if rag is None or not hasattr(rag, "mass_ingestor"):
            await _mark_failed(maker, job_id, "RAG engine not initialised")
            return
        await _run_mass_source(maker, rag, job_id, user_id, target, directory)
        return
    if url is not None:
        if rag is None:
            await _mark_failed(maker, job_id, "RAG engine not initialised")
        elif _looks_like_sitemap(url):
            if not hasattr(rag, "mass_ingestor"):
                await _mark_failed(maker, job_id, "RAG engine not initialised")
                return
            await _run_mass_source(maker, rag, job_id, user_id, target, url)
        else:
            await _run_single_url(maker, rag, job_id, user_id, target, url)
        return

    if rag is None or not hasattr(rag, "ingest_bytes"):
        async with maker() as session:
            await update_job_progress(
                session, job_id, status="failed", detail="RAG engine not initialised"
            )
        return

    processed = 0
    failed = 0
    errors: list[str] = []
    async with maker() as session:
        await update_job_progress(session, job_id, status="running")
    for name, data in list(files or []):
        if not is_supported_name(name):
            failed += 1
            errors.append(f"{name}: unsupported file type")
        else:
            try:
                async with maker() as session:
                    doc = await rag.ingest_bytes(
                        session, user_id, name, data, global_pool=(target == "global")
                    )
                    await _register_rag_document(
                        session,
                        doc,
                        owner_id=user_id,
                        visibility=target,
                        name=name,
                        ext=Path(name).suffix.lower(),
                        size_bytes=len(data),
                    )
                processed += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{name}: {exc}")
                logger.warning("ingest failed for %s in job %s: %s", name, job_id, exc)
        async with maker() as session:
            await update_job_progress(session, job_id, processed=processed, failed=failed)
    final_status = "done" if failed == 0 else "done_with_errors"
    async with maker() as session:
        await update_job_progress(
            session, job_id, status=final_status, detail=_clip(" | ".join(errors))
        )


def _looks_like_sitemap(url: str) -> bool:
    """True for XML sitemap URLs; anything else is treated as a single page."""
    lowered = url.lower()
    path = lowered.split("?", 1)[0].split("#", 1)[0]
    return path.endswith(".xml") or "sitemap" in path


async def _run_mass_source(
    maker: async_sessionmaker[AsyncSession],
    rag: Any,
    job_id: str,
    user_id: str,
    target: str,
    source: str,
) -> None:
    """Ingest a directory or sitemap through the rag-mass-inject pipeline.

    Polls the MassIngestor's own job record while it runs so the web-side
    UploadJob row shows live progress, then registers every resulting
    document in the web registry under the target pool's visibility.
    """
    global_pool = target == "global"
    async with maker() as session:
        mass = await rag.mass_ingestor(session, user_id, global_pool=global_pool)
        mass_job_id = mass.submit(source)
        await update_job_progress(session, job_id, status="running")

    waiter = asyncio.create_task(mass.wait(mass_job_id))
    try:
        while True:
            done, _ = await asyncio.wait({waiter}, timeout=_POLL_INTERVAL_S)
            async with maker() as session:
                pipeline_job = await mass.status(mass_job_id)
                await update_job_progress(
                    session,
                    job_id,
                    total=pipeline_job.total_items,
                    processed=pipeline_job.processed_items,
                    status=_STATUS_MAP.get(str(pipeline_job.status), "running"),
                )
            if waiter in done:
                break
        documents = waiter.result()
    except Exception as exc:
        logger.warning("mass ingestion failed for job %s: %s", job_id, exc)
        await _mark_failed(maker, job_id, _clip(str(exc)))
        return

    ingested = 0
    errors: list[str] = []
    async with maker() as session:
        for doc in documents:
            try:
                await _register_rag_document(
                    session,
                    doc,
                    owner_id=None if global_pool else user_id,
                    visibility=target,
                    name=_source_display_name(str(getattr(doc, "source_uri", "") or "")),
                )
                ingested += 1
            except Exception as exc:
                errors.append(f"{getattr(doc, 'source_uri', '?')}: {exc}")
                logger.warning("registry registration failed in job %s: %s", job_id, exc)
        detail = f"{ingested} document(s) ingested"
        if errors:
            detail = _clip(detail + " | " + " | ".join(errors))
        await update_job_progress(
            session,
            job_id,
            status="done" if not errors else "done_with_errors",
            detail=detail,
        )


async def _run_single_url(
    maker: async_sessionmaker[AsyncSession],
    rag: Any,
    job_id: str,
    user_id: str,
    target: str,
    url: str,
) -> None:
    """Ingest one web page through the pool's (scheme-routing) pipeline."""
    global_pool = target == "global"
    async with maker() as session:
        await update_job_progress(session, job_id, status="running", total=1)
        try:
            doc = await _ingest_single_source(rag, session, user_id, url, global_pool)
            await _register_rag_document(
                session,
                doc,
                owner_id=None if global_pool else user_id,
                visibility=target,
                name=_source_display_name(str(getattr(doc, "source_uri", "") or url)),
            )
        except Exception as exc:
            logger.warning("single-URL ingestion failed for job %s: %s", job_id, exc)
            await update_job_progress(
                session, job_id, status="failed", failed=1, detail=_clip(str(exc))
            )
            return
        await update_job_progress(session, job_id, processed=1, status="done", detail="1 document")


async def _ingest_single_source(
    rag: Any, session: AsyncSession, user_id: str, url: str, global_pool: bool
) -> Any:
    """Ingest one URL via the pool's RAG facade (seam for offline tests)."""
    if global_pool:
        pool_rag = await rag.get_global(session)
    else:
        pool_rag = await rag.get(session, user_id)
    return await pool_rag.ingest(url)


async def _register_rag_document(
    session: AsyncSession,
    doc: Any,
    *,
    owner_id: str | None,
    visibility: str,
    name: str,
    ext: str = "",
    size_bytes: int = 0,
) -> None:
    """Mirror an ingested rag Document into the web document registry."""
    source_uri = str(getattr(doc, "source_uri", "") or "")
    if not ext:
        suffix = Path(source_uri.split("?", 1)[0]).suffix.lower()
        is_web = source_uri.startswith(("http://", "https://"))
        known = suffix in SUPPORTED_EXTENSIONS or suffix in {".html", ".htm", ".xml"}
        if known:
            ext = suffix
        elif is_web:
            ext = ".html"
    size = size_bytes
    if size == 0 and source_uri and not source_uri.startswith(("http://", "https://")):
        size = await asyncio.to_thread(_file_size, source_uri)
    await register_document(
        session,
        str(doc.id),
        owner_id=owner_id,
        visibility=visibility,
        name=name or Path(source_uri).name or source_uri,
        source_uri=source_uri,
        ext=ext,
        size_bytes=size,
        content_hash=str(getattr(doc, "content_hash", "") or ""),
    )


def _file_size(source_uri: str) -> int:
    """Size of a local source file, or 0 (sync helper; call via to_thread)."""
    path = Path(source_uri)
    return path.stat().st_size if path.is_file() else 0


def _source_display_name(source_uri: str) -> str:
    """A human-readable name for a discovered source (file name or URL tail)."""
    if not source_uri:
        return "untitled"
    if source_uri.startswith(("http://", "https://")):
        tail = source_uri.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
        return tail or source_uri.split("//", 1)[1]
    return Path(source_uri).name


async def _mark_failed(maker: async_sessionmaker[AsyncSession], job_id: str, detail: str) -> None:
    async with maker() as session:
        await update_job_progress(session, job_id, status="failed", detail=detail)


def _clip(detail: str) -> str:
    """Truncate *detail* to roughly ``_DETAIL_LIMIT`` characters."""
    if len(detail) <= _DETAIL_LIMIT:
        return detail
    return detail[: _DETAIL_LIMIT - 3] + "..."
