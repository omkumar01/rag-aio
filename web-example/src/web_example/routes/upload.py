"""Mass upload / bulk ingestion routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from rag_core.errors import CrawlError
from rag_mass_inject import discover_directory
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.auth.deps import get_current_user
from web_example.db import UploadJob, User
from web_example.services import ingest_service

CurrentUser = Annotated[User, Depends(get_current_user)]

try:  # SSRF guard, optional per contract (workspace member provides it).
    from rag_doc_handler.security import validate_public_url
except ImportError:  # pragma: no cover
    validate_public_url = None  # type: ignore[assignment]

__all__ = ["router"]

router = APIRouter(tags=["upload"])

_VALID_TARGETS = ("user", "global")


def _maker(request: Request) -> async_sessionmaker[AsyncSession]:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return maker


def _resolve_target(target: str, user: User) -> str:
    if target not in _VALID_TARGETS:
        raise HTTPException(status_code=400, detail="target must be 'user' or 'global'.")
    if target == "global" and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required for global uploads.")
    return target


def _job_payload(job: UploadJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "target": job.target,
        "kind": job.kind,
        "status": job.status,
        "total": job.total,
        "processed": job.processed,
        "failed": job.failed,
        "detail": job.detail,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


def _count_supported_files(directory: str) -> int:
    """Count supported files at the top level of *directory* (sync, run in a thread)."""
    return sum(
        1 for entry in discover_directory(directory) if ingest_service.is_supported_name(entry.name)
    )


def _is_directory(path: str) -> bool:
    """Existence + kind check for *path* (sync, run in a thread)."""
    return Path(path).is_dir()


async def _start_job(
    request: Request,
    user: User,
    *,
    target: str,
    kind: str,
    total: int,
    files: list[tuple[str, bytes]] | None = None,
    directory: str | None = None,
    url: str | None = None,
) -> Response:
    """Persist the job row, spawn the background runner, redirect to the page."""
    async with _maker(request)() as session:
        job = await ingest_service.create_job(session, user.id, target, kind, total)
    ingest_service.spawn_run_job(
        request.app, job.id, user.id, target, files=files, directory=directory, url=url
    )
    return RedirectResponse("/upload", status_code=303)


@router.get("/upload")
async def upload_page(request: Request, user: CurrentUser) -> Response:
    templates = request.app.state.templates
    return cast(
        Response,
        templates.TemplateResponse(
            request,
            "upload.html",
            {
                "is_admin": user.is_admin,
                "max_files": ingest_service.MAX_FILES_PER_JOB,
                "supported_extensions": sorted(ingest_service.SUPPORTED_EXTENSIONS),
            },
        ),
    )


@router.post("/upload/files")
async def upload_files(
    request: Request,
    user: CurrentUser,
    files: Annotated[list[UploadFile] | None, File(description="One or more files.")] = None,
    target: Annotated[str, Form()] = "user",
) -> Response:
    target = _resolve_target(target, user)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    limit = ingest_service.MAX_FILES_PER_JOB
    if len(files) > limit:
        raise HTTPException(status_code=400, detail=f"Too many files: {len(files)} > {limit}.")
    bad = [
        f.filename or "<unnamed>"
        for f in files
        if not ingest_service.is_supported_name(f.filename or "")
    ]
    if bad:
        shown = ", ".join(bad[:20])
        raise HTTPException(status_code=400, detail="Unsupported file type(s): " + shown)
    payload: list[tuple[str, bytes]] = []
    for f in files:
        payload.append((f.filename or "upload.bin", await f.read()))
        await f.close()
    return await _start_job(
        request, user, target=target, kind="files", total=len(payload), files=payload
    )


@router.post("/upload/directory")
async def upload_directory(
    request: Request,
    user: CurrentUser,
    directory: Annotated[str, Form()],
    target: Annotated[str, Form()] = "user",
) -> Response:
    target = _resolve_target(target, user)
    directory = directory.strip()
    if not directory:
        raise HTTPException(status_code=400, detail="Directory path is required.")
    if not await asyncio.to_thread(_is_directory, directory):
        raise HTTPException(status_code=400, detail="Not a directory: " + directory)
    supported_count = await asyncio.to_thread(_count_supported_files, directory)
    if supported_count == 0:
        raise HTTPException(status_code=400, detail="No supported files in directory.")
    return await _start_job(
        request, user, target=target, kind="directory", total=supported_count, directory=directory
    )


@router.post("/upload/url")
async def upload_url(
    request: Request,
    user: CurrentUser,
    url: Annotated[str, Form()],
    target: Annotated[str, Form()] = "user",
) -> Response:
    target = _resolve_target(target, user)
    url = url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    if validate_public_url is not None:
        try:
            url = validate_public_url(url)
        except CrawlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _start_job(request, user, target=target, kind="url", total=0, url=url)


@router.get("/upload/jobs")
async def upload_jobs_partial(request: Request, user: CurrentUser) -> Response:
    async with _maker(request)() as session:
        jobs = await ingest_service.list_jobs(session, user.id)
    return cast(
        Response,
        request.app.state.templates.TemplateResponse(
            request, "partials/upload_jobs.html", {"jobs": jobs}
        ),
    )


@router.get("/upload/api/jobs")
async def upload_jobs_api(request: Request, user: CurrentUser) -> list[dict[str, Any]]:
    async with _maker(request)() as session:
        jobs = await ingest_service.list_jobs(session, user.id)
    return [_job_payload(job) for job in jobs]
