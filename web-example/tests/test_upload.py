"""Route-level tests for the mass-upload (bulk ingestion) feature.

The tests stub ``app.state.rag`` with a tiny fake exposing ``ingest_bytes``
so they pass regardless of whether the parallel RagManager track has landed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.routes import upload as upload_routes
from web_example.services import ingest_service


class _FakeDoc:
    """Stand-in for a RAG document."""

    def __init__(self, name: str) -> None:
        self.id = f"doc-{name}"
        self.name = name


class _FakeRag:
    """Minimal RagManager double that records ``ingest_bytes`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def ingest_bytes(
        self,
        session: object,
        user_id: str,
        name: str,
        data: bytes,
        *,
        global_pool: bool = False,
    ) -> _FakeDoc:
        self.calls.append((name, global_pool))
        return _FakeDoc(name)


async def _drain_background_jobs() -> None:
    """Await every spawn_run_job task spawned so far."""
    for task in list(ingest_service.BACKGROUND_TASKS):
        await task


async def _user_id(app: Any, username: str) -> str:
    maker: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    async with maker() as db_session:
        result = await db_session.execute(
            text("SELECT id FROM users WHERE username = :name"), {"name": username}
        )
        return str(result.scalar_one())


async def test_upload_page_requires_login(client: AsyncClient) -> None:
    response = await client.get("/upload", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


async def test_upload_page_renders_for_user(client: AsyncClient, register: Any) -> None:
    await register("alice")
    response = await client.get("/upload")
    assert response.status_code == 200
    assert "uploadPanel" in response.text
    assert 'hx-get="/upload/jobs"' in response.text
    assert "every 2s" in response.text


def test_x_data_attribute_is_not_broken_by_tojson() -> None:
    """Jinja's ``tojson`` does not escape double quotes, so the x-data value must
    live in a single-quoted attribute — a double-quoted one would terminate at
    the first JSON string and Alpine would never bind the component."""
    content = Path(
        Path(__file__).resolve().parents[1] / "src/web_example/templates/upload.html"
    ).read_text(encoding="utf-8")
    assert "x-data='" in content
    assert 'x-data="' not in content


def test_page_scripts_load_before_alpine() -> None:
    """Deferred scripts run in document order and Alpine starts scanning the
    DOM as soon as it executes, so page components must load before Alpine."""
    base = Path(
        Path(__file__).resolve().parents[1] / "src/web_example/templates/base.html"
    ).read_text(encoding="utf-8")
    head = base.split("</head>", 1)[0]
    htmx_tag = head.index('src="/static/vendor/htmx.min.js"')
    alpine_tag = head.index('src="/static/vendor/alpine.min.js"')
    block_tag = head.index("  {% block head %}")
    assert htmx_tag < block_tag < alpine_tag


def test_jobs_polling_is_conditional_on_active_jobs() -> None:
    upload = Path(
        Path(__file__).resolve().parents[1] / "src/web_example/templates/upload.html"
    ).read_text(encoding="utf-8")
    partial = Path(
        Path(__file__).resolve().parents[1] / "src/web_example/templates/partials/upload_jobs.html"
    ).read_text(encoding="utf-8")
    assert "every 2s [window.uploadJobsActive()]" in upload
    assert "data-jobs-active" in partial


async def test_files_job_end_to_end_with_fake_rag(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")
    fake = _FakeRag()
    app.state.rag = fake

    response = await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[
            ("files", ("a.txt", b"hello", "text/plain")),
            ("files", ("b.md", b"# heading", "text/markdown")),
        ],
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    await _drain_background_jobs()

    listing = await client.get("/upload/api/jobs")
    assert listing.status_code == 200
    jobs = listing.json()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["kind"] == "files"
    assert job["target"] == "user"
    assert job["status"] == "done"
    assert job["total"] == 2
    assert job["processed"] == 2
    assert job["failed"] == 0
    assert [name for name, _global in fake.calls] == ["a.txt", "b.md"]
    assert all(global_pool is False for _name, global_pool in fake.calls)


async def test_unsupported_extension_rejected_with_error_listing(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")
    app.state.rag = _FakeRag()

    response = await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert "evil.exe" in response.text


async def test_unsupported_extension_counted_as_failed(
    client: AsyncClient, app: Any, register: Any
) -> None:
    """run_job (defence in depth) marks unsupported files as failed."""
    await register("alice")
    app.state.rag = _FakeRag()
    user = await _user_id(app, "alice")

    maker: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    async with maker() as db_session:
        job = await ingest_service.create_job(db_session, user, "user", "files", 2)
    ingest_service.spawn_run_job(
        app,
        job.id,
        user,
        "user",
        files=[("good.txt", b"payload"), ("bad.exe", b"MZ")],
    )
    await _drain_background_jobs()

    maker2: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    async with maker2() as db_session:
        done = await ingest_service.get_job(db_session, user, job.id)
    assert done is not None
    assert done.status == "done_with_errors"
    assert done.processed == 1
    assert done.failed == 1
    assert "bad.exe" in (done.detail or "")


async def test_file_limit_rejected(
    client: AsyncClient, app: Any, register: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await register("alice")
    app.state.rag = _FakeRag()
    monkeypatch.setattr(ingest_service, "MAX_FILES_PER_JOB", 2)

    response = await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[("files", (f"f{i}.txt", b"x", "text/plain")) for i in range(3)],
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert "too many" in response.text.lower()


async def test_empty_file_upload_rejected(client: AsyncClient, register: Any) -> None:
    await register("alice")
    response = await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[],
        follow_redirects=False,
    )
    assert response.status_code == 400


async def test_non_admin_global_target_forbidden(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")  # first signup is the admin
    await client.post("/logout")
    await register("bob")  # second signup is a plain user
    app.state.rag = _FakeRag()

    response = await client.post(
        "/upload/files",
        data={"target": "global"},
        files=[("files", ("a.txt", b"x", "text/plain"))],
        follow_redirects=False,
    )
    assert response.status_code == 403


async def test_jobs_partial_shows_job(client: AsyncClient, app: Any, register: Any) -> None:
    await register("alice")
    app.state.rag = _FakeRag()

    response = await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[("files", ("a.txt", b"x", "text/plain"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    await _drain_background_jobs()

    partial = await client.get("/upload/jobs")
    assert partial.status_code == 200
    assert "files" in partial.text
    assert "done" in partial.text
    assert "1/1" in partial.text


async def test_user_b_cannot_see_user_a_jobs(client: AsyncClient, app: Any, register: Any) -> None:
    await register("alice")
    app.state.rag = _FakeRag()
    await client.post(
        "/upload/files",
        data={"target": "user"},
        files=[("files", ("a.txt", b"x", "text/plain"))],
        follow_redirects=False,
    )
    await _drain_background_jobs()

    await client.post("/logout")
    await register("bob")

    listing = await client.get("/upload/api/jobs")
    assert listing.status_code == 200
    assert listing.json() == []

    partial = await client.get("/upload/jobs")
    assert partial.status_code == 200
    assert "a.txt" not in partial.text


async def test_admin_global_target_allowed_and_forwarded(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")  # admin
    fake = _FakeRag()
    app.state.rag = fake

    response = await client.post(
        "/upload/files",
        data={"target": "global"},
        files=[("files", ("a.txt", b"x", "text/plain"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    await _drain_background_jobs()

    listing = await client.get("/upload/api/jobs")
    job = listing.json()[0]
    assert job["target"] == "global"
    assert job["status"] == "done"
    assert fake.calls and fake.calls[0][1] is True


async def test_directory_job_runs_real_mass_pipeline(
    client: AsyncClient, app: Any, register: Any, tmp_path: Path
) -> None:
    """Directory jobs go through the real MassIngestor (mock RAG profile)."""
    await register("alice")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "note.txt").write_text("the sky is blue", encoding="utf-8")
    (inbox / "other.md").write_text("grass is green", encoding="utf-8")

    response = await client.post(
        "/upload/directory",
        data={"target": "user", "directory": str(inbox)},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    await _drain_background_jobs()

    listing = await client.get("/upload/api/jobs")
    job = listing.json()[0]
    assert job["kind"] == "directory"
    assert job["status"] == "done"
    assert job["total"] == 2
    assert job["processed"] == 2

    listing = await client.get("/documents/api/list")
    names = {row["name"] for row in listing.json()}
    assert {"note.txt", "other.md"} <= names

    missing = await client.post(
        "/upload/directory",
        data={"target": "user", "directory": str(tmp_path / "nope")},
        follow_redirects=False,
    )
    assert missing.status_code == 400


async def test_url_sitemap_job_ingests_discovered_sources(
    client: AsyncClient,
    app: Any,
    register: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sitemap URLs run the mass pipeline over each discovered source."""
    await register("alice")
    page1 = tmp_path / "page1.txt"
    page1.write_text("apples are red", encoding="utf-8")
    page2 = tmp_path / "page2.md"
    page2.write_text("leaves are green", encoding="utf-8")

    async def _fake_discover(url: str, client: Any = None) -> list[str]:
        return [str(page1), str(page2)]

    monkeypatch.setattr("rag_mass_inject.pipeline.discover_sitemap", _fake_discover)

    def _fake_validate(url: str, **_kwargs: Any) -> str:
        return url

    monkeypatch.setattr(upload_routes, "validate_public_url", _fake_validate)

    response = await client.post(
        "/upload/url",
        data={"target": "user", "url": "https://example.com/sitemap.xml"},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    await _drain_background_jobs()

    listing = await client.get("/upload/api/jobs")
    job = listing.json()[0]
    assert job["kind"] == "url"
    assert job["status"] == "done"
    assert job["processed"] == 2
    assert "2 document(s)" in (job["detail"] or "")

    listing = await client.get("/documents/api/list")
    names = {row["name"] for row in listing.json()}
    assert {"page1.txt", "page2.md"} <= names


async def test_url_single_page_job_ingests_one_document(
    client: AsyncClient,
    app: Any,
    register: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-sitemap URL is ingested as a single page via the pool pipeline."""
    await register("alice")

    class _FakePageDoc:
        id = "doc-web-1"
        source_uri = "https://example.com/hello"
        content_hash = "abc123"

    async def _fake_ingest_single(rag: Any, session: Any, uid: str, url: str, pool: bool) -> Any:
        return _FakePageDoc()

    monkeypatch.setattr(ingest_service, "_ingest_single_source", _fake_ingest_single)

    def _fake_validate(url: str, **_kwargs: Any) -> str:
        return url

    monkeypatch.setattr(upload_routes, "validate_public_url", _fake_validate)

    response = await client.post(
        "/upload/url",
        data={"target": "user", "url": "https://example.com/hello"},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    await _drain_background_jobs()

    listing = await client.get("/upload/api/jobs")
    job = listing.json()[0]
    assert job["status"] == "done"
    assert job["processed"] == 1

    listing = await client.get("/documents/api/list")
    rows = listing.json()
    assert any(row["id"] == "doc-web-1" and row["name"] == "hello" for row in rows)


async def test_url_job_validation_rejects_non_http(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")
    app.state.rag = _FakeRag()

    bad = await client.post(
        "/upload/url",
        data={"target": "user", "url": "ftp://example.com/sitemap.xml"},
        follow_redirects=False,
    )
    assert bad.status_code == 400


async def test_run_job_without_rag_engine_marks_failed(
    client: AsyncClient, app: Any, register: Any
) -> None:
    await register("alice")
    # The chat track wires a real RagManager in create_app; force it off here.
    app.state.rag = None
    user = await _user_id(app, "alice")

    maker: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    async with maker() as db_session:
        job = await ingest_service.create_job(db_session, user, "user", "files", 1)
    ingest_service.spawn_run_job(app, job.id, user, "user", files=[("a.txt", b"x")])
    await _drain_background_jobs()

    verify: async_sessionmaker[AsyncSession] = app.state.sessionmaker
    async with verify() as db_session:
        done = await ingest_service.get_job(db_session, user, job.id)
    assert done is not None
    assert done.status == "failed"
    assert "rag engine not initialised" in (done.detail or "").lower()
