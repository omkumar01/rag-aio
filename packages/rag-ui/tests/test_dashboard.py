"""Tests for :class:`rag_ui.dashboard.Dashboard` using ``httpx.MockTransport``.

No real network/server is involved — every endpoint is served by an in-process
async handler, including an "unreachable backend" case where the transport
itself raises a connection error.
"""

from __future__ import annotations

import json

import httpx
from rag_ui.dashboard import Dashboard

SSE_TEXT = 'data: {"delta": "hello"}\n\ndata: {"delta": " world"}\n\ndata: [DONE]\n\n'


def _dash(handler: object) -> Dashboard:
    """Build a Dashboard wired to an in-memory mock transport."""
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return Dashboard("http://testserver", 5.0, transport=transport)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Synchronous (JSON) endpoints
# --------------------------------------------------------------------------- #


async def test_health() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    result = await _dash(handler).health()
    assert "error" not in result
    assert result == {"status": "ok"}


async def test_ready() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ready": True, "components": {"vector_store": True}})

    result = await _dash(handler).ready()
    assert result["ready"] is True


async def test_metrics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "pipeline": "local_fast",
                "schema_version": "1.0",
                "vector_points": 5,
                "cache": {"hits": 1},
            },
        )

    result = await _dash(handler).metrics()
    assert result["pipeline"] == "local_fast"
    assert result["vector_points"] == 5


async def test_list_pipelines() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "schema_version": "1.0",
                "pipelines": [{"id": "p1", "config": {"name": "local_fast"}}],
            },
        )

    result = await _dash(handler).list_pipelines()
    assert result["pipelines"][0]["config"]["name"] == "local_fast"


# --------------------------------------------------------------------------- #
# ask: JSON + streaming
# --------------------------------------------------------------------------- #


async def test_ask_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "hi"
        assert body["stream"] is False
        assert body["overrides"] == {}
        return httpx.Response(200, json={"answer": "hello", "citations": [], "query_id": "q1"})

    result = await _dash(handler).ask("hi", stream=False)
    assert result["answer"] == "hello"
    assert "error" not in result


async def test_ask_json_with_overrides() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["overrides"] == {"top_k": 3}
        return httpx.Response(200, json={"answer": "ok", "citations": []})

    result = await _dash(handler).ask("hi", stream=False, overrides={"top_k": 3})
    assert result["answer"] == "ok"


async def test_ask_stream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, text=SSE_TEXT, headers={"content-type": "text/event-stream"})

    gen = await _dash(handler).ask("hi", stream=True)
    deltas = [delta async for delta in gen]
    assert deltas == [{"delta": "hello"}, {"delta": " world"}]


# --------------------------------------------------------------------------- #
# ingest + jobs
# --------------------------------------------------------------------------- #


async def test_ingest() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["source"] == "/tmp/x.md"
        assert body["recursive"] is False
        return httpx.Response(200, json={"content_hash": "abc", "pages": 1})

    result = await _dash(handler).ingest("/tmp/x.md")
    assert result["content_hash"] == "abc"


async def test_ingest_recursive() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["recursive"] is True
        return httpx.Response(200, json={"content_hash": "def", "pages": 3})

    result = await _dash(handler).ingest("/tmp/docs", recursive=True)
    assert result["pages"] == 3


async def test_job_found() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "123", "status": "completed", "kind": "ingest"})

    result = await _dash(handler).list_jobs("123")
    assert result["status"] == "completed"


async def test_job_not_found() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/jobs/nope"
        return httpx.Response(404, json={"detail": "job nope not found"})

    result = await _dash(handler).list_jobs("nope")
    assert "error" in result
    assert result["status_code"] == 404
    assert result["endpoint"] == "list_jobs"


# --------------------------------------------------------------------------- #
# Failure handling: unreachable backend and streaming errors
# --------------------------------------------------------------------------- #


async def test_unreachable_backend_health() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = await _dash(handler).health()
    assert "error" in result
    assert result["endpoint"] == "health"
    assert result["ok"] is False


async def test_unreachable_backend_ask_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = await _dash(handler).ask("hi", stream=False)
    assert "error" in result


async def test_unreachable_backend_ask_stream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    gen = await _dash(handler).ask("hi", stream=True)
    deltas = [delta async for delta in gen]
    assert len(deltas) == 1
    assert "error" in deltas[0]


async def test_http_status_error_ingest() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": {"job_id": "j1", "error": "bad source"}})

    result = await _dash(handler).ingest("bad.txt")
    assert result["status_code"] == 422
    assert "error" in result
