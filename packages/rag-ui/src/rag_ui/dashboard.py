"""Async HTTP client for the rag-aio FastAPI service.

:class:`Dashboard` wraps the small surface area exposed by the FastAPI app
created via :func:`rag_aio.app.get_app` (health, ready, metrics, pipelines,
ask, ingest, jobs). Every method is fault-tolerant: an unreachable or
unhealthy backend is reported back as a plain ``dict`` containing an ``"error"``
key instead of raising, so Streamlit never crashes on a down backend.

Streaming ``ask`` responses (SSE) are exposed as an *async iterator* of decoded
JSON delta dicts.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

__all__ = ["Dashboard"]

_SENTINEL_DONE = "[DONE]"
"""Server-sent-event marker signalling end of a streaming ``ask``."""


class Dashboard:
    """A fault-tolerant async client for the rag-aio management API.

    Parameters
    ----------
    base_url:
        Root URL of the FastAPI service (``"/health"``, ``"/v1/ask"`` … are
        resolved relative to it).
    timeout:
        Per-request timeout in seconds. Unreachable backends fail fast
        (connection-refused) and return an error dict immediately.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.timeout: float = float(timeout)
        # Injectable transport so callers (notably tests with httpx.MockTransport)
        # can avoid hitting a real network. ``None`` → httpx's default transport.
        self._transport: httpx.AsyncBaseTransport | None = transport

    # ------------------------------------------------------------------ #
    # low-level transport
    # ------------------------------------------------------------------ #
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, transport=self._transport
        )

    @staticmethod
    def _error(endpoint: str, message: object, *, status_code: int | None = None) -> dict[str, Any]:
        err: dict[str, Any] = {"ok": False, "error": str(message), "endpoint": endpoint}
        if status_code is not None:
            err["status_code"] = status_code
        return err

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async with self._client() as client:
            return await client.request(method, url, **kwargs)

    async def _ok_json(self, method: str, url: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """GET/POST that returns parsed JSON, or an error dict on any failure."""
        try:
            resp = await self._request(method, url, **kwargs)
        except httpx.RequestError as exc:
            return self._error(endpoint, exc)
        if resp.status_code >= 400:
            return self._error(endpoint, f"HTTP {resp.status_code}", status_code=resp.status_code)
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error(endpoint, f"invalid JSON response: {exc}")
        if not isinstance(body, dict):
            return self._error(endpoint, f"expected JSON object, got {type(body).__name__}")
        return body

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    async def health(self) -> dict[str, Any]:
        """GET ``/health`` — liveness probe."""
        return await self._ok_json("GET", "/health", "health")

    async def ready(self) -> dict[str, Any]:
        """GET ``/ready`` — readiness probe with component checks."""
        return await self._ok_json("GET", "/ready", "ready")

    async def metrics(self) -> dict[str, Any]:
        """GET ``/metrics`` — pipeline/ingestion/cache metrics."""
        return await self._ok_json("GET", "/metrics", "metrics")

    async def list_pipelines(self) -> dict[str, Any]:
        """GET ``/v1/pipelines`` — registered pipeline specs."""
        return await self._ok_json("GET", "/v1/pipelines", "list_pipelines")

    async def list_jobs(self, job_id: str | None = None) -> dict[str, Any]:
        """GET ``/v1/jobs`` (summary) or ``/v1/jobs/{job_id}`` (status record).

        The backend only implements per-job lookups; calling without a
        ``job_id`` still attempts the request so the caller learns the
        endpoint is unsupported (404 -> error dict).
        """
        if job_id is None:
            return await self._ok_json("GET", "/v1/jobs", "list_jobs")
        return await self._ok_json("GET", f"/v1/jobs/{job_id}", "list_jobs")

    async def ask(
        self,
        query: str,
        stream: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        """POST ``/v1/ask``.

        With ``stream=False`` (default) returns the decoded JSON response dict
        (an :class:`~rag_orchestrator.ask.AskResult` payload). With
        ``stream=True`` returns an *async iterator* of SSE delta dicts; the
        iterator is safe to consume only within a running event loop.
        """
        payload: dict[str, Any] = {
            "query": query,
            "stream": stream,
            "overrides": overrides or {},
        }
        if not stream:
            return await self._ok_json("POST", "/v1/ask", "ask", json=payload)
        return self._stream_sse("POST", "/v1/ask", json=payload)

    async def ingest(self, source: str, recursive: bool = False) -> dict[str, Any]:
        """POST ``/v1/ingest`` — ingest a file or directory."""
        payload: dict[str, Any] = {"source": source, "recursive": recursive}
        return await self._ok_json("POST", "/v1/ingest", "ingest", json=payload)

    # ------------------------------------------------------------------ #
    # streaming helper
    # ------------------------------------------------------------------ #
    async def _stream_sse(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async with self._client() as client, client.stream(method, url, **kwargs) as resp:
                if resp.status_code >= 400:
                    yield self._error(
                        "ask", f"HTTP {resp.status_code}", status_code=resp.status_code
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == _SENTINEL_DONE:
                        return
                    try:
                        decoded: Any = json.loads(data)
                    except json.JSONDecodeError:
                        yield {"raw": data}
                        continue
                    if isinstance(decoded, dict):
                        yield decoded
                    else:
                        yield {"delta": decoded}
        except httpx.RequestError as exc:
            yield self._error("ask", exc)
