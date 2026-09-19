"""Shared HTTP plumbing for provider adapters.

All adapters talk to providers over raw HTTP via :mod:`httpx` (no vendor SDKs).
Authentication is resolved *per request* through an ``api_key_provider`` callable
(secrets-by-reference: rotatable env vars, keyring, files). HTTP status codes are
mapped onto the rag-core error taxonomy. Response bodies are attached to error
``details`` only as truncated, header-free snippets.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from rag_core.errors import (
    ProviderError,
    ProviderUnavailableError,
    RagError,
    RateLimitError,
)

ApiKeyProvider = Callable[[], str | None]

_SNIPPET_LIMIT = 200


def _no_key() -> str | None:
    return None


class ProviderHttpClient:
    """Minimal httpx-backed client with provider-aware auth and error mapping.

    ``api_key_provider`` is resolved per request so a secret rotated between
    requests is picked up without rebuilding the client. ``transport`` is injectable
    so tests can wire in :class:`httpx.MockTransport`.
    """

    def __init__(
        self,
        base_url: str,
        api_key_provider: ApiKeyProvider | None = None,
        *,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._api_key_provider: ApiKeyProvider = api_key_provider or _no_key
        self._timeout = httpx.Timeout(timeout_s)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- auth / headers ------------------------------------------------------

    def auth_headers(self) -> dict[str, str]:
        """Auth headers resolved per request. Override for non-Bearer schemes."""
        key = self._api_key_provider()
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}"}

    def request_headers(self) -> dict[str, str]:
        """Content + auth headers merged per request."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.auth_headers())
        return headers

    # -- non-streaming JSON --------------------------------------------------

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self.client.post(path, headers=self.request_headers(), json=payload)
        except httpx.TimeoutException as e:
            raise self._timeout_error(path) from e
        except httpx.TransportError as e:
            raise self._transport_error(path, e) from e
        except httpx.HTTPError as e:
            raise ProviderUnavailableError(
                f"HTTP error contacting {path}",
                details=self._safe_details(0, str(e)),
            ) from e
        return self._handle_response(resp)

    async def get_json(self, path: str) -> dict[str, Any]:
        try:
            resp = await self.client.get(path, headers=self.request_headers())
        except httpx.TimeoutException as e:
            raise self._timeout_error(path) from e
        except httpx.TransportError as e:
            raise self._transport_error(path, e) from e
        except httpx.HTTPError as e:
            raise ProviderUnavailableError(
                f"HTTP error contacting {path}",
                details=self._safe_details(0, str(e)),
            ) from e
        return self._handle_response(resp)

    # -- streaming SSE -------------------------------------------------------

    async def stream_sse(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream ``text/event-stream`` lines, decoding each ``data:`` field as JSON.

        ``[DONE]`` terminates the stream. Lines that are comments, empty, or fail
        to parse as JSON are skipped gracefully.
        """
        try:
            async with self.client.stream(
                "POST", path, headers=self.request_headers(), json=payload
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp.status_code, resp.text)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(":") or line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:") :].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            # Malformed SSE data line: skip gracefully.
                            continue
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
            if not isinstance(e, (ProviderError, RateLimitError)):
                raise ProviderUnavailableError(
                    f"Transport error streaming {path}",
                    details=self._safe_details(0, str(e)),
                ) from e
            raise

    async def stream_ndjson(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream line-delimited JSON (e.g. Ollama ``/api/chat``).

        Each non-empty line is parsed as JSON; ``[DONE]`` terminates the stream
        and malformed lines are skipped gracefully.
        """
        try:
            async with self.client.stream(
                "POST", path, headers=self.request_headers(), json=payload
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp.status_code, resp.text)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line == "[DONE]":
                        if line == "[DONE]":
                            break
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
            if not isinstance(e, (ProviderError, RateLimitError)):
                raise ProviderUnavailableError(
                    f"Transport error streaming {path}",
                    details=self._safe_details(0, str(e)),
                ) from e
            raise

    # -- response handling ---------------------------------------------------

    def _handle_response(self, resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code >= 400:
            self._raise_for_status(resp.status_code, resp.text)
        if resp.status_code == 204 or not resp.content:
            return {}
        data: object = resp.json()
        return data if isinstance(data, dict) else {}

    def _raise_for_status(self, status_code: int, body: str) -> None:
        details = self._safe_details(status_code, body)
        if status_code in (401, 403):
            raise ProviderError("Provider authentication failed", code="auth", details=details)
        if status_code == 404:
            raise ProviderUnavailableError("Provider endpoint not found", details=details)
        if status_code == 429:
            raise RateLimitError("Rate limited by provider", details=details)
        if status_code >= 500:
            raise ProviderUnavailableError(f"Provider error (HTTP {status_code})", details=details)
        if status_code >= 400:
            raise ProviderError(f"Provider request failed (HTTP {status_code})", details=details)

    @staticmethod
    def _safe_details(status_code: int, body: str) -> dict[str, Any]:
        """Build error details with a truncated, header-free response snippet."""
        snippet = body[:_SNIPPET_LIMIT]
        return {"status_code": status_code, "response_snippet": snippet}

    @staticmethod
    def _timeout_error(path: str) -> RagError:
        return ProviderUnavailableError(
            f"Request to {path} timed out", details={"status_code": 0, "response_snippet": ""}
        )

    @staticmethod
    def _transport_error(path: str, exc: Exception) -> RagError:
        return ProviderUnavailableError(
            f"Connection error contacting {path}",
            details={"status_code": 0, "response_snippet": str(exc)[:_SNIPPET_LIMIT]},
        )
