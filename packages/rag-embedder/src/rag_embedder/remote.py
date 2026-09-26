"""OpenAI-compatible remote embedding client.

:class:`OpenAICompatibleEmbedder` turns any ``POST {base_url}/embeddings``
endpoint (LM Studio, Ollama's OpenAI mode, vLLM, OpenAI, ...) into a
:class:`rag_core.protocols.Embedder`, so remote and local (``FastEmbedDense``)
embedders are interchangeable across the pipeline.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

import httpx
from rag_core import EmbeddingError

from .embedding import _l2_normalize

__all__ = ["OpenAICompatibleEmbedder"]


class OpenAICompatibleEmbedder:
    """Dense embedder backed by an OpenAI-compatible ``/embeddings`` endpoint.

    ``api_key_ref`` names an environment variable holding the bearer token. The
    key is resolved at request time (so rotation does not require a restart)
    and is never stored on the instance, so it cannot leak via
    ``repr``/logs/serialization.

    Requests are batched to ``batch_size`` and bounded by a semaphore of
    ``max_concurrency``, mirroring :func:`embed_many`. A single
    :class:`httpx.AsyncClient` is created lazily and reused; close it with
    :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_ref: str | None = None,
        batch_size: int = 64,
        max_concurrency: int = 4,
        timeout_s: float = 60.0,
        normalize: bool = False,
        dim: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_ref = api_key_ref
        self._batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s
        self._normalize = normalize
        self._dim = dim
        self._transport: httpx.AsyncBaseTransport | None = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality: the constructor value, else learned from
        the first successful response (``None`` until then)."""
        return self._dim

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleEmbedder(base_url={self._base_url!r}, "
            f"model={self._model!r}, api_key_ref={self._api_key_ref!r})"
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        batches = [
            text_list[i : i + self._batch_size] for i in range(0, len(text_list), self._batch_size)
        ]

        async def _run(batch: list[str]) -> list[list[float]]:
            async with self._sem:
                return await self._embeddings_request(batch)

        results = await asyncio.gather(*[_run(b) for b in batches])
        return [vec for batch_result in results for vec in batch_result]

    async def aclose(self) -> None:
        """Close the underlying HTTP client (no-op if never opened)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- internals -----------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_s),
                transport=self._transport,
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key_ref is not None:
            key = os.environ.get(self._api_key_ref)
            if key is None:
                raise EmbeddingError(f"api_key_ref env var {self._api_key_ref!r} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def _embeddings_request(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        headers = self._auth_headers()
        body: dict[str, Any] = {"model": self._model, "input": texts}
        try:
            resp = await client.post(f"{self._base_url}/embeddings", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embeddings request failed: {exc}") from exc
        if resp.status_code != 200:
            raise EmbeddingError(f"/embeddings failed: HTTP {resp.status_code} {resp.text[:200]}")
        return self._parse_embeddings(resp.json(), len(texts))

    def _parse_embeddings(self, data: Any, n: int) -> list[list[float]]:
        """Parse an OpenAI ``/embeddings`` response into an index-aligned list."""
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != n:
            got = len(items) if isinstance(items, list) else "missing"
            raise EmbeddingError(f"expected {n} embeddings, got {got}")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingError("malformed /embeddings response item")
        vectors: list[list[float]]
        # Providers may reorder results; honour explicit indices when present.
        indices = [item.get("index") for item in items]
        if all(isinstance(i, int) and not isinstance(i, bool) for i in indices) and sorted(
            indices
        ) == list(range(n)):
            vectors = [[] for _ in range(n)]
            for item, idx in zip(items, indices, strict=True):
                vectors[idx] = self._to_vector(item["embedding"])
        else:
            vectors = [self._to_vector(item["embedding"]) for item in items]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        if self._normalize:
            vectors = [_l2_normalize(vec) for vec in vectors]
        return vectors

    def _to_vector(self, embedding: list[Any]) -> list[float]:
        try:
            return [float(v) for v in embedding]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("non-numeric embedding value in response") from exc
