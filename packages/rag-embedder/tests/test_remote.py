"""Tests for :class:`OpenAICompatibleEmbedder` HTTP adapters."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import httpx
import pytest
from rag_core import EmbeddingError
from rag_embedder import OpenAICompatibleEmbedder


def make_embedder(
    handler: Any, api_key_ref: str | None = None, **kwargs: Any
) -> OpenAICompatibleEmbedder:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        model="text-embedding-test",
        api_key_ref=api_key_ref,
        transport=transport,
        **kwargs,
    )


def _embeddings_handler(vectors: list[list[float]], check: Any = None) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if check is not None:
            check(request)
        body = json.loads(request.content)
        inputs = body.get("input", [])
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vectors[i % len(vectors)]}
                    for i in range(len(inputs))
                ],
                "model": body.get("model", "text-embedding-test"),
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )

    return handler


async def test_embed_returns_vectors_in_order() -> None:
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    embedder = make_embedder(_embeddings_handler(vectors))
    out = await embedder.embed(["a", "b", "c"])
    assert out == vectors
    assert embedder.model_name == "text-embedding-test"
    assert embedder.dim == 3
    assert embedder.normalize is False
    await embedder.aclose()


async def test_embed_empty_input_returns_empty() -> None:
    embedder = make_embedder(_embeddings_handler([]))
    assert await embedder.embed([]) == []
    await embedder.aclose()


async def test_embed_batches_requests() -> None:
    calls: list[list[str]] = []

    def check(request: httpx.Request) -> None:
        body = json.loads(request.content)
        calls.append(body["input"])
        assert body["model"] == "text-embedding-test"

    embedder = make_embedder(
        _embeddings_handler([[1.0, 0.0], [0.0, 1.0]], check=check), batch_size=2
    )
    out = await embedder.embed(["a", "b", "c"])
    assert len(calls) == 2  # [a, b] then [c]
    assert calls[0] == ["a", "b"]
    assert calls[1] == ["c"]
    assert out == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    await embedder.aclose()


async def test_embed_honours_reordered_indices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.4]},
                    {"index": 0, "embedding": [0.1]},
                    {"index": 2, "embedding": [0.7]},
                ]
            },
        )

    embedder = make_embedder(handler)
    out = await embedder.embed(["a", "b", "c"])
    assert out == [[0.1], [0.4], [0.7]]
    await embedder.aclose()


async def test_embed_normalize_l2() -> None:
    embedder = make_embedder(_embeddings_handler([[3.0, 4.0]]), normalize=True)
    out = await embedder.embed(["x"])
    assert out[0] == pytest.approx([0.6, 0.8])
    await embedder.aclose()


async def test_api_key_ref_resolved_at_request_time(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def check(request: httpx.Request) -> None:
        captured["auth"] = request.headers.get("Authorization", "")

    embedder = make_embedder(_embeddings_handler([[1.0]], check=check), api_key_ref="TEST_EMB_KEY")
    # Key missing -> request-time failure, not construction-time.
    monkeypatch.delenv("TEST_EMB_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="TEST_EMB_KEY"):
        await embedder.embed(["x"])
    monkeypatch.setenv("TEST_EMB_KEY", "sekrit")
    await embedder.embed(["x"])
    assert captured["auth"] == "Bearer sekrit"
    await embedder.aclose()


async def test_repr_does_not_leak_secret() -> None:
    embedder = make_embedder(_embeddings_handler([[1.0]]), api_key_ref="TEST_EMB_KEY")
    os.environ.setdefault("TEST_EMB_KEY", "sekrit")
    assert "sekrit" not in repr(embedder)
    assert "api_key_ref='TEST_EMB_KEY'" in repr(embedder)


async def test_http_error_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        await embedder.embed(["x"])
    await embedder.aclose()


async def test_malformed_response_raises() -> None:
    embedder = make_embedder(_embeddings_handler([[1.0]]))
    with pytest.raises(EmbeddingError):
        await embedder._parse_embeddings({"data": [{"index": 0}]}, 1)
    await embedder.aclose()


async def test_dim_can_be_declared_upfront() -> None:
    embedder = make_embedder(_embeddings_handler([[1.0, 2.0]]), dim=2, normalize=True)
    assert embedder.dim == 2
    out = await embedder.embed(["x"])
    norm = math.sqrt(sum(v * v for v in out[0]))
    assert norm == pytest.approx(1.0)
    await embedder.aclose()
