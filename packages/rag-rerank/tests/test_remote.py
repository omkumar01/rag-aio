"""Tests for :class:`RemoteReranker` HTTP adapters and endpoint auto-detection."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
from rag_core import Query, RerankError, RetrievalHit
from rag_rerank import RemoteReranker
from rag_rerank.config import RerankConfig


def make_query(text: str = "q") -> Query:
    return Query(text=text)


def make_hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id="a",
            document_id="d",
            score=0.2,
            normalized_score=0.2,
            rank=0,
            strategy="dense",
            text="alpha",
            strategy_scores={"dense": 0.2},
        ),
        RetrievalHit(
            chunk_id="b",
            document_id="d",
            score=0.9,
            normalized_score=0.9,
            rank=1,
            strategy="dense",
            text="beta",
            strategy_scores={"dense": 0.9},
        ),
        RetrievalHit(
            chunk_id="c",
            document_id="d",
            score=0.5,
            normalized_score=0.5,
            rank=2,
            strategy="dense",
            text="gamma",
            strategy_scores={"dense": 0.5},
        ),
    ]


def make_reranker(handler: Any, api_key_ref: str | None = None) -> RemoteReranker:
    transport = httpx.MockTransport(handler)
    return RemoteReranker(
        base_url="http://localhost:1234",
        model="rerank-model",
        api_key_ref=api_key_ref,
        transport=transport,
    )


def _rerank_handler(results: list[dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        return httpx.Response(200, json={"model": "rerank-model", "results": results})

    return handler


@pytest.mark.asyncio
async def test_rerank_endpoint_returns_ordered_results() -> None:
    results = [
        {"index": 0, "score": 0.9, "document": {"text": "alpha"}},
        {"index": 1, "score": 0.3, "document": {"text": "beta"}},
        {"index": 2, "score": 0.8, "document": {"text": "gamma"}},
    ]
    reranker = make_reranker(_rerank_handler(results))
    out = await reranker.rerank(make_query(), make_hits())
    assert [h.chunk_id for h in out] == ["a", "c", "b"]
    assert [h.score for h in out] == [0.9, 0.8, 0.3]
    assert all(h.model == "rerank-model" for h in out)
    assert reranker.mode == "rerank"
    await reranker.aclose()


@pytest.mark.asyncio
async def test_fallback_to_chat_completions_on_404() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        if request.url.path == "/rerank":
            return httpx.Response(404)
        if request.url.path == "/chat/completions":
            body = json.loads(request.content)
            assert "Documents:" in body["messages"][0]["content"]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "[8.0, 2.5, 7.0]"}}]
                },
            )
        return httpx.Response(404)

    reranker = make_reranker(handler)
    out = await reranker.rerank(make_query(), make_hits())
    # 8.0/10 -> 0.8 (a), 2.5/10 -> 0.25 (b), 7.0/10 -> 0.7 (c)
    assert reranker.mode == "chat"
    assert captured["path"] == "/chat/completions"
    assert [h.chunk_id for h in out] == ["a", "c", "b"]
    assert out[0].score == pytest.approx(0.8)
    await reranker.aclose()


@pytest.mark.asyncio
async def test_chat_garbage_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rerank":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "no array here"}}]},
        )

    reranker = make_reranker(handler)
    with pytest.raises(RerankError):
        await reranker.rerank(make_query(), make_hits())
    await reranker.aclose()


@pytest.mark.asyncio
async def test_chat_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rerank":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "[0.5, 0.2]"}}]},
        )

    reranker = make_reranker(handler)
    with pytest.raises(RerankError):
        await reranker.rerank(make_query(), make_hits())
    await reranker.aclose()


@pytest.mark.asyncio
async def test_api_key_from_env_in_auth_header_and_not_in_repr() -> None:
    os.environ["RERANK_TEST_KEY"] = "sk-secret-1234"
    try:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            assert request.url.path == "/rerank"
            return httpx.Response(
                200,
                json={"results": [{"index": 0, "score": 0.5}]},
            )

        reranker = make_reranker(handler, api_key_ref="RERANK_TEST_KEY")
        # Need exactly 1 candidate to align with the single-item mock.
        hits = [
            RetrievalHit(
                chunk_id="a",
                document_id="d",
                score=0.2,
                normalized_score=0.2,
                rank=0,
                strategy="dense",
                text="alpha",
            )
        ]
        await reranker.rerank(make_query(), hits)
        assert captured["auth"] == "Bearer sk-secret-1234"
        # Secret value must never appear in repr.
        assert "sk-secret-1234" not in repr(reranker)
        await reranker.aclose()
    finally:
        del os.environ["RERANK_TEST_KEY"]


@pytest.mark.asyncio
async def test_unset_api_key_env_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    reranker = make_reranker(handler, api_key_ref="RERANK_DEFINITELY_MISSING_VAR_X")
    with pytest.raises(RerankError):
        await reranker.rerank(make_query(), make_hits())
    await reranker.aclose()


@pytest.mark.asyncio
async def test_batches_documents_per_request() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/rerank":
            return httpx.Response(404)
        body = json.loads(request.content)
        docs = body["documents"]
        # The detection probe sends a synthetic ["probe"] document.
        if docs == ["probe"]:
            return httpx.Response(200, json={"results": []})
        batch_sizes.append(len(docs))
        results = [{"index": i, "score": float(i)} for i in range(len(docs))]
        return httpx.Response(200, json={"results": results})

    cfg = RerankConfig(batch_size=2, max_candidates=10)
    reranker = RemoteReranker(
        base_url="http://localhost:1234",
        model="rerank-model",
        config=cfg,
        transport=httpx.MockTransport(handler),
    )
    hits = [
        RetrievalHit(
            chunk_id=str(i),
            document_id="d",
            score=float(i),
            normalized_score=float(i),
            rank=i,
            strategy="dense",
            text=f"doc{i}",
        )
        for i in range(4)
    ]
    await reranker.rerank(make_query(), hits)
    # 4 hits with batch_size 2 -> two batches of two documents each.
    assert batch_sizes == [2, 2]
    await reranker.aclose()
