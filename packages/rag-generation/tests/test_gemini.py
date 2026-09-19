"""Tests for the Gemini adapter (httpx.MockTransport, offline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
from rag_core.generation import Message, Usage
from rag_generation import GeminiProvider


async def test_contents_role_mapping_and_auth(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello"}, {"text": " world"}],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokens": 5,
                    "candidatesTokens": 3,
                    "totalTokens": 8,
                },
                "modelVersion": "gemini-1.5-flash",
            },
        )

    prov = wire(GeminiProvider, handler)
    req = make_request(
        messages=[
            Message(role="system", content="sys-inst"),
            Message(role="user", content="u"),
            Message(role="assistant", content="a"),
        ]
    )
    result = await prov.complete(req)

    headers = captured["headers"]
    assert headers["x-goog-api-key"] == "test-key"
    assert "authorization" not in headers

    body = captured["body"]
    assert "systemInstruction" in body
    assert body["systemInstruction"]["parts"][0]["text"] == "sys-inst"
    # assistant -> model role mapping
    assert [c["role"] for c in body["contents"]] == ["user", "model"]
    assert result.text == "Hello world"
    assert result.model == "gemini-1.5-flash"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=5, completion_tokens=3)


async def test_gemini_length_finish_reason(make_request, wire) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "cut"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ],
                "usageMetadata": {"promptTokens": 2, "candidatesTokens": 9},
            },
        )

    prov = wire(GeminiProvider, handler)
    result = await prov.complete(make_request())
    assert result.finish_reason == "length"
    assert result.usage == Usage(prompt_tokens=2, completion_tokens=9)


async def test_gemini_stream(make_request, wire) -> None:
    # Gemini streaming sends one JSON object per data: line, ending with [DONE].
    sse = (
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    prov = wire(GeminiProvider, handler)
    chunks = [c async for c in prov.stream(make_request(model="gemini-1.5-flash"))]
    assert chunks == ["Hel", "lo"]
