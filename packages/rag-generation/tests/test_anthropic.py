"""Tests for the Anthropic adapter (httpx.MockTransport, offline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
from rag_core.generation import Message, Usage
from rag_generation import AnthropicProvider


async def test_system_extracted_to_field_and_headers(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "text", "text": " there"},
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )

    prov = wire(AnthropicProvider, handler)
    req = make_request(
        messages=[
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
        ]
    )
    result = await prov.complete(req)

    headers = captured["headers"]
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    # No bearer token leaks for Anthropic.
    assert "authorization" not in headers

    body = captured["body"]
    assert body["system"] == "You are helpful"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["max_tokens"] >= 1
    assert result.id == "msg_1"
    assert result.text == "Hi there"
    assert result.model == "claude-3-5-sonnet"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=2)


async def test_anthropic_finish_reason_length(make_request, wire) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "truncated"}],
                "stop_reason": "max_tokens",
                "model": "claude",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    prov = wire(AnthropicProvider, handler)
    result = await prov.complete(make_request())
    assert result.finish_reason == "length"


async def test_anthropic_stream_deltas(make_request, wire) -> None:
    sse = (
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n'
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n'
        'data: {"type":"message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    prov = wire(AnthropicProvider, handler)
    chunks = [c async for c in prov.stream(make_request())]
    assert chunks == ["Hel", "lo"]
