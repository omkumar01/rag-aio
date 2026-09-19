"""Tests for the OpenAI-compatible adapter (httpx.MockTransport, offline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from rag_core.errors import ProviderError, ProviderUnavailableError, RateLimitError
from rag_core.generation import Usage
from rag_generation import OpenAICompatibleProvider


def _ok_response(text: str = "Hello there") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


async def test_complete_body_shape_and_parsing(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_ok_response("Hello there"))

    prov = wire(OpenAICompatibleProvider, handler)
    req = make_request(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=64,
        top_p=0.9,
        stop=["END"],
    )
    result = await prov.complete(req)

    body = captured["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 64
    assert body["top_p"] == 0.9
    assert body["stop"] == ["END"]
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert result.id == "chatcmpl-test"
    assert result.text == "Hello there"
    assert result.model == "gpt-4o-mini"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=5, completion_tokens=3)


async def test_json_schema_sets_response_format(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_response("{}"))

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    # Fallback: supports_json_schema=False -> json_object
    prov = wire(OpenAICompatibleProvider, handler)
    await prov.complete(make_request(json_schema=schema))
    assert captured["body"]["response_format"] == {"type": "json_object"}

    # Capable: supports_json_schema=True -> json_schema
    prov2 = wire(OpenAICompatibleProvider, handler, supports_json_schema=True)
    await prov2.complete(make_request(json_schema=schema))
    rf = captured["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


async def test_tools_passthrough_and_finish_tool_calls(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "foo", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            },
        )

    tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    prov = wire(OpenAICompatibleProvider, handler)
    result = await prov.complete(make_request(tools=tools))
    assert captured["body"]["tools"] == tools
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0]["id"] == "t1"


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_mapping(status, make_request, wire) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "bad key"}})

    prov = wire(OpenAICompatibleProvider, handler)
    with pytest.raises(ProviderError) as exc:
        await prov.complete(make_request())
    assert exc.value.code == "auth"
    # auth failures must not be conflated with transient errors
    assert not isinstance(exc.value, (RateLimitError, ProviderUnavailableError))


@pytest.mark.parametrize(
    "status, exc_cls",
    [
        (404, ProviderUnavailableError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
    ],
)
async def test_status_error_mapping(status, exc_cls, make_request, wire) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "fail"})

    prov = wire(OpenAICompatibleProvider, handler)
    with pytest.raises(exc_cls):
        await prov.complete(make_request())


async def test_stream_yields_deltas_in_order(make_request, wire) -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        'data: {"choices":[{}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    prov = wire(OpenAICompatibleProvider, handler)
    chunks = [c async for c in prov.stream(make_request())]
    assert chunks == ["Hel", "lo"]


async def test_stream_malformed_lines_skipped(make_request, wire) -> None:
    sse = (
        "data: {not valid json}\n\n"
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    prov = wire(OpenAICompatibleProvider, handler)
    assert [c async for c in prov.stream(make_request())] == ["hi"]


async def test_stream_done_terminates(make_request, wire) -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        "data: [DONE]\n\n"
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    prov = wire(OpenAICompatibleProvider, handler)
    assert [c async for c in prov.stream(make_request())] == ["a"]


async def test_list_models(make_request, wire) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m1"}, {"id": "m2"}]})

    prov = wire(OpenAICompatibleProvider, handler)
    assert await prov.list_models() == ["m1", "m2"]
