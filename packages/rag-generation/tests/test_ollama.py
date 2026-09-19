"""Tests for the Ollama native adapter (httpx.MockTransport, offline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
from rag_core.generation import Usage
from rag_generation import OllamaProvider, collect_stream


async def test_ollama_complete_and_usage(make_request, wire) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llama3",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "Hi!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        )

    prov = wire(OllamaProvider, handler)
    result = await prov.complete(make_request())

    body = captured["body"]
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"
    # options carry the generation params
    assert body["options"]["temperature"] == 0.2
    assert result.text == "Hi!"
    assert result.model == "llama3"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=2)


async def test_ollama_ndjson_stream(make_request, wire) -> None:
    ndjson = (
        '{"model":"llama3","message":{"role":"assistant","content":"Hel"},"done":false}\n'
        '{"model":"llama3","message":{"role":"assistant","content":"lo!"},"done":false}\n'
        '{"model":"llama3","done":true,"prompt_eval_count":4,"eval_count":2}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            content=ndjson.encode(),
            headers={"content-type": "application/x-ndjson"},
        )

    prov = wire(OllamaProvider, handler)
    text = await collect_stream(prov.stream(make_request()))
    assert text == "Hello!"
