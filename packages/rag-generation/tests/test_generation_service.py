"""Tests for the GenerationService routing/retry/fallback facade."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from rag_core.errors import ProviderError
from rag_core.generation import GenerationRequest, Message
from rag_generation import GenerationService, OpenAICompatibleProvider, collect_stream


def _ok_body(text: str = "ok") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "model": "x",
    }


def _req(messages: list[Message] | None = None, **over: Any) -> GenerationRequest:
    defaults: dict[str, Any] = {
        "messages": messages or [Message(role="user", content="Hi")],
    }
    defaults.update(over)
    return GenerationRequest(**defaults)


def _make_provider(handler: Any) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleProvider(api_key_provider=lambda: "k", transport=transport)


async def test_default_picks_first_provider() -> None:
    calls = {"a": 0, "b": 0}

    def ha(r: httpx.Request) -> httpx.Response:
        calls["a"] += 1
        return httpx.Response(200, json=_ok_body("a"))

    def hb(r: httpx.Request) -> httpx.Response:
        calls["b"] += 1
        return httpx.Response(200, json=_ok_body("b"))

    svc = GenerationService({"a": _make_provider(ha), "b": _make_provider(hb)}, backoff_base=0.0)
    result = await svc.generate(_req())
    assert result.text == "a"
    assert calls["a"] == 1 and calls["b"] == 0


async def test_retry_on_429_then_success() -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slowdown"})
        return httpx.Response(200, json=_ok_body("recovered"))

    svc = GenerationService({"oai": _make_provider(handler)}, backoff_base=0.0)
    result = await svc.generate(_req())
    assert result.text == "recovered"
    assert calls["n"] == 2


async def test_no_retry_on_401() -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    svc = GenerationService({"oai": _make_provider(handler)}, backoff_base=0.0)
    with pytest.raises(ProviderError) as exc:
        await svc.generate(_req())
    assert exc.value.code == "auth"
    assert calls["n"] == 1


async def test_fallback_to_second_provider_when_first_unavailable() -> None:
    calls = {"a": 0, "b": 0}

    def ha(r: httpx.Request) -> httpx.Response:
        calls["a"] += 1
        return httpx.Response(500, json={"error": "down"})

    def hb(r: httpx.Request) -> httpx.Response:
        calls["b"] += 1
        return httpx.Response(200, json=_ok_body("from-b"))

    svc = GenerationService(
        {"a": _make_provider(ha), "b": _make_provider(hb)},
        backoff_base=0.0,
        max_retries=1,
        fallback_names=["b"],
    )
    result = await svc.generate(_req())
    assert result.text == "from-b"
    assert calls["b"] == 1
    # first provider exhausted its retry budget (1 retry => 2 attempts)
    assert calls["a"] == 2


async def test_model_override_applied() -> None:
    captured: dict[str, Any] = {}

    def handler(r: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(r.content)
        return httpx.Response(200, json=_ok_body())

    svc = GenerationService({"oai": _make_provider(handler)}, backoff_base=0.0)
    await svc.generate(_req(), provider_name="oai", model="override-model")
    assert captured["body"]["model"] == "override-model"


async def test_on_usage_hook_called() -> None:
    seen = []

    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body())

    svc = GenerationService(
        {"oai": _make_provider(handler)},
        backoff_base=0.0,
        on_usage=lambda r: seen.append(r),
    )
    result = await svc.generate(_req())
    assert seen == [result]
    assert seen[0].usage is not None
    assert seen[0].usage.prompt_tokens == 5


async def test_collect_stream_helper() -> None:
    async def gen() -> Any:
        yield "foo"
        yield "bar"

    assert await collect_stream(gen()) == "foobar"
