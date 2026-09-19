"""OpenAI-compatible adapter (LM Studio, vLLM, Ollama OpenAI layer, llama.cpp).

This is the reference adapter. Covers the OpenAI Chat Completions wire format shared
by every OpenAI-compatible server. No vendor SDK: all calls go through
:class:`ProviderHttpClient` (raw httpx).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_core.errors import ProviderError
from rag_core.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    Usage,
)
from rag_core.protocols import LLMProvider

from .http import ApiKeyProvider, ProviderHttpClient

_OPENAI_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


class OpenAICompatibleProvider(ProviderHttpClient, LLMProvider):
    """Adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` should include the ``/v1`` path segment (e.g.
    ``http://localhost:1234/v1``). ``supports_json_schema`` gates whether a
    ``json_schema`` request uses the structured ``json_schema`` response format
    (falls back to ``json_object`` otherwise, best-effort).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key_provider: ApiKeyProvider | None = None,
        *,
        timeout_s: float = 120.0,
        transport: Any = None,
        supports_json_schema: bool = False,
    ) -> None:
        super().__init__(base_url, api_key_provider, timeout_s=timeout_s, transport=transport)
        self.supports_json_schema = supports_json_schema

    # -- public API ----------------------------------------------------------

    async def complete(self, request: GenerationRequest) -> GenerationResult:
        payload = self._build_payload(request)
        data = await self.post_json("/chat/completions", payload)
        return self._parse_completion(data, request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        payload = self._build_payload(request)
        payload["stream"] = True
        async for chunk in self.stream_sse("/chat/completions", payload):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content

    async def list_models(self) -> list[str]:
        data = await self.get_json("/models")
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    # -- payload construction --------------------------------------------------

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or "default",
            "messages": [self._map_message(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = request.tools
        self._apply_response_format(payload, request)
        return payload

    def _apply_response_format(self, payload: dict[str, Any], request: GenerationRequest) -> None:
        if not request.json_schema:
            return
        if self.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",
                    "schema": request.json_schema,
                    "strict": False,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

    @staticmethod
    def _map_message(message: Any) -> dict[str, Any]:
        # rag_core.Message exposes role/content/name; OpenAI mirrors these directly.
        mapped: dict[str, Any] = {"role": message.role, "content": message.content}
        if getattr(message, "name", None) is not None:
            mapped["name"] = message.name
        return mapped

    # -- response parsing ----------------------------------------------------

    def _parse_completion(
        self, data: dict[str, Any], request: GenerationRequest
    ) -> GenerationResult:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("No choices returned by provider", details={"status_code": 200})
        choice = choices[0]
        msg = choice.get("message") or {}
        finish = self._map_finish_reason(choice.get("finish_reason"))
        kwargs: dict[str, Any] = {
            "text": msg.get("content") or "",
            "model": data.get("model") or request.model or "unknown",
            "finish_reason": finish,
            "usage": self._map_usage(data.get("usage")),
            "tool_calls": msg.get("tool_calls") or None,
        }
        if isinstance(data.get("id"), str):
            kwargs["id"] = data["id"]
        return GenerationResult(**kwargs)

    @staticmethod
    def _map_finish_reason(reason: str | None) -> FinishReason:
        if not reason:
            return "stop"
        return _OPENAI_FINISH_MAP.get(reason, "stop")

    @staticmethod
    def _map_usage(usage: dict[str, Any] | None) -> Usage | None:
        if not usage:
            return None
        return Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
