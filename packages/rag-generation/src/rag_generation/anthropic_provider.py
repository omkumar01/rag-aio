"""Anthropic adapter (Bedrock-style ``/v1/messages`` endpoint)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_core.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    Usage,
)

from .http import ApiKeyProvider, ProviderHttpClient

# Anthropic requires max_tokens; default when the request omits it.
_DEFAULT_MAX_TOKENS = 4096
_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicProvider(ProviderHttpClient):
    """Adapter for Anthropic's ``/v1/messages`` API.

    Uses ``x-api-key`` plus the ``anthropic-version`` header (no bearer token).
    System messages are hoisted to the top-level ``system`` body field; remaining
    messages are forwarded to ``messages``.
    """

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com",
        api_key_provider: ApiKeyProvider | None = None,
        *,
        timeout_s: float = 120.0,
        transport: Any = None,
    ) -> None:
        super().__init__(base_url, api_key_provider, timeout_s=timeout_s, transport=transport)

    def auth_headers(self) -> dict[str, str]:
        key = self._api_key_provider()
        if not key:
            return {}
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }

    # -- public API ----------------------------------------------------------

    async def complete(self, request: GenerationRequest) -> GenerationResult:
        payload, system = self._build_payload(request)
        if system:
            payload["system"] = system
        data = await self.post_json("/v1/messages", payload)
        return self._parse_message(data, request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        payload, system = self._build_payload(request)
        if system:
            payload["system"] = system
        payload["stream"] = True
        async for event in self.stream_sse("/v1/messages", payload):
            etype = event.get("type")
            if etype == "content_block_delta":
                delta = event.get("delta") or {}
                text = delta.get("text") or delta.get("partial_text") or ""
                if text:
                    yield text

    # -- payload -------------------------------------------------------------

    def _build_payload(self, request: GenerationRequest) -> tuple[dict[str, Any], str]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": request.model or "claude",
            "messages": messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        if request.tools:
            payload["tools"] = request.tools
        return payload, "\n".join(system_parts)

    # -- response ------------------------------------------------------------

    def _parse_message(self, data: dict[str, Any], request: GenerationRequest) -> GenerationResult:
        blocks = data.get("content") or []
        text = self._join_text_blocks(blocks)
        finish = _STOP_REASON_MAP.get(data.get("stop_reason", ""), "stop")
        kwargs: dict[str, Any] = {
            "text": text,
            "model": data.get("model") or request.model or "unknown",
            "finish_reason": finish,
            "usage": self._map_usage(data.get("usage")),
            "tool_calls": self._extract_tool_calls(blocks),
        }
        if isinstance(data.get("id"), str):
            kwargs["id"] = data["id"]
        return GenerationResult(**kwargs)

    @staticmethod
    def _join_text_blocks(blocks: list[Any]) -> str:
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    @staticmethod
    def _extract_tool_calls(blocks: list[Any]) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "tool_use":
                calls.append(
                    {
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": b.get("input", {}),
                        },
                    }
                )
        return calls or None

    @staticmethod
    def _map_usage(usage: dict[str, Any] | None) -> Usage | None:
        if not usage:
            return None
        prompt = int(usage.get("input_tokens", 0)) + int(
            usage.get("cache_creation_input_tokens", 0)
        )
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=int(usage.get("output_tokens", 0)),
        )
