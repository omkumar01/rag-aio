"""Ollama native adapter (``/api/chat`` NDJSON streaming)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_core.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    Usage,
)
from rag_core.protocols import LLMProvider

from .http import ApiKeyProvider, ProviderHttpClient

_DEFAULT_MODEL = "llama3"
_DONE_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "error": "error",
}


def _map_done_reason(reason: str | None) -> FinishReason:
    if not reason:
        return "stop"
    return _DONE_REASON_MAP.get(reason, "stop")


class OllamaProvider(ProviderHttpClient, LLMProvider):
    """Adapter for Ollama's native ``/api/chat`` endpoint (NDJSON streaming).

    Ollama does not use an auth header by default; ``api_key_provider`` is accepted
    for API symmetry but is ignored unless a self-hosted gateway requires it.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key_provider: ApiKeyProvider | None = None,
        *,
        timeout_s: float = 120.0,
        transport: Any = None,
    ) -> None:
        super().__init__(base_url, api_key_provider, timeout_s=timeout_s, transport=transport)

    # -- public API ----------------------------------------------------------

    async def complete(self, request: GenerationRequest) -> GenerationResult:
        payload = self._build_payload(request, stream=False)
        data = await self.post_json("/api/chat", payload)
        message = data.get("message") or {}
        finish = _map_done_reason(data.get("done_reason"))
        return GenerationResult(
            text=message.get("content") or "",
            model=data.get("model") or request.model or _DEFAULT_MODEL,
            finish_reason=finish,
            usage=self._map_usage(data),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        payload = self._build_payload(request, stream=True)
        async for chunk in self.stream_ndjson("/api/chat", payload):
            if chunk.get("done"):
                continue
            message = chunk.get("message") or {}
            content = message.get("content")
            if content:
                yield content

    # -- helpers -------------------------------------------------------------

    def _build_payload(self, request: GenerationRequest, stream: bool) -> dict[str, Any]:
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system" or m.content
        ]
        payload: dict[str, Any] = {
            "model": request.model or _DEFAULT_MODEL,
            "messages": messages,
            "stream": stream,
        }
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = request.stop
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        payload["options"] = options
        return payload

    @staticmethod
    def _map_usage(data: dict[str, Any]) -> Usage | None:
        prompt = int(data.get("prompt_eval_count", 0))
        completion = int(data.get("eval_count", 0))
        if prompt == 0 and completion == 0:
            return None
        return Usage(prompt_tokens=prompt, completion_tokens=completion)
