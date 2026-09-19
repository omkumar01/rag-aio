"""Google Gemini adapter (``/v1beta/models/{model}:generateContent``)."""

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

_DEFAULT_MODEL = "gemini-1.5-flash"
# role used inside `contents`; rag-core `assistant` maps to Gemini `model`.
_ROLE_MAP: dict[str, str] = {
    "system": "user",
    "user": "user",
    "assistant": "model",
    "tool": "tool",
}
_REASON_MAP: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
}


class GeminiProvider(ProviderHttpClient, LLMProvider):
    """Adapter for Gemini's ``generateContent`` REST API.

    Authentication is via the ``x-goog-api-key`` header (cleaner than query param).
    The model name is taken from the request (defaulting to a sensible flash
    model) and embedded in the URL path.
    """

    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com",
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
        return {"x-goog-api-key": key}

    # -- public API ----------------------------------------------------------

    async def complete(self, request: GenerationRequest) -> GenerationResult:
        path = self._path(request.model, stream=False)
        payload = self._build_payload(request)
        data = await self.post_json(path, payload)
        return self._parse_generate_content(data, request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        path = self._path(request.model, stream=True)
        payload = self._build_payload(request)
        async for chunk in self.stream_sse(path, payload):
            candidates = chunk.get("candidates") or []
            if not candidates:
                continue
            parts = candidates[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if p.get("text"))
            if text:
                yield text

    # -- helpers -------------------------------------------------------------

    def _path(self, model: str | None, stream: bool) -> str:
        model = model or _DEFAULT_MODEL
        suffix = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
        return f"/v1beta/models/{model}{suffix}"

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            contents.append({"role": _ROLE_MAP.get(m.role, "user"), "parts": [{"text": m.content}]})

        config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            config["topP"] = request.top_p
        if request.stop:
            config["stopSequences"] = request.stop

        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        if request.tools:
            payload["tools"] = request.tools
        return payload

    def _parse_generate_content(
        self, data: dict[str, Any], request: GenerationRequest
    ) -> GenerationResult:
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(
                "No candidates returned by provider",
                details={"status_code": 200, "response_snippet": str(data)[:200]},
            )
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if p.get("text"))
        finish = _REASON_MAP.get(cand.get("finishReason", ""), "stop")
        kwargs: dict[str, Any] = {
            "text": text,
            "model": data.get("modelVersion") or request.model or _DEFAULT_MODEL,
            "finish_reason": finish,
            "usage": self._map_usage(data.get("usageMetadata")),
        }
        return GenerationResult(**kwargs)

    @staticmethod
    def _map_usage(usage: dict[str, Any] | None) -> Usage | None:
        if not usage:
            return None
        return Usage(
            prompt_tokens=int(usage.get("promptTokens", 0)),
            completion_tokens=int(usage.get("candidatesTokens", 0)),
        )
