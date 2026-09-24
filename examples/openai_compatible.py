"""Talk directly to any OpenAI-compatible chat-completions endpoint.

Demonstrates :class:`rag_generation.OpenAICompatibleProvider` (the reference
adapter for LM Studio, vLLM, Ollama's OpenAI layer, llama.cpp, ...) used
standalone and wrapped in a :class:`rag_generation.GenerationService` with
retry/fallback policy. Shows a non-streaming completion and a streamed one.

Configuration (environment variables, never hardcoded):
  OPENAI_BASE_URL  endpoint base, default http://localhost:1234/v1 (LM Studio)
  OPENAI_API_KEY   optional bearer token, read per-request
  OPENAI_MODEL     optional model id; auto-detected from /models when unset

Exits early with a clear message when the endpoint is unreachable.

Run: uv run python examples/openai_compatible.py
"""

from __future__ import annotations

import asyncio
import os

from _net import EndpointBlocked, validate_base_url
from rag_core import GenerationRequest, Message
from rag_generation import GenerationService, OpenAICompatibleProvider, collect_stream

BASE_URL = validate_base_url(os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))

CONTEXT = """[1] The Alpha staging cluster activation key is ALP-2026-KX.
Keys rotate every ninety days and the on-call engineer owns rotation.
[2] The Bravo pipeline feeds the Alpha dashboards and deploys weekly."""

QUESTION = "What is the activation key for the Alpha staging cluster?"


async def _endpoint_reachable(base_url: str) -> bool:
    """Return True when an OpenAI-compatible server answers at *base_url*."""
    validate_base_url(base_url)
    provider = OpenAICompatibleProvider(
        base_url=base_url, api_key_provider=lambda: None, timeout_s=2.0
    )
    try:
        await provider.list_models()
    except Exception:
        return False
    finally:
        await provider.aclose()
    return True


def _build_request(model: str | None) -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role="system", content="Answer strictly from the provided context."),
            Message(role="user", content=f"Context:\n\n{CONTEXT}\n\nQuestion: {QUESTION}"),
        ],
        model=model,
        temperature=0.2,
        max_tokens=200,
    )


def _pick_chat_model(models: list[str]) -> str | None:
    """Pick the first model id that looks like a chat model (skip embed/rerank)."""
    return next(
        (m for m in models if not any(tag in m.lower() for tag in ("embed", "rerank"))),
        None,
    )


async def main() -> None:
    try:
        reachable = await _endpoint_reachable(BASE_URL)
    except EndpointBlocked as exc:
        print(f"Endpoint blocked: {exc}")
        return
    if not reachable:
        print(
            f"No OpenAI-compatible server reachable at {BASE_URL}.\n"
            "Start LM Studio (or set OPENAI_BASE_URL) and re-run this example."
        )
        return

    # Secrets stay secret: the key is resolved per-request from the environment.
    provider = OpenAICompatibleProvider(
        base_url=BASE_URL,
        api_key_provider=lambda: os.environ.get("OPENAI_API_KEY"),
    )
    models = await provider.list_models()
    print(f"Endpoint: {BASE_URL}")
    print(f"Available models: {models or '(none reported)'}")

    model = os.environ.get("OPENAI_MODEL") or _pick_chat_model(models)
    request = _build_request(model)

    service = GenerationService(
        {"local": provider},
        max_retries=1,
        fallback_names=[],
    )
    result = await service.generate(request)
    print(f"\nCompletion (model={result.model}, finish={result.finish_reason}):")
    print(result.text)
    if result.usage is not None:
        print(
            f"Tokens: prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens}"
        )

    print("\nStreaming deltas:")
    streamed = await collect_stream(provider.stream(request))
    print(streamed)


if __name__ == "__main__":
    asyncio.run(main())
