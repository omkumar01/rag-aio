"""Stage 5 — Generation: answer a question from an assembled context.

Builds a :class:`rag_core.GenerationRequest` (system + user message carrying
the rendered context) and sends it to an OpenAI-compatible endpoint via
:class:`rag_generation.OpenAICompatibleProvider` — the same adapter the
orchestrator uses for LM Studio. Shows a non-streaming completion, then a
streamed one.

Requires LM Studio (or any OpenAI-compatible server) on
``http://localhost:1234/v1``; override with ``OPENAI_BASE_URL``, optionally
``OPENAI_API_KEY`` / ``OPENAI_MODEL``. Exits early with a clear message when
the endpoint is unreachable. This is the only per-stage example that is not
fully offline.

Run: uv run python examples/stages/generate.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from rag_core import GenerationRequest, Message
from rag_generation import OpenAICompatibleProvider, collect_stream

# The shared network guard lives one directory up (examples/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _net import EndpointBlocked, validate_base_url

BASE_URL = validate_base_url(os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))

CONTEXT = """[1] The activation code for the reactor is RAG-8891. Keep it confidential.
[2] Inspect coolant pumps every ninety days; replace filters quarterly."""

QUESTION = "What is the activation code for the reactor?"


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


async def main() -> None:
    try:
        reachable = await _endpoint_reachable(BASE_URL)
    except EndpointBlocked as exc:
        print(f"Endpoint blocked: {exc}")
        return
    if not reachable:
        print(
            f"No OpenAI-compatible server reachable at {BASE_URL}.\n"
            "Start LM Studio (or set OPENAI_BASE_URL) and re-run this example. "
            "All other examples/stages scripts run fully offline."
        )
        return

    provider = OpenAICompatibleProvider(
        base_url=BASE_URL,
        api_key_provider=lambda: os.environ.get("OPENAI_API_KEY"),
    )
    models = await provider.list_models()
    chat_models = [m for m in models if not any(tag in m.lower() for tag in ("embed", "rerank"))]
    model = os.environ.get("OPENAI_MODEL") or (chat_models[0] if chat_models else None)
    print(f"Endpoint: {BASE_URL} | model: {model or '(server default)'}")

    request = GenerationRequest(
        messages=[
            Message(role="system", content="Answer strictly from the provided context."),
            Message(role="user", content=f"Context:\n\n{CONTEXT}\n\nQuestion: {QUESTION}"),
        ],
        model=model,
        temperature=0.2,
        max_tokens=120,
    )

    result = await provider.complete(request)
    print(f"\nAnswer (finish={result.finish_reason}): {result.text}")

    streamed = await collect_stream(provider.stream(request))
    print(f"Streamed answer: {streamed}")


if __name__ == "__main__":
    asyncio.run(main())
