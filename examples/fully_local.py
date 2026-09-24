"""End-to-end example against a fully local (non-mock) stack.

Wires the real components — FastEmbed dense + sparse embeddings, Qdrant in
embedded local mode, a SQLite document store, and LM Studio for generation —
via ``rag_orchestrator.load_local_services``, ingests a small document, and
asks a question.

Requires LM Studio (or any OpenAI-compatible server) listening on
``http://localhost:1234/v1`` (override with ``LM_STUDIO_URL``). The script
exits early with a clear message when the server is unreachable. A loaded chat
model is auto-detected from ``/models`` (pin with ``LM_STUDIO_MODEL``); the
first run downloads the embedding model (~30 MB) unless already cached.

Run: uv run python examples/fully_local.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
from pathlib import Path

from _net import EndpointBlocked, validate_base_url
from rag_generation import OpenAICompatibleProvider
from rag_orchestrator import (
    AskResult,
    Orchestrator,
    OrchestratorServices,
    PipelineConfig,
    ingest,
    load_local_services,
)

BASE_URL = validate_base_url(os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1"))

SAMPLE = """Reactor Operations Manual.

The activation code for the reactor is RAG-8891. Keep it confidential.

Maintenance schedule: inspect coolant pumps every ninety days and
replace filters quarterly. Log all interventions in the operations journal.
"""


async def _probe_endpoint(base_url: str) -> list[str] | None:
    """Return the model ids served at *base_url*, or None when unreachable."""
    validate_base_url(base_url)
    provider = OpenAICompatibleProvider(base_url=base_url, timeout_s=2.0)
    try:
        return await provider.list_models()
    except Exception:
        return None
    finally:
        await provider.aclose()


def _pick_chat_model(models: list[str]) -> str | None:
    """Pick the first model id that looks like a chat model (skip embed/rerank)."""
    return next(
        (m for m in models if not any(tag in m.lower() for tag in ("embed", "rerank"))),
        None,
    )


async def _close_stores(services: OrchestratorServices) -> None:
    """Best-effort close of backend resources (keeps Windows temp cleanup happy)."""
    for store in (services.vector_store, services.document_store):
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            outcome = close_fn()
            if inspect.isawaitable(outcome):
                await outcome


async def main() -> None:
    try:
        models = await _probe_endpoint(BASE_URL)
    except EndpointBlocked as exc:
        print(f"Endpoint blocked: {exc}")
        return
    if models is None:
        print(
            f"LM Studio is not reachable at {BASE_URL}.\n"
            "Start LM Studio (or set LM_STUDIO_URL to an OpenAI-compatible "
            "endpoint) and re-run this example. For a fully offline demo, see "
            "examples/quickstart.py instead."
        )
        return
    print(f"Generation backend: {BASE_URL} ({len(models)} model(s) loaded)")

    with tempfile.TemporaryDirectory(
        prefix="rag-aio-fully-local-", ignore_cleanup_errors=True
    ) as tmp:
        services = load_local_services(
            qdrant_path=str(Path(tmp) / "qdrant"),
            db_url=f"sqlite+aiosqlite:///{Path(tmp) / 'rag.db'}",
            lm_studio_url=BASE_URL,
        )
        config = PipelineConfig.local_default()
        # The default profile routes to a static model id; point the generation
        # stage at a model the server actually serves so the request succeeds.
        model = os.environ.get("LM_STUDIO_MODEL") or _pick_chat_model(models)
        if model is not None:
            config.generation = config.generation.model_copy(
                update={"overrides": {**config.generation.overrides, "model": model}}
            )
            print(f"Generation model: {model}")

        doc_path = Path(tmp) / "reactor.md"
        doc_path.write_text(SAMPLE, encoding="utf-8")
        document = await ingest(services, str(doc_path), config=config)
        print(f"Ingested: {document.source_uri} ({len(document.text)} chars)")

        orchestrator = Orchestrator(services, config)
        result = await orchestrator.ask("What is the activation code for the reactor?")
        assert isinstance(result, AskResult), "expected a non-streaming AskResult"

        print("\nAnswer:")
        print(result.answer)
        print("\nCitations:")
        for citation in result.citations:
            print(f"  [{citation.citation_id}] {citation.source_uri or citation.document_id}")
        print("\nStage timings (ms):")
        for stage, ms in result.timings_ms.items():
            print(f"  {stage:12s} {ms:8.2f}")

        await _close_stores(services)


if __name__ == "__main__":
    asyncio.run(main())
