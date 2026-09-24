"""End-to-end smoke: ingest a real PDF -> ask -> answer with citations.

Tier 1 runs fully offline against the mock profile (real parse -> chunk ->
embed -> retrieve -> rerank -> context -> generate chain, stub generator).
Tier 2 targets a local LM Studio server and auto-skips when it is unreachable.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from rag_aio import RAG, RAGConfig
from rag_core.errors import ProviderError
from rag_orchestrator import Orchestrator
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.orchestrator import AskResult

pytestmark = pytest.mark.e2e


def _make_pdf(path: Path) -> Path:
    """Write a small 2-page PDF with distinctive content to *path*."""
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page(width=300, height=300)
    p1.insert_text(
        (30, 50),
        "Reactor Operations Manual.\n"
        "The activation code for the reactor\n"
        "is RAG-8891. Keep it confidential.",
        fontsize=9,
    )
    p2 = doc.new_page(width=300, height=300)
    p2.insert_text(
        (30, 50),
        "Maintenance schedule:\n"
        "inspect coolant pumps every ninety days.\n"
        "Replace filters quarterly.",
        fontsize=9,
    )
    doc.set_metadata({"title": "Reactor Manual", "author": "rag-aio e2e"})
    path.write_bytes(doc.tobytes())
    doc.close()
    return path


# --------------------------------------------------------------------------- #
# Tier 1 — fully offline
# --------------------------------------------------------------------------- #


async def test_e2e_pdf_ingest_ask_citations(tmp_path: Path) -> None:
    """PDF ingest -> ask -> non-empty answer citing the ingested document."""
    pdf_path = _make_pdf(tmp_path / "manual.pdf")

    async with RAG.from_config(RAGConfig.mock()) as rag:
        doc = await rag.ingest(pdf_path)
        assert doc.id
        assert "RAG-8891" in doc.text

        result = await rag.ask("What is the activation code for the reactor?")
        assert isinstance(result, AskResult)
        assert result.answer, "answer must be non-empty"
        assert "RAG-8891" in result.answer
        assert result.citations, "citations must reference the ingested document"
        assert any("manual.pdf" in (c.source_uri or "") for c in result.citations), (
            f"citations should point at the ingested PDF, got {result.citations}"
        )
        for stage in ("query", "retrieve", "rerank", "context", "generate", "total_ms"):
            assert stage in result.timings_ms


async def test_e2e_pdf_ask_streaming(tmp_path: Path) -> None:
    """Streaming ask over an ingested PDF yields non-empty deltas."""
    pdf_path = _make_pdf(tmp_path / "manual.pdf")

    async with RAG.from_config(RAGConfig.mock()) as rag:
        await rag.ingest(pdf_path)
        stream = await rag.ask("What is the activation code?", stream=True)
        assert not isinstance(stream, AskResult)
        collected = ""
        async for chunk in stream:  # type: ignore[union-attr]
            collected += chunk
        assert collected.strip(), "streamed output must be non-empty"


# --------------------------------------------------------------------------- #
# Tier 2 — local LM Studio (auto-skip)
# --------------------------------------------------------------------------- #


def _lm_studio_reachable(base_url: str = "http://localhost:1234/v1") -> bool:
    """True only when LM Studio answers a minimal completion request.

    ``/models`` lists downloaded models even when none are loaded, so the
    probe actually generates one token; any failure means "not usable".
    """
    import httpx

    try:
        models = httpx.get(f"{base_url}/models", timeout=2.0).json().get("data", [])
        if not models:
            return False
        probe = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": models[0].get("id", ""),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=10.0,
        )
    except Exception:
        return False
    return probe.status_code == 200


def _list_served_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
    """Model ids served at *base_url* (empty when the endpoint is unusable)."""
    import httpx

    try:
        data = httpx.get(f"{base_url}/models", timeout=2.0).json()
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [m["id"] for m in data.get("data", []) if isinstance(m.get("id"), str)]


def _pick_chat_model(models: list[str]) -> str | None:
    """First model id that looks like a chat model (skip embed/rerank).

    Mirrors ``examples/fully_local.py`` so the test does not depend on the
    exact model id the default profile routes to (``qwen3-8b``).
    """
    return next(
        (m for m in models if not any(tag in m.lower() for tag in ("embed", "rerank"))),
        None,
    )


async def _close_stores(services: object) -> None:
    """Best-effort close of backend stores (keeps Windows temp cleanup happy)."""
    for store in (
        getattr(services, "vector_store", None),
        getattr(services, "document_store", None),
    ):
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            outcome = close_fn()
            if inspect.isawaitable(outcome):
                await outcome


async def test_e2e_lm_studio_ask(tmp_path: Path) -> None:
    """Full local stack: fastembed + Qdrant local + sqlite + LM Studio."""
    if not _lm_studio_reachable():
        pytest.skip("LM Studio not reachable at localhost:1234")

    from rag_orchestrator import load_local_services
    from rag_orchestrator.ingestion import ingest as _ingest

    pdf_path = _make_pdf(tmp_path / "manual.pdf")
    services = load_local_services(
        qdrant_path=str(tmp_path / "qdrant"),
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'rag.db'}",
        lm_studio_url="http://localhost:1234/v1",
    )
    config = PipelineConfig.local_default()
    # The default profile routes to a static model id (qwen3-8b) that a given
    # LM Studio instance may not serve; point the generation stage at a model
    # the server actually serves (same auto-detection as examples/fully_local.py).
    model = _pick_chat_model(_list_served_models())
    if model is not None:
        config.generation = config.generation.model_copy(
            update={"overrides": {**config.generation.overrides, "model": model}}
        )
    try:
        await _ingest(services, str(pdf_path), config=config)
        orch = Orchestrator(services, config)
        try:
            result = await orch.ask("What is the activation code for the reactor?")
        except ProviderError as exc:
            pytest.skip(f"LM Studio is up but no served model could generate: {exc}")
    finally:
        await _close_stores(services)

    assert isinstance(result, AskResult)
    assert result.query_id
    assert result.answer, "LM Studio must return a non-empty answer"
    # NB: unlike Tier 1 we do not assert "RAG-8891" in the answer — a live LLM
    # is not guaranteed to echo the code verbatim, only the stub is.
    assert result.citations, "citations must reference the ingested document"
    assert any("manual.pdf" in (c.source_uri or "") for c in result.citations), (
        f"citations should point at the ingested PDF, got {result.citations}"
    )
    assert all(c.citation_id and c.document_id and c.chunk_id for c in result.citations), (
        "every citation must carry provenance ids"
    )
    for stage in ("query", "retrieve", "rerank", "context", "generate", "total_ms"):
        assert stage in result.timings_ms
