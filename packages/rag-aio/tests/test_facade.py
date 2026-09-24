"""End-to-end tests for the RAG facade using fully-offline mock services."""

from __future__ import annotations

from pathlib import Path

import aio_fakes
from rag_aio import RAG, RAGConfig
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.orchestrator import AskResult

# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_from_config_builds_services() -> None:
    rag = RAG.from_config(RAGConfig.mock())
    assert rag.services is not None
    assert rag.services.query_engine is not None
    assert rag.services.generator is not None
    assert rag.services.vector_store is not None


def test_default_config_uses_fastembed_backend() -> None:
    """Non-mock backend requires real services; just verify the config."""
    cfg = RAGConfig()
    assert cfg.embedder.backend == "fastembed"


def test_rag_with_stub_services() -> None:
    """RAG can be constructed directly with fakes-based services."""
    services = aio_fakes.make_services(generator=aio_fakes.StubGenerator(answer="hello world"))
    rag = RAG(services, PipelineConfig())
    assert rag is not None


# --------------------------------------------------------------------------- #
# Ingestion + asking
# --------------------------------------------------------------------------- #


async def test_ingest_then_ask(tmp_path: Path) -> None:
    """Full offline round-trip: ingest a document, then ask a question."""
    config = RAGConfig.mock()
    rag = RAG.from_config(config)

    doc_path = tmp_path / "auth.md"
    doc_path.write_text(
        "# Authentication\n\n"
        "Users must provide a username and password to log in.\n"
        "Passwords must be at least eight characters long.\n"
        "Multi-factor authentication is strongly recommended.\n",
        encoding="utf-8",
    )

    doc = await rag.ingest(doc_path)
    assert doc.id
    assert doc.content_hash
    assert "Authentication" in doc.text

    result = await rag.ask("What are the authentication requirements?")
    assert isinstance(result, AskResult)
    assert result.answer, "answer must be non-empty"
    assert result.citations, "citations must be derived from the context"
    assert result.timings_ms, "timings must be recorded"
    for stage in ("query", "retrieve", "rerank", "context", "generate", "total_ms"):
        assert stage in result.timings_ms


async def test_ask_streaming(tmp_path: Path) -> None:
    """Streaming ask returns deltas that reassemble to the full answer."""
    config = RAGConfig.mock()
    rag = RAG.from_config(config)

    doc_path = tmp_path / "doc.md"
    doc_path.write_text("The capital of France is Paris. ", encoding="utf-8")
    await rag.ingest(doc_path)

    stream = await rag.ask("What is the capital of France?", stream=True)
    assert not isinstance(stream, AskResult)
    collected = ""
    async for chunk in stream:  # type: ignore[union-attr]
        collected += chunk
    assert collected.strip(), "streamed output must be non-empty"


async def test_ask_with_overrides(tmp_path: Path) -> None:
    """Per-request overrides are merged into the pipeline config."""
    config = RAGConfig.mock()
    rag = RAG.from_config(config)

    doc_path = tmp_path / "doc.md"
    doc_path.write_text("Content about machine learning. " * 20, encoding="utf-8")
    await rag.ingest(doc_path)

    result = await rag.ask("What is this about?", overrides={"top_k": 5})
    assert isinstance(result, AskResult)
    assert result.answer


# --------------------------------------------------------------------------- #
# Async context manager
# --------------------------------------------------------------------------- #


async def test_async_context_manager(tmp_path: Path) -> None:
    """RAG can be used as an async context manager."""
    config = RAGConfig.mock()
    async with RAG.from_config(config) as rag:
        assert rag is not None
        result = await rag.ask("test question")
        assert isinstance(result, AskResult)


# --------------------------------------------------------------------------- #
# App property
# --------------------------------------------------------------------------- #


def test_app_property_returns_fastapi() -> None:
    rag = RAG.from_config(RAGConfig.mock())
    application = rag.app
    from fastapi import FastAPI

    assert isinstance(application, FastAPI)
    assert application.title


# --------------------------------------------------------------------------- #
# Direct construction with injected services
# --------------------------------------------------------------------------- #


async def test_rag_ask_with_injected_services() -> None:
    """RAG works with hand-built stub services (no mock wiring)."""
    services = aio_fakes.make_services(generator=aio_fakes.StubGenerator(answer="injected answer"))
    rag = RAG(services, PipelineConfig())
    result = await rag.ask("anything?")
    assert isinstance(result, AskResult)
    assert result.answer == "injected answer"
    assert result.citations
    assert result.timings_ms


def test_from_config_creates_distinct_services() -> None:
    """Each from_config call builds an independent services bag."""
    rag1 = RAG.from_config(RAGConfig.mock())
    rag2 = RAG.from_config(RAGConfig.mock())
    assert rag1.services is not rag2.services
