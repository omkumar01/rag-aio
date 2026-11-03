"""Tests for ``build_mock_services`` — the fully-offline mock wiring.

The mock stack (``MockEmbedder``, ``InMemoryVectorStore``, ``StubGenerator``,
…) used to live only in the ``rag-aio`` facade; member-only consumers of
``rag-aio-orchestrator`` must be able to build it directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rag_cache.backends.memory import MemoryCache
from rag_cache.stats import InstrumentedCache
from rag_db_handler.memory_store import InMemoryVectorStore
from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
from rag_orchestrator import Orchestrator, PipelineConfig, ingest
from rag_orchestrator.mock_services import StubGenerator, build_mock_services
from rag_orchestrator.orchestrator import AskResult
from rag_orchestrator.services import OrchestratorServices

_local_default = PipelineConfig.local_default()
_MOCK_CONFIG = _local_default.model_copy(
    update={"embedding": _local_default.embedding.model_copy(update={"policy": "mock"})}
)


# --------------------------------------------------------------------------- #
# Component wiring
# --------------------------------------------------------------------------- #


def test_build_mock_services_returns_services_bag() -> None:
    services = build_mock_services()
    assert isinstance(services, OrchestratorServices)


def test_build_mock_services_wires_mock_components() -> None:
    services = build_mock_services()
    assert isinstance(services.embedder, MockEmbedder)
    assert isinstance(services.sparse_embedder, MockSparseEmbedder)
    assert isinstance(services.vector_store, InMemoryVectorStore)
    assert isinstance(services.generator, StubGenerator)
    assert isinstance(services.cache, InstrumentedCache)
    assert isinstance(services.cache._inner, MemoryCache)
    assert services.document_store is None
    assert services.retrievers, "at least one retriever must be wired"
    assert services.reranker is not None
    assert services.context_builder is not None
    assert services.query_engine is not None
    assert services.ingestion_pipeline is not None
    assert services.embedding_pipeline is not None


def test_build_mock_services_lazy_export_via_package_root() -> None:
    import rag_orchestrator

    assert rag_orchestrator.build_mock_services is build_mock_services
    assert rag_orchestrator.StubGenerator is StubGenerator


def test_build_mock_services_vector_size_parameter() -> None:
    services = build_mock_services(vector_size=128)
    assert isinstance(services.vector_store, InMemoryVectorStore)
    assert services.vector_store._vector_size == 128
    assert isinstance(services.embedder, MockEmbedder)
    assert services.embedder._dim == 128


def test_build_mock_services_cache_max_items_parameter() -> None:
    services = build_mock_services(cache_max_items=7)
    assert isinstance(services.cache, InstrumentedCache)
    assert isinstance(services.cache._inner, MemoryCache)
    assert services.cache._inner._max_items == 7
    # Default matches the facade's historic value.
    default_services = build_mock_services()
    assert isinstance(default_services.cache._inner, MemoryCache)
    assert default_services.cache._inner._max_items == 4096


# --------------------------------------------------------------------------- #
# Generator behavior
# --------------------------------------------------------------------------- #


def test_stub_generator_streams_answer_words() -> None:
    import asyncio

    from rag_core.generation import GenerationRequest

    async def _collect() -> str:
        gen = StubGenerator()
        request = GenerationRequest(messages=[{"role": "user", "content": "hello world question"}])
        chunks = [chunk async for chunk in gen.stream(request)]
        return "".join(chunks)

    collected = asyncio.run(_collect())
    assert "Based on the provided context." in collected


def test_stub_generator_extracts_context_from_prompt() -> None:
    import asyncio

    from rag_core.generation import GenerationRequest

    async def _answer() -> str:
        gen = StubGenerator()
        request = GenerationRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Context:\nUsers must log in.\n\nQuestion: How do I log in?",
                },
            ]
        )
        result = await gen.generate(request)
        return result.text

    assert asyncio.run(_answer()) == "Context:\nUsers must log in."


# --------------------------------------------------------------------------- #
# End-to-end through the orchestrator
# --------------------------------------------------------------------------- #


async def test_mock_services_ingest_then_ask(tmp_path: Path) -> None:
    """Full offline round-trip with only rag-orchestrator members."""
    services = build_mock_services()
    doc_path = tmp_path / "auth.md"
    doc_path.write_text(
        "# Authentication\n\n"
        "Users must provide a username and password to log in.\n"
        "Passwords must be at least eight characters long.\n",
        encoding="utf-8",
    )

    doc = await ingest(services, str(doc_path), config=_MOCK_CONFIG)
    assert doc.id
    assert doc.content_hash

    orchestrator = Orchestrator(services, _MOCK_CONFIG)
    result = await orchestrator.ask("What are the authentication requirements?")
    assert isinstance(result, AskResult)
    assert result.answer, "answer must be non-empty"
    assert result.citations, "citations must be derived from the context"
    for stage in ("query", "retrieve", "rerank", "context", "generate", "total_ms"):
        assert stage in result.timings_ms


async def test_mock_services_ask_streaming(tmp_path: Path) -> None:
    services = build_mock_services()
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("The capital of France is Paris. ", encoding="utf-8")
    await ingest(services, str(doc_path), config=_MOCK_CONFIG)

    orchestrator = Orchestrator(services, _MOCK_CONFIG)
    stream = await orchestrator.ask("What is the capital of France?", stream=True)
    assert not isinstance(stream, AskResult)
    collected = ""
    async for chunk in stream:  # type: ignore[union-attr]
        collected += chunk
    assert "Paris" in collected


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
