"""Benchmark scenario tests for the rag-aio pipeline stages.

Every test here is marked ``benchmark`` and is meant to run explicitly via
``uv run pytest benchmarks -m benchmark``. Workloads are deliberately small
(synthetic docs, mock embedders, in-memory stores) so the whole suite completes
in well under a minute on a laptop; the generation benchmark auto-skips unless
a local OpenAI-compatible endpoint answers at ``http://localhost:1234/v1``.

Each test runs its scenario through :func:`benchmarks.harness.run_benchmark`,
asserts the recorded statistics are sane, and appends the
:class:`~benchmarks.harness.BenchmarkResult` to the session-scoped report
writer flushed by ``conftest.py`` at session end.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from rag_aio import RAG, RAGConfig
from rag_context.builder import ContextBuilderImpl
from rag_context.config import ContextConfig
from rag_context.tokenizer import ContextTokenizer, WhitespaceCounter
from rag_core.generation import GenerationRequest, Message
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit
from rag_db_handler import InMemoryVectorStore
from rag_doc_handler import DedupIndex, default_pipeline
from rag_embedder import (
    ChunkerConfig,
    FastEmbedDense,
    MockEmbedder,
    MockSparseEmbedder,
    SimpleTokenizer,
    create_chunker,
)
from rag_generation import GenerationService, OpenAICompatibleProvider
from rag_orchestrator import AskResult
from rag_rerank import HeuristicReranker, RerankConfig, RerankPipeline
from rag_retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
    RetrievalConfig,
)

from benchmarks.harness import ReportWriter, run_benchmark
from benchmarks.support import (
    check_sane,
    fastembed_model_cached,
    generate_chunk_text,
    generate_doc_text,
)

pytestmark = pytest.mark.benchmark

DIM = 384
N_CHUNKS = 200
QUERY_TEXT = "retrieval augmented generation pipeline latency throughput"


def _make_hits(count: int) -> list[RetrievalHit]:
    """Build *count* retrieval hits carrying synthetic text for rerank/context."""
    return [
        RetrievalHit(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i // 10}",
            score=1.0 - i / (count * 2.0),
            normalized_score=1.0 - i / (count * 2.0),
            rank=i,
            strategy="dense",
            model="mock-dense",
            text=generate_chunk_text(i),
        )
        for i in range(count)
    ]


async def test_doc_throughput(tmp_path: Path, report_writer: ReportWriter) -> None:
    """Parse + chunk 30 generated text docs through the doc-handler pipeline."""
    n_docs = 30
    paths: list[Path] = []
    for i in range(n_docs):
        path = tmp_path / f"doc_{i:03d}.txt"
        path.write_text(generate_doc_text(i), encoding="utf-8")
        paths.append(path)

    pipeline = default_pipeline()
    chunker = create_chunker(
        ChunkerConfig(strategy="recursive", chunk_size=256, overlap=32), SimpleTokenizer()
    )

    async def scenario() -> int:
        dedup = DedupIndex()
        total_chunks = 0
        for path in paths:
            document, _ = await pipeline.ingest_file(path, dedup=dedup)
            chunks = await chunker.chunk(document)
            total_chunks += len(chunks)
        return total_chunks

    result = await run_benchmark(
        "doc_throughput",
        scenario,
        iterations=10,
        warmup=1,
        description=f"parse + chunk {n_docs} generated text docs "
        "(rag_doc_handler default_pipeline + recursive chunker)",
    )
    check_sane(result, expected_iterations=10)
    report_writer.add(result)


async def test_embedding_throughput(report_writer: ReportWriter) -> None:
    """Chunk-embed 200 synthetic chunks (mock dense + sparse) into an in-memory store."""
    texts = [generate_chunk_text(i) for i in range(N_CHUNKS)]
    embedder = MockEmbedder(dim=DIM)
    sparse_embedder = MockSparseEmbedder()

    async def scenario() -> int:
        store = InMemoryVectorStore(vector_size=DIM)
        dense = await embedder.embed(texts)
        sparse = await sparse_embedder.embed_sparse(texts)
        for i, text in enumerate(texts):
            await store.upsert(
                f"chunk-{i}",
                dense[i],
                {"document_id": f"doc-{i // 10}", "text": text},
            )
        return len(dense) + len(sparse)

    result = await run_benchmark(
        "embedding_throughput_mock",
        scenario,
        iterations=5,
        warmup=1,
        description=f"mock dense+sparse embed + index {N_CHUNKS} chunks into InMemoryVectorStore",
    )
    check_sane(result, expected_iterations=5)
    report_writer.add(result)


async def test_fastembed_embedding_throughput(report_writer: ReportWriter) -> None:
    """FastEmbed dense embedding of a small batch — skipped unless cached locally."""
    model_name = "BAAI/bge-small-en-v1.5"
    if not fastembed_model_cached(model_name):
        pytest.skip(
            f"fastembed model {model_name} not present in the local cache; "
            "skipping to stay offline-safe"
        )
    embedder = FastEmbedDense(model_name=model_name)
    try:
        await embedder.embed(["warmup probe"])
    except Exception as exc:  # pragma: no cover - depends on machine state
        pytest.skip(f"fastembed model could not be loaded ({exc}); skipping")

    texts = [generate_chunk_text(i) for i in range(32)]

    async def scenario() -> int:
        vectors = await embedder.embed(texts)
        return len(vectors)

    result = await run_benchmark(
        "embedding_throughput_fastembed",
        scenario,
        iterations=3,
        warmup=1,
        description=f"FastEmbedDense ({model_name}) embedding of 32 chunk texts",
    )
    check_sane(result, expected_iterations=3)
    report_writer.add(result)


async def _indexed_store() -> tuple[InMemoryVectorStore, MockEmbedder, list[str]]:
    """Index 200 mock-embedded chunks; shared by the retrieval latency scenarios."""
    texts = [generate_chunk_text(i) for i in range(N_CHUNKS)]
    embedder = MockEmbedder(dim=DIM)
    store = InMemoryVectorStore(vector_size=DIM)
    vectors = await embedder.embed(texts)
    for i, text in enumerate(texts):
        await store.upsert(
            f"chunk-{i}",
            vectors[i],
            {"document_id": f"doc-{i // 10}", "text": text},
        )
    return store, embedder, texts


async def test_retrieval_latency(report_writer: ReportWriter) -> None:
    """Dense and hybrid (dense + BM25 with RRF fusion) query latency over 200 chunks."""
    store, embedder, texts = await _indexed_store()
    config = RetrievalConfig(strategies=["dense"], top_k=10, candidate_k=50)
    dense = DenseRetriever(store, embedder, config)

    corpus = [(f"chunk-{i}", f"doc-{i // 10}", text) for i, text in enumerate(texts)]

    def corpus_provider() -> list[tuple[str, str, str]]:
        return list(corpus)

    hybrid = HybridRetriever(
        [dense, BM25Retriever(corpus_provider, config)],
        ReciprocalRankFusion(),
        config,
    )
    query = Query(text=QUERY_TEXT, top_k=10)

    dense_result = await run_benchmark(
        "retrieval_dense_latency",
        lambda: dense.retrieve(query),
        iterations=20,
        warmup=2,
        description=f"DenseRetriever query latency over {N_CHUNKS} mock-embedded chunks "
        "(InMemoryVectorStore, candidate_k=50)",
    )
    check_sane(dense_result, expected_iterations=20)
    report_writer.add(dense_result)

    hybrid_result = await run_benchmark(
        "retrieval_hybrid_latency",
        lambda: hybrid.retrieve(query),
        iterations=10,
        warmup=1,
        description=f"HybridRetriever (dense + BM25, ReciprocalRankFusion) latency over "
        f"{N_CHUNKS} chunks; cold run includes the BM25 index build",
    )
    check_sane(hybrid_result, expected_iterations=10)
    report_writer.add(hybrid_result)


async def test_rerank_latency(report_writer: ReportWriter) -> None:
    """Heuristic (Jaccard) rerank latency over 50 candidates."""
    candidates = _make_hits(50)
    pipeline = RerankPipeline(HeuristicReranker(RerankConfig(max_candidates=100)), RerankConfig())
    query = Query(text=QUERY_TEXT, top_k=10)

    async def scenario() -> object:
        return await pipeline.rerank(query, candidates, top_k=10)

    result = await run_benchmark(
        "rerank_latency",
        scenario,
        iterations=20,
        warmup=2,
        description="RerankPipeline + HeuristicReranker over 50 candidates (top_k=10)",
    )
    check_sane(result, expected_iterations=20)
    report_writer.add(result)


async def test_context_build_latency(report_writer: ReportWriter) -> None:
    """Token-budgeted context assembly latency over 50 retrieved hits."""
    hits = _make_hits(50)
    builder = ContextBuilderImpl(
        ContextTokenizer(WhitespaceCounter()), ContextConfig(token_budget=2048)
    )
    query = Query(text=QUERY_TEXT, top_k=10)

    async def scenario() -> object:
        return await builder.build(query, hits, token_budget=2048)

    result = await run_benchmark(
        "context_build_latency",
        scenario,
        iterations=20,
        warmup=2,
        description="ContextBuilderImpl (WhitespaceCounter) assembling context from "
        "50 retrieved hits under a 2048-token budget",
    )
    check_sane(result, expected_iterations=20)
    report_writer.add(result)


async def test_full_query_mock(tmp_path: Path, report_writer: ReportWriter) -> None:
    """End-to-end mock query: ingest 10 docs, then time RAG.ask() via the facade."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    paths: list[Path] = []
    for i in range(10):
        path = docs_dir / f"doc_{i:03d}.txt"
        path.write_text(generate_doc_text(i), encoding="utf-8")
        paths.append(path)

    async with RAG.from_config(RAGConfig.mock()) as rag:
        for path in paths:
            await rag.ingest(path)

        ask_query = "What does the corpus say about retrieval latency and throughput?"

        async def scenario() -> object:
            return await rag.ask(ask_query)

        result = await run_benchmark(
            "full_query_mock",
            scenario,
            iterations=5,
            warmup=1,
            description="end-to-end RAG.from_config(RAGConfig.mock()) ask() after "
            "ingesting 10 synthetic docs",
        )
        check_sane(result, expected_iterations=5)
        answer = await rag.ask(ask_query)
        assert isinstance(answer, AskResult)
        assert answer.answer
    report_writer.add(result)


async def test_generation_latency(report_writer: ReportWriter) -> None:
    """Tiny-prompt generation latency against local LM Studio; skipped if unreachable."""
    base_url = "http://localhost:1234/v1"
    async with httpx.AsyncClient(timeout=2.0) as probe:
        try:
            response = await probe.get(f"{base_url}/models")
            response.raise_for_status()
            models = response.json().get("data") or []
        except (httpx.HTTPError, ValueError):
            models = []
    if not models:
        pytest.skip("no OpenAI-compatible endpoint answering at localhost:1234/v1; skipping")

    model_id = str(models[0]["id"])
    provider = OpenAICompatibleProvider(base_url=base_url, timeout_s=30.0)
    service = GenerationService({"lm_studio": provider}, max_retries=0)
    request = GenerationRequest(
        messages=[Message(role="user", content="Reply with the single word: ok")],
        model=model_id,
        temperature=0.0,
        max_tokens=8,
    )
    try:
        result = await run_benchmark(
            "generation_latency",
            lambda: service.generate(request),
            iterations=3,
            warmup=0,
            description=f"OpenAICompatibleProvider chat completion ({model_id}), "
            "max_tokens=8, against local LM Studio",
        )
    finally:
        await provider.aclose()
    check_sane(result, expected_iterations=3)
    report_writer.add(result)
