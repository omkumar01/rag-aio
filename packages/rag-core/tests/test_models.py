"""Tests for canonical domain models: documents, chunks, embeddings, retrieval,
reranking, context, generation, jobs, provider info, evaluation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_core.chunks import Chunk, ChunkMetadata
from rag_core.context import Citation, Context, ContextItem
from rag_core.documents import (
    BoundingBox,
    Document,
    DocumentAsset,
    DocumentMetadata,
    DocumentPage,
    PageBlock,
)
from rag_core.embeddings import Embedding, SparseEmbedding
from rag_core.evaluation import MetricResult
from rag_core.generation import GenerationRequest, GenerationResult, Message, Usage
from rag_core.jobs import JobStatus, PipelineJob
from rag_core.models_info import ModelInfo, ProviderInfo
from rag_core.queries import Query, QueryVariant
from rag_core.rerank import RerankHit
from rag_core.retrieval import RetrievalHit, RetrievalResult
from rag_core.serde import from_json, to_json


class TestDocuments:
    def test_document_roundtrip(self) -> None:
        doc = Document(
            source_uri="file:///tmp/report.pdf",
            metadata=DocumentMetadata(title="Report", mime_type="application/pdf"),
            pages=[DocumentPage(page_number=1, text="hello world")],
            text="hello world",
        )
        assert doc.id
        assert doc.content_hash
        revived = from_json(to_json(doc), Document)
        assert revived == doc

    def test_document_hash_covers_content(self) -> None:
        d1 = Document(source_uri="s3://a", text="one")
        d2 = Document(source_uri="s3://a", text="two")
        assert d1.content_hash != d2.content_hash

    def test_document_auto_ids(self) -> None:
        page = DocumentPage(page_number=1, text="t")
        assert page.id
        block = PageBlock(page_id=page.id, kind="text", text="t")
        assert block.id
        asset = DocumentAsset(asset_id="a1", kind="image", page_number=1)
        assert asset.id

    def test_bounding_box_ordering_enforced(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x0=10, y0=0, x1=5, y1=10)

    def test_tenant_namespace(self) -> None:
        doc = Document(source_uri="x", tenant="acme", namespace="acme-prod")
        assert doc.tenant == "acme"
        assert doc.namespace == "acme-prod"


class TestChunks:
    def test_chunk_deterministic_id(self) -> None:
        meta = ChunkMetadata(
            document_id="d1", document_hash="h", chunker="token", chunker_version="1"
        )
        c1 = Chunk(document_id="d1", text="same text", index=0, metadata=meta)
        c2 = Chunk(document_id="d1", text="same text", index=0, metadata=meta)
        assert c1.id == c2.id
        c3 = Chunk(document_id="d1", text="other", index=1, metadata=meta)
        assert c3.id != c1.id

    def test_parent_child_link(self) -> None:
        meta = ChunkMetadata(
            document_id="d1",
            document_hash="h",
            chunker="token",
            chunker_version="1",
            parent_chunk_id="p1",
        )
        child = Chunk(document_id="d1", text="child", index=0, metadata=meta)
        assert child.metadata.parent_chunk_id == "p1"

    def test_chunk_roundtrip(self) -> None:
        meta = ChunkMetadata(
            document_id="d1", document_hash="h", chunker="token", chunker_version="1"
        )
        chunk = Chunk(document_id="d1", text="t", index=0, metadata=meta)
        assert from_json(to_json(chunk), Chunk) == chunk


class TestEmbeddings:
    def test_dense_embedding(self) -> None:
        e = Embedding(chunk_id="c1", vector=[0.1, 0.2], model="fastembed", dimension=2)
        assert e.dimension == 2
        assert from_json(to_json(e), Embedding) == e

    def test_sparse_embedding_validates_alignment(self) -> None:
        with pytest.raises(ValidationError):
            SparseEmbedding(chunk_id="c", indices=[1, 2], values=[0.5], model="bm25")


class TestQueries:
    def test_query_defaults(self) -> None:
        q = Query(text="what is auth?")
        assert q.id
        assert q.top_k == 10
        assert q.filters == {}

    def test_variant_kinds(self) -> None:
        v = QueryVariant(query_id="q1", text="rewritten", kind="rewrite")
        assert v.kind == "rewrite"


class TestRetrieval:
    def test_hit_explainability(self) -> None:
        hit = RetrievalHit(
            chunk_id="c1",
            document_id="d1",
            score=0.9,
            normalized_score=0.9,
            rank=0,
            strategy="dense",
            model="fastembed",
        )
        assert hit.strategy_scores == {}

    def test_result_roundtrip(self) -> None:
        hit = RetrievalHit(
            chunk_id="c1",
            document_id="d1",
            score=0.5,
            normalized_score=0.5,
            rank=0,
            strategy="dense",
        )
        result = RetrievalResult(query_id="q1", hits=[hit], strategies=["dense"])
        assert from_json(to_json(result), RetrievalResult) == result


class TestRerank:
    def test_rerank_hit_preserves_original(self) -> None:
        hit = RerankHit(
            chunk_id="c1",
            document_id="d1",
            score=0.95,
            rank=0,
            original_score=0.5,
            original_rank=2,
        )
        assert hit.original_rank == 2


class TestContext:
    def test_context_budget_tracking(self) -> None:
        item = ContextItem(
            chunk_id="c1", document_id="d1", text="hello", token_count=2, citation_id="[1]"
        )
        ctx = Context(items=[item], token_budget=100, strategy="relevance_first")
        assert ctx.total_tokens == 2

    def test_citation_roundtrip(self) -> None:
        c = Citation(
            citation_id="[1]",
            document_id="d1",
            chunk_id="c1",
            source_uri="file:///a.pdf",
            page_numbers=[1, 2],
        )
        assert from_json(to_json(c), Citation) == c


class TestGeneration:
    def test_request_defaults(self) -> None:
        req = GenerationRequest(messages=[Message(role="user", content="hi")])
        assert req.temperature == 0.2
        assert req.stream is False
        assert req.model is None  # resolved by routing later

    def test_result_usage(self) -> None:
        res = GenerationResult(
            text="answer",
            model="qwen3.8-27b",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        assert res.usage.total_tokens == 15

    def test_usage_totals(self) -> None:
        assert Usage(prompt_tokens=1, completion_tokens=2).total_tokens == 3


class TestJobs:
    def test_job_lifecycle_fields(self) -> None:
        job = PipelineJob(job_id="j1", kind="ingest")
        assert job.status == JobStatus.queued
        assert job.progress == 0.0
        running = job.model_copy(update={"status": JobStatus.running, "progress": 0.5})
        assert running.status == JobStatus.running

    def test_progress_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PipelineJob(job_id="j", kind="ingest", progress=1.5)


class TestProviderInfo:
    def test_model_info(self) -> None:
        m = ModelInfo(
            model_id="qwen3.8-27b",
            provider="lmstudio",
            capabilities=["generate", "stream"],
            context_window=32768,
        )
        assert m.embedding_dim is None

    def test_provider_secret_never_stored(self) -> None:
        p = ProviderInfo(
            name="lmstudio",
            kind="openai_compatible",
            base_url="http://localhost:1234/v1",
            auth_ref="LMSTUDIO_API_KEY",
        )
        dumped = p.model_dump()
        assert "api_key" not in dumped
        assert dumped["auth_ref"] == "LMSTUDIO_API_KEY"


class TestEvaluation:
    def test_metric_result(self) -> None:
        m = MetricResult(name="recall@10", value=0.75, k=10)
        assert m.k == 10


class TestSerde:
    def test_to_json_compact(self) -> None:
        doc = Document(source_uri="x", text="t")
        assert "\n" not in to_json(doc)

    def test_from_json_type_safety(self) -> None:
        with pytest.raises(ValidationError):
            from_json('{"text": 1}', Document)
