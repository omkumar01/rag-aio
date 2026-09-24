"""Tests for the FastAPI service surface (offline, stub components)."""

from __future__ import annotations

import fakes
import pytest
from fastapi.testclient import TestClient
from rag_core.jobs import JobStatus
from rag_orchestrator.app.app import create_app
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.orchestrator import AskResult


@pytest.fixture
def client() -> TestClient:
    services = fakes.make_services()
    return TestClient(create_app(services, PipelineConfig()))


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_with_minimal_services(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["components"] == {}


def test_ready_with_healthy_and_unhealthy_store() -> None:
    for healthy, expected in [(True, True), (False, False)]:
        services = fakes.make_services()
        services.vector_store = fakes.StubVectorStore(healthy=healthy, count=5)
        services.cache = fakes.StubCache()
        app = create_app(services, PipelineConfig())
        client = TestClient(app)
        r = client.get("/ready")
        body = r.json()
        assert body["components"]["vector_store"] is healthy
        assert body["components"]["cache"] is True
        assert body["ready"] is expected


def test_metrics_endpoint() -> None:
    services = fakes.make_services()
    services.vector_store = fakes.StubVectorStore(healthy=True, count=42)
    services.cache = fakes.StubCache()
    app = create_app(services, PipelineConfig())
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "default"
    assert body["schema_version"] == "1.0"
    assert body["vector_points"] == 42
    assert body["cache"]["size"] == 0


def test_metrics_endpoint_async_count() -> None:
    """Qdrant/SQL stores expose an async count(); /metrics must await it."""
    from rag_cache.stats import InstrumentedCache

    services = fakes.make_services()
    services.vector_store = fakes.StubVectorStore(healthy=True, count=7, async_count=True)
    services.cache = InstrumentedCache(fakes.StubCache())
    app = create_app(services, PipelineConfig())
    client = TestClient(app)
    body = client.get("/metrics").json()
    assert body["vector_points"] == 7
    assert body["cache"]["hits"] == 0
    assert body["cache"]["misses"] == 0
    assert "hit_rate" in body["cache"]


def test_metrics_endpoint_no_store_no_cache(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["vector_points"] is None
    assert body["cache"] == {}


def test_version(client: TestClient) -> None:
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.0"
    assert "version" in body


def test_ask_endpoint(client: TestClient) -> None:
    r = client.post("/v1/ask", json={"query": "what is auth?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "stub answer"
    assert body["citations"]
    assert "timings_ms" in body
    assert body["metrics"]["correlation_id"]


def test_ask_endpoint_stream(client: TestClient) -> None:
    with client.stream("POST", "/v1/ask", json={"query": "q", "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "[DONE]" in body


def test_pipelines_listing(client: TestClient) -> None:
    r = client.get("/v1/pipelines")
    assert r.status_code == 200
    names = [p["config"]["name"] for p in r.json()["pipelines"]]
    assert "local_fast" in names


def test_job_not_found(client: TestClient) -> None:
    r = client.get("/v1/jobs/nope")
    assert r.status_code == 404


def test_job_recorded_after_ingest_failure(client_factory=None) -> None:
    # Query components wired, but the ingest path is not: the /v1/ingest
    # endpoint records a failed job and returns 422.
    services = fakes.make_services()  # no ingestion_pipeline wired
    app = create_app(services, PipelineConfig())
    client2 = TestClient(app, raise_server_exceptions=False)
    r = client2.post("/v1/ingest", json={"source": "whatever.txt"})
    assert r.status_code == 422
    job_id = r.json()["detail"]["job_id"]
    status = client2.get(f"/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == JobStatus.failed


def test_ask_returns_askresult_model(client: TestClient) -> None:
    r = client.post("/v1/ask", json={"query": "q", "query_id": "qid-1"})
    body = r.json()
    assert body["query_id"] == "qid-1"
    # ensure the payload validates back into the domain model
    AskResult.model_validate(body)


def test_ingest_endpoint_success(tmp_path) -> None:
    """The /v1/ingest endpoint succeeds when the full pipeline is wired."""
    from rag_cache.backends.memory import MemoryCache
    from rag_db_handler.memory_store import InMemoryVectorStore
    from rag_doc_handler import default_pipeline
    from rag_embedder.chunking.base import ChunkerConfig
    from rag_embedder.chunking.registry import create_chunker
    from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
    from rag_embedder.pipeline import EmbeddingPipeline
    from rag_embedder.tokenization import SimpleTokenizer

    source = tmp_path / "doc.md"
    source.write_text("content " * 50, encoding="utf-8")
    services = fakes.make_services()
    chunker = create_chunker(ChunkerConfig(strategy="fixed", chunk_size=64), SimpleTokenizer())
    store = InMemoryVectorStore()
    services.ingestion_pipeline = default_pipeline()
    services.embedding_pipeline = EmbeddingPipeline(
        chunker=chunker,
        embedder=MockEmbedder(dim=8),
        sparse_embedder=MockSparseEmbedder(),
        store=store,
    )
    services.vector_store = store
    services.cache = MemoryCache(max_items=128)
    app = create_app(services, PipelineConfig())
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/ingest", json={"source": str(source)})
    assert r.status_code == 200
    body = r.json()
    assert body["content_hash"]
    assert body["pages"]
    # the job is recorded as completed
    jobs = list(app.state.jobs.values())
    assert jobs and jobs[0].status == JobStatus.completed
