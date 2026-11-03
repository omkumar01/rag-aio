"""Tests for the FastAPI service surface (offline, stub components)."""

from __future__ import annotations

import json
from typing import Any

import fakes
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from rag_core.jobs import JobStatus, PipelineJob
from rag_orchestrator.app.app import create_app
from rag_orchestrator.app.hooks import AuthContext
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.orchestrator import AskResult


class RecordingAuthenticator:
    """Returns a fixed AuthContext (or None to reject)."""

    def __init__(self, context: AuthContext | None) -> None:
        self.context = context
        self.calls = 0

    async def authenticate(self, request: Request) -> AuthContext | None:
        self.calls += 1
        return self.context


class RecordingNamespaceResolver:
    """Records the AuthContext it sees; returns a fixed namespace."""

    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace
        self.seen: list[AuthContext] = []

    def namespace_for(self, auth: AuthContext) -> str | None:
        self.seen.append(auth)
        return self.namespace


class NamespaceRecordingRetriever(fakes.StubRetriever):
    """StubRetriever that records the namespace of each incoming query."""

    def __init__(self, hits: list[fakes.RetrievalHit]) -> None:
        super().__init__(hits)
        self.namespaces: list[str | None] = []

    async def retrieve(self, query: Any) -> Any:
        self.namespaces.append(query.namespace)
        return await super().retrieve(query)


def make_client_with_recording_ns() -> tuple[TestClient, NamespaceRecordingRetriever]:
    services = fakes.make_services()
    retriever = NamespaceRecordingRetriever(fakes.make_hits())
    services.retrievers = [retriever]  # type: ignore[list-item]
    return TestClient(create_app(services, PipelineConfig())), retriever


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


# --- auth hooks (#1) -----------------------------------------------------------


def test_auth_rejects_unauthenticated_request() -> None:
    services = fakes.make_services()
    authenticator = RecordingAuthenticator(context=None)
    app = create_app(services, PipelineConfig(), authenticator=authenticator)
    client = TestClient(app)
    r = client.post("/v1/ask", json={"query": "q"})
    assert r.status_code == 401
    assert r.json() == {"error": "unauthenticated"}


def test_auth_protects_ingest_and_job_endpoints() -> None:
    services = fakes.make_services()
    authenticator = RecordingAuthenticator(context=None)
    app = create_app(services, PipelineConfig(), authenticator=authenticator)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/v1/ingest", json={"source": "x.txt"}).status_code == 401
    assert client.get("/v1/jobs/whatever").status_code == 401


def test_auth_passes_through_when_authenticated() -> None:
    services = fakes.make_services()
    authenticator = RecordingAuthenticator(context=AuthContext(subject="user-1"))
    app = create_app(services, PipelineConfig(), authenticator=authenticator)
    client = TestClient(app)
    r = client.post("/v1/ask", json={"query": "what is auth?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub answer"
    assert authenticator.calls == 1


def test_health_ready_metrics_pipelines_unprotected() -> None:
    services = fakes.make_services()
    # The authenticator rejects everything: these endpoints must not care.
    authenticator = RecordingAuthenticator(context=None)
    app = create_app(services, PipelineConfig(), authenticator=authenticator)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/v1/pipelines").status_code == 200
    assert client.get("/version").status_code == 200
    assert authenticator.calls == 0


# --- namespace resolution (#11 app side) ---------------------------------------


def test_namespace_resolver_wins_over_body_namespace() -> None:
    services = fakes.make_services()
    retriever = NamespaceRecordingRetriever(fakes.make_hits())
    services.retrievers = [retriever]  # type: ignore[list-item]
    authenticator = RecordingAuthenticator(AuthContext(subject="u", tenant="acme"))
    resolver = RecordingNamespaceResolver(namespace="acme-ns")
    app = create_app(
        services,
        PipelineConfig(),
        authenticator=authenticator,
        namespace_resolver=resolver,
    )
    client = TestClient(app)
    r = client.post("/v1/ask", json={"query": "q", "namespace": "body-ns"})
    assert r.status_code == 200
    assert retriever.namespaces == ["acme-ns"]
    assert resolver.seen and resolver.seen[0].tenant == "acme"


def test_body_namespace_used_without_hooks() -> None:
    client, retriever = make_client_with_recording_ns()
    r = client.post("/v1/ask", json={"query": "q", "namespace": "body-ns"})
    assert r.status_code == 200
    assert retriever.namespaces == ["body-ns"]


def test_body_namespace_used_when_resolver_returns_none() -> None:
    services = fakes.make_services()
    retriever = NamespaceRecordingRetriever(fakes.make_hits())
    services.retrievers = [retriever]  # type: ignore[list-item]
    authenticator = RecordingAuthenticator(AuthContext(subject="u"))
    resolver = RecordingNamespaceResolver(namespace=None)
    app = create_app(
        services,
        PipelineConfig(),
        authenticator=authenticator,
        namespace_resolver=resolver,
    )
    client = TestClient(app)
    r = client.post("/v1/ask", json={"query": "q", "namespace": "body-ns"})
    assert r.status_code == 200
    assert retriever.namespaces == ["body-ns"]


def test_no_namespace_defaults_to_none(client: TestClient) -> None:
    services = fakes.make_services()
    retriever = NamespaceRecordingRetriever(fakes.make_hits())
    services.retrievers = [retriever]  # type: ignore[list-item]
    app = create_app(services, PipelineConfig())
    plain = TestClient(app)
    r = plain.post("/v1/ask", json={"query": "q"})
    assert r.status_code == 200
    assert retriever.namespaces == [None]


def test_ingest_namespace_set_on_document(tmp_path) -> None:
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
    r = client.post("/v1/ingest", json={"source": str(source), "namespace": "tenant-x"})
    assert r.status_code == 200
    assert r.json()["namespace"] == "tenant-x"


# --- typed SSE (#12) -----------------------------------------------------------


def test_stream_events_typed_sse(client: TestClient) -> None:
    with client.stream("POST", "/v1/ask", json={"query": "q", "stream": True, "events": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "event: delta" in body
    assert body.count("event: done") == 1
    assert "[DONE]" not in body
    # the done frame carries the final payload with the contract keys
    done_frames = [
        line[6:] for line in body.splitlines() if line.startswith("data: ") and '"final"' in line
    ]
    assert len(done_frames) == 1
    payload = json.loads(done_frames[0])
    assert payload["type"] == "final"
    assert "stub" in payload["answer"]
    for key in ("citations", "timings_ms", "query_id"):
        assert key in payload


def test_stream_default_regression_frames(client: TestClient) -> None:
    """events=False (default) keeps the exact legacy wire format."""
    with client.stream("POST", "/v1/ask", json={"query": "q", "stream": True}) as r:
        body = "".join(r.iter_text())
    assert body.endswith("data: [DONE]\n\n")
    assert "event: delta" not in body and "event: done" not in body
    # each delta frame is `data: <json-encoded-string>` exactly as before
    frames = [f for f in body.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    for frame in frames[:-1]:
        assert frame.startswith("data: ")
        raw = json.loads(frame[len("data: ") :])
        assert isinstance(raw, str)  # json.dumps of the raw str delta


# --- job completion callback (#14) ---------------------------------------------


def test_on_job_complete_invoked_after_successful_ingest(tmp_path) -> None:
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
    seen: list[PipelineJob] = []

    async def on_job_complete(job: PipelineJob) -> None:
        seen.append(job)

    app = create_app(services, PipelineConfig(), on_job_complete=on_job_complete)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/ingest", json={"source": str(source)})
    assert r.status_code == 200
    assert len(seen) == 1
    assert isinstance(seen[0], PipelineJob)
    assert seen[0].status == JobStatus.completed


def test_on_job_complete_error_does_not_fail_request(tmp_path) -> None:
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

    def broken_callback(job: PipelineJob) -> None:
        msg = "callback blew up"
        raise RuntimeError(msg)

    app = create_app(services, PipelineConfig(), on_job_complete=broken_callback)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/ingest", json={"source": str(source)})
    assert r.status_code == 200


def test_on_job_complete_invoked_after_failed_ingest() -> None:
    services = fakes.make_services()  # no ingestion_pipeline wired -> 422 path
    seen: list[PipelineJob] = []

    async def on_job_complete(job: PipelineJob) -> None:
        seen.append(job)

    app = create_app(services, PipelineConfig(), on_job_complete=on_job_complete)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/ingest", json={"source": "whatever.txt"})
    assert r.status_code == 422
    assert len(seen) == 1
    assert seen[0].status == JobStatus.failed


def test_sync_on_job_complete_invoked(tmp_path) -> None:
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
    seen: list[PipelineJob] = []
    app = create_app(services, PipelineConfig(), on_job_complete=seen.append)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/v1/ingest", json={"source": str(source)}).status_code == 200
    assert len(seen) == 1
