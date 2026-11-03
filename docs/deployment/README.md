# Deployment

rag-aio ships as a uv workspace of 18 packages that can be composed in one Python
process or split into services. Three topologies cover everything from a laptop demo to a
scaled server deployment (see [final-report.md](../development/final-report.md) §7 and
`architecture.md` for the reasoning behind each choice).

## Topology 1 — In-process embedded (default)

The whole platform runs inside one Python process with zero infrastructure: no Docker, no
external services, no network beyond the first model download.

Component choices (the `RAGConfig` defaults, per ADR-0003 and ADR-0006):

| Concern | Component | Where it lives |
| --- | --- | --- |
| Vector store | `QdrantVectorStore` in **Qdrant local mode** | `./data/qdrant` |
| Document/KV store | `SQLDocumentStore` / `SQLKeyValueStore` | `sqlite+aiosqlite:///./data/rag.db` |
| Cache | memory backend (sqlite also available) | in-process |
| Embeddings | `FastEmbedDense` / `FastEmbedSparse` (ONNX, default dense model `BAAI/bge-small-en-v1.5`) | in-process |
| Rerank | `HeuristicReranker` (lexical) or `CrossEncoderReranker` | in-process |
| Generation | `OpenAICompatibleProvider` — bring your own server, or `RAGConfig.mock()` for fully offline | in-process client |

Run the full local stack end to end (parse → chunk → FastEmbed dense+sparse → Qdrant local
mode → hybrid RRF → rerank → context → LM Studio generation):

```bash
uv run python examples/fully_local.py
```

For a zero-dependency offline run (deterministic hash-based `MockEmbedder`, in-memory
vector store, heuristic reranker, ~25 ms per full query) use the mock profile:

```bash
uv run python examples/quickstart.py
```

## Topology 2 — Workstation with LM Studio

The same in-process platform as topology 1, with models served by a localhost model
server — the default profile per ADR-0004.

Component choices:

- Vector store / db / cache: identical to topology 1 (Qdrant local mode + sqlite).
- Model server: **LM Studio** (or vLLM, Ollama, llama.cpp — anything OpenAI-compatible)
  for generation, and optionally for reranking via `RemoteReranker`, which auto-detects
  LM Studio's `/rerank` endpoint and falls back to chat-based scoring.

### LM Studio setup

1. Install and start LM Studio; load a chat model.
2. Start the local server on the default port: `http://localhost:1234/v1` (this is the
   `generation.base_url` default in `RAGConfig`; override with `LM_STUDIO_URL` or
   `OPENAI_BASE_URL` depending on the entry point).
3. A loaded chat model is auto-detected from the `/models` endpoint; pin one explicitly
   with `LM_STUDIO_MODEL` (or set `generation.model` in the TOML config).
4. First run of the FastEmbed embedding model downloads ~30 MB unless already cached.

Auto-skip behaviour: everything that needs the model server probes it first and degrades
cleanly when it is unreachable — `examples/fully_local.py` exits with a clear message, the
generation benchmark and the Tier 2 e2e tests `pytest.skip`, so no run ever hangs on a
missing server.

### Run the service + console

Start the FastAPI service (defaults to `127.0.0.1:8000`; pass `--config` for a TOML file
instead of the mock default):

```bash
uv run rag-aio serve                 # ingest/ask/serve/config are the other CLI commands
```

In a second terminal, start the Streamlit management console (Config / Dashboard / Ask
tabs; it talks to the backend at `API_URL`, default `http://localhost:8000`):

```bash
uv run streamlit run packages/rag-ui/src/rag_ui/app.py   # opens http://localhost:8501
```

The console shares the same Pydantic `RAGConfig` models as the backend, so the config
editor validates exactly what the service accepts; secrets stay masked
(`api_key_ref` stores an environment-variable name, never a credential value).

## Topology 3 — Scaled services

When one machine is no longer enough, shared state moves to servers and selected modules
become independent FastAPI services behind the orchestrator (ADR-0005).

Component choices:

| Concern | Component | Notes |
| --- | --- | --- |
| Vector store | **Qdrant server** via `docker-compose.qdrant.yml` | exposed on `localhost:6333` (HTTP) and `6334` (gRPC), named volume for storage |
| Metadata/document store | **PostgreSQL** | point `storage.db_url` at an `asyncpg` URL instead of sqlite |
| Cache | **Redis** | the `rag-cache` redis extra; replaces the memory/sqlite backends |
| Model serving | LM Studio / vLLM behind a shared OpenAI-compatible endpoint | one endpoint, many workers |
| Services | doc-handler, ocr, embedder, retrieval, rerank, generation as FastAPI services | each exposes `/health`, `/ready`, `/metrics`, `/version` |

Start the Qdrant server and point the config at it:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

```toml
[rag.db_handler.vector_store]
mode = "server"
url = "http://localhost:6333"
```

Individual modules are launched as services with the same app factory the orchestrator
uses (`create_app(services, pipeline_config)` in `rag-orchestrator`); long operations
(mass ingestion, OCR of large files, reindexing, eval runs) run as jobs with IDs you can
poll via `/v1/jobs/{job_id}` (ADR-0007). Bulk ingestion itself is handled by
`rag-mass-inject` (checkpoints, backpressure, dead-letter handling, resume).

Note: the scaled topology is supported by design (contracts are network-transparent) but
is not exercised by the benchmark suite, which measures the single-node profiles.

## Authentication & multi-tenancy

The FastAPI service surface is identity-provider agnostic. `create_app`
(`packages/rag-orchestrator/src/rag_orchestrator/app/app.py`) accepts two optional
hooks, defined in `packages/rag-orchestrator/src/rag_orchestrator/app/hooks.py`, that a
host process supplies:

| Piece | Shape | Role |
| --- | --- | --- |
| `AuthContext` | `RagBaseModel` | Identity of an authenticated request: `subject` (required), `tenant`, `namespace`, free-form `attributes` |
| `authenticator` | `RequestAuthenticator` protocol | `async authenticate(request) -> AuthContext \| None`; returning `None` rejects the request |
| `namespace_resolver` | `NamespaceResolver` protocol | `namespace_for(auth) -> str \| None`; maps an identity to the namespace its data lives in |

```python
from rag_orchestrator.app import create_app
from rag_orchestrator.app.hooks import AuthContext, NamespaceResolver


class BearerAuthenticator:
    async def authenticate(self, request) -> AuthContext | None:
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        return decode_token(token)  # your IdP / JWT check -> AuthContext | None


class TenantNamespaceResolver(NamespaceResolver):
    def namespace_for(self, auth: AuthContext) -> str | None:
        return f"user_{auth.subject}"


app = create_app(
    services,
    pipeline_config,
    authenticator=BearerAuthenticator(),
    namespace_resolver=TenantNamespaceResolver(),
)
```

Rules as implemented:

- **Protected endpoints**: with an `authenticator` wired, `POST /v1/ask`,
  `POST /v1/ingest`, and `GET /v1/jobs/{job_id}` require authentication; a rejected
  request gets `401 {"error": "unauthenticated"}`. `GET /health`, `/ready`, `/metrics`,
  `/version`, and `/v1/pipelines` stay open so probes work without credentials.
- **Resolver wins over the request body** (tenant isolation is not overridable): when
  both hooks are set and the resolver returns a non-`None` namespace, any
  client-supplied `namespace` in the request body is ignored.
- **Per-request namespace fields**: `POST /v1/ask` accepts `namespace` and threads it
  through `Orchestrator.ask` / `ask_stream` / `stream_events` / `prepare` into
  retrieval; `POST /v1/ingest` accepts `namespace`, which is stamped onto the resulting
  `Document` (`Document.tenant` / `Document.namespace`). Without a resolver the body
  namespace is used as-is; without either, all data lives in one unnamespaced pool.

### Recipe: `user_{id}` + `global` namespace merging

A common pattern is per-user data plus a shared corpus everyone can read. Keep the
resolver mapping identities to `user_{subject}` and wrap the vector store so *reads*
search both the caller's namespace and `global`, while *writes* stay in the resolved
namespace. Every `VectorStore` method accepts a `namespace` argument, so the wrapper is
mechanical:

```python
class SharedPlusUserVectorStore:
    """Read the resolved namespace plus `global`; write to the resolved namespace."""

    def __init__(self, inner, shared_namespace: str = "global") -> None:
        self._inner = inner
        self._shared = shared_namespace

    async def search(self, vector, top_k, filters=None, namespace=None):
        hits = list(await self._inner.search(vector, top_k, filters, namespace=namespace))
        hits += await self._inner.search(vector, top_k, filters, namespace=self._shared)
        return sorted(hits, key=lambda h: h.normalized_score, reverse=True)[:top_k]

    async def upsert(self, chunk_id, vector, payload, namespace=None):
        await self._inner.upsert(chunk_id, vector, payload, namespace=namespace)

    def __getattr__(self, name):  # delete / count / health / close / ... delegate verbatim
        return getattr(self._inner, name)


services.vector_store = SharedPlusUserVectorStore(services.vector_store)
```

The merge above is a plain sort by normalized score — for production quality, re-rank
the merged candidate set (the pipeline's reranker or a cross-encoder) instead.

### Alternative: FastAPI middleware

Teams with existing auth middleware can keep it and adapt it to the hooks: the
middleware verifies the protected paths and stashes the `AuthContext` on
`request.state`; a thin adapter authenticator reads it back, which keeps the
resolver-wins namespace logic fully active:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

PROTECTED = ("/v1/ask", "/v1/ingest", "/v1/jobs")


class StateAuthenticator:
    """Adapter: authentication already happened in middleware."""

    async def authenticate(self, request: Request) -> AuthContext | None:
        return getattr(request.state, "auth", None)


def build_app(services, pipeline_config):
    app = create_app(
        services,
        pipeline_config,
        authenticator=StateAuthenticator(),
        namespace_resolver=TenantNamespaceResolver(),
    )

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        if request.url.path.startswith(PROTECTED):
            auth = await verify_bearer(request)  # your IdP / JWT verification
            if auth is None:
                return JSONResponse({"error": "unauthenticated"}, status_code=401)
            request.state.auth = auth
        return await call_next(request)

    return app
```

## Resource notes

- **Mock profile**: no model downloads, no disk writes outside tmp dirs; a full
  ingest+ask query runs in ~25 ms — suitable for laptops and CI.
- **Local default profile**: the ONNX embedding model needs a one-time ~30 MB download;
  Qdrant local mode and sqlite persist under `./data/`. `RAGConfig.mock()` needs none of
  this.
- **Heavy imports are lazy**: `fastembed`, `qdrant_client`, Streamlit, and uvicorn are
  imported only when the feature using them is actually loaded, so CLI startup and test
  collection stay fast.
- **RAM reporting caveat**: benchmark RAM figures via `tracemalloc` cover Python
  allocations only; native buffers (ONNX runtime, model weights, Qdrant storage) are
  invisible — see [Sizing on process RSS](#sizing-on-process-rss) below.

## Shutdown ordering

Close the wiring bag before the process exits:

```python
try:
    ...  # serve / run jobs / answer queries
finally:
    await services.aclose()  # idempotent — safe to call more than once
```

`OrchestratorServices.aclose()` closes the vector store, document store, and cache
first, then any other owned component that exposes `aclose`/`close` (via
`rag_core.lifecycle.aclose_component` / `rag_core.Closable`); components without a
close method are skipped silently. Calling it twice is a no-op.

Windows + local-mode Qdrant: Qdrant local mode holds file locks on its persistence
directory, so `services.aclose()` must run *before* any temp-dir cleanup — the pattern
in [examples/fully_local.py](../../examples/fully_local.py) calls it in a `finally`
block inside the `TemporaryDirectory` context. Relatedly, `qdrant_path` is
caller-managed: it is a persistent directory the library neither creates as a temp dir
nor deletes; reclaim the disk yourself when the path is no longer needed.

## Sizing on process RSS

Benchmark RAM numbers are produced with `tracemalloc`, which sees Python allocations
only. Native buffers — the ONNX runtime and its model weights, Qdrant local-mode
storage, C-extension arenas — are invisible to it, so treat the benchmark RAM figures
as a *lower bound* on real footprint and size production machines on measured process
RSS.

Measuring real RSS:

| Platform | Command |
| --- | --- |
| Linux | `grep VmRSS /proc/<pid>/status` (peak: `VmHWM`), or `ps -o rss= -p <pid>` (KiB) |
| macOS | `ps -o rss= -p <pid>` (KiB) |
| Windows | Task Manager → Details, or `psutil.Process(pid).memory_info().rss` |
| Containers | `docker stats` (add `--no-stream` for a single sample) |

Recipe:

1. Deploy with the production model weights (real generation/embedding models, not the
   mock profile) and a representative corpus — Qdrant local mode mmaps its index, so
   corpus size is part of the footprint.
2. Load-test at expected concurrency and record peak RSS (Linux `VmHWM` is the
   high-water mark; sample `docker stats` continuously for containers).
3. Size the host or container limit at ~1.5–2x peak RSS to absorb spikes (large PDF
   parsing, rerank candidate batches, ONNX arena growth).

## Health and monitoring

Every deployable module (doc-handler, ocr, embedder, retrieval, rerank, generation and the
orchestrator) exposes an identical operational API (ADR-0005):

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness probe |
| `GET /ready` | readiness probe with per-component detail |
| `GET /metrics` | pipeline metrics (vector points, cache hit rate) — rendered by the Streamlit Dashboard tab |
| `GET /version` | build/version identification |

The orchestrator's pipeline API additionally serves:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/ask` | query endpoint (JSON; SSE streaming with `stream=true`) |
| `POST /v1/ingest` | single-file ingestion |
| `GET /v1/pipelines` | registered pipeline definitions |
| `GET /v1/jobs/{job_id}` | long-running job status |

`rag-observe` emits OpenTelemetry spans/metrics with correlation IDs and log redaction,
exporting to a pluggable OTel backend for distributed tracing across services.

### Server-sent events and job callbacks

`POST /v1/ask` with `stream=true` has two wire formats, selected by the `events` field
on the request body:

- **Default (`events=false`)** — unchanged: `data: {...}` JSON text deltas terminated by
  a `data: [DONE]` sentinel.
- **Typed SSE (`events=true`)** — named frames instead: one `event: delta` per
  generation fragment carrying `{"type": "delta", "text": ...}`, then exactly one
  `event: done` carrying the terminal payload with nothing discarded — the fully
  accumulated answer, citation provenance, per-stage timings, and the query id. There
  is no `[DONE]` sentinel to parse; the client switches on the event name.

```text
event: delta
data: {"type":"delta","text":"The activation code"}

event: done
data: {"type":"final","answer":"The activation code is RAG-8891.","citations":[...],"timings_ms":{"query":0.4,"retrieve":12.1},"query_id":"..."}
```

For long-running work, `create_app(..., on_job_complete=callback)` fires `callback`
after every terminal ingest job (success or failure), passing the `PipelineJob` as its
only argument. The callback may be sync or async; exceptions are logged and swallowed
so job recording never breaks the endpoint. Use it for webhook-style side effects —
enqueueing a UI refresh, pushing metrics, or triggering downstream indexing. Job
status remains pollable via `GET /v1/jobs/{job_id}`.
