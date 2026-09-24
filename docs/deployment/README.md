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
  invisible — size production machines on measured process RSS, not benchmark numbers.

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
