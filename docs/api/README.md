# APIs

rag-aio exposes the same operations through three surfaces: a Python facade
(`packages/rag-aio`), a Typer CLI (`rag-aio`), and a FastAPI service application
(`rag_orchestrator.app`). Every service surface publishes an OpenAPI schema at
`/docs`. See [ADR-0005](../../adr/ADR-0005-fastapi-service-apis.md) for why HTTP/JSON +
SSE is the only wire protocol.

## Python API

### The `RAG` facade

`RAG` (in `rag_aio.facade`, re-exported from `rag_aio`) is the single entry point for most
users. Heavy backends are imported lazily, so `import rag_aio` stays dependency-light.

```python
import asyncio

from rag_aio import RAG, RAGConfig


async def main() -> None:
    async with RAG.from_config(RAGConfig.mock()) as rag:  # fully offline
        await rag.ingest("documents/meeting-notes.md")
        result = await rag.ask("Who led the quarterly review?")
        print(result.answer)
        for c in result.citations:
            print(f"[{c.citation_id}] {c.source_uri or c.document_id}")


asyncio.run(main())
```

`from_config` accepts a `RAGConfig`, or a `str`/`Path` to a TOML file (loaded via
`RAGConfig.from_file`), so `RAG.from_config("config.toml")` works as written.

Streaming — with `stream=True`, `ask` returns an async iterator of text deltas instead of
an `AskResult`:

```python
async with RAG.from_config("config.toml") as rag:
    deltas = await rag.ask("Summarize the document", stream=True)
    async for delta in deltas:
        print(delta, end="", flush=True)
```

The facade surface:

| Member | Signature | Notes |
| --- | --- | --- |
| `RAG.from_config` | `(config: RAGConfig \| str \| Path) -> RAG` | Builds services from config; the primary constructor. |
| `ingest` | `async (path: str \| Path) -> Document` | One file through the stage-cached ingest pipeline. |
| `ingest_directory` | `async (directory, recursive=False) -> list[Document]` | All supported files under a directory. |
| `ask` | `async (query, *, stream=False, overrides=None) -> AskResult \| AsyncIterator[str]` | `overrides` are folded into the resolved `PipelineConfig` (see [Configuration](../configuration/README.md)). |
| `app` | `FastAPI` (property) | A FastAPI service backed by this facade's services. |
| `__aenter__` / `__aexit__` | async context manager | Best-effort `close()` on the vector and document stores on exit. |

`build_services(config: RAGConfig) -> OrchestratorServices` is also public: it selects the
mock wiring when `config.embedder.backend == "mock"`, otherwise delegates to
`rag_orchestrator.load_local_services(...)`.

### `AskResult`

Returned by non-streaming `ask` (defined in `rag_orchestrator.orchestrator`):

| Field | Type | Meaning |
| --- | --- | --- |
| `answer` | `str` | The generated answer text. |
| `citations` | `list[Citation]` | Source citations derived from the assembled context. |
| `context` | `Context \| None` | The token-budgeted context (inspection/debugging). |
| `timings_ms` | `dict[str, float]` | Per-stage wall-clock timings: `query`, `retrieve`, `rerank`, `context`, `generate`, plus `merge_ms` and `total_ms`. |
| `query_id` | `str` | Identifier for the query (stable within one ask). |
| `metrics` | `dict[str, Any]` | Token usage from generation plus `cached` and `correlation_id`. |

### Per-stage entry points

The facade composes contract-bound components; each is directly usable in-process:

- `rag_query.engine.QueryEngine` — query normalization, rewriting, expansion, and routing.
  See [rag-query](../../packages/rag-query/README.md).
- `rag_retrieval` retrievers — `dense.DenseRetriever`, `sparse.BM25Retriever`,
  `hybrid.HybridRetriever`, and `fusion.ReciprocalRankFusion`. See
  [rag-retrieval](../../packages/rag-retrieval/README.md).
- `rag_rerank.pipeline.RerankPipeline` — thin orchestrator over a `Reranker`; applies
  thresholding, normalization, and dedup. See
  [rag-rerank](../../packages/rag-rerank/README.md).
- `rag_context.builder.ContextBuilderImpl` — token-budgeted context assembly with numeric
  citations. See [rag-context](../../packages/rag-context/README.md).
- `rag_generation.generation.GenerationService` — provider routing with retry and fallback.
  See [rag-generation](../../packages/rag-generation/README.md).

All cross-module contracts are runtime-checkable `Protocol`s in `rag_core.protocols`
(`Embedder`, `VectorStore`, `Retriever`, `Reranker`, `ContextBuilder`, `Generator`, …).

## CLI

The `rag-aio` console script (`rag_aio.cli:app`, Typer) mirrors the facade. Without
`--config` every command uses `RAGConfig.mock()` — zero configuration, fully offline.

```bash
rag-aio ingest PATH [-c/--config CONFIG]              # ingest one document file
rag-aio ask QUERY [-c/--config CONFIG] [-s/--stream]  # ask through the pipeline
rag-aio config [-c/--config CONFIG]                   # print resolved RAGConfig as JSON
rag-aio serve [-h/--host HOST] [-p/--port PORT] [-c/--config CONFIG]  # start FastAPI
```

Examples:

```bash
rag-aio ask "What is machine learning?"            # mock backends, deterministic answer
rag-aio ask "…" --config config.toml --stream      # live providers, streamed to stdout
rag-aio config --config config.toml                # inspect the merged, validated config
rag-aio serve --host 0.0.0.0 --port 8000           # uvicorn under the hood
```

## HTTP API

`rag_orchestrator.app.app.create_app(services, pipeline_config)` builds the FastAPI
application. Reach it via `RAG.app`, `rag_aio.app.get_app("config.toml")`, the module-level
`rag_aio.app:app` (`uvicorn rag_aio.app:app`), or `rag-aio serve`. Interactive OpenAPI
documentation is served at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness probe; returns `{"status": "ok"}`. |
| GET | `/ready` | Readiness: per-component health (`vector_store`, `document_store`, `cache`) and an aggregate `ready` flag. |
| GET | `/metrics` | Pipeline name, `schema_version`, vector point count, cache statistics. |
| GET | `/version` | Package `version`, `schema_version`, pipeline name. |
| GET | `/v1/pipelines` | Lists the bundled pipeline profiles from the registry (e.g. `local_fast`). |
| POST | `/v1/ask` | Run the RAG pipeline. Body: `AskRequest` (`query`, `stream=False`, `overrides`, `query_id`, `correlation_id`). Returns the `AskResult` JSON. |
| POST | `/v1/ingest` | Ingest one document. Body: `IngestRequest` (`source`, `recursive=False`). Returns the ingested `Document` JSON; failures return HTTP 422 with `detail.job_id`. |
| GET | `/v1/jobs/{job_id}` | Status of a recorded `PipelineJob` (HTTP 404 when unknown). |

### SSE streaming contract

When `POST /v1/ask` is called with `"stream": true`, the response is
`text/event-stream`. Each text delta is emitted as an SSE event whose data is a JSON
string, terminated by a sentinel event:

```
data: "token one "

data: "token two "

data: [DONE]
```

This is the single streaming contract for all service deployments (ADR-0005); the CLI's
`--stream` flag and the facade's `ask(stream=True)` use the in-process equivalent.

## Versioning

Boundary schemas are versioned and strict: all models derive from
`rag_core.base.RagBaseModel` (`extra="forbid"`), so unknown fields fail loudly at load
time instead of drifting silently. The pipeline specification carries
`PipelineConfig.schema_version` (constant `pipeline_schema_version = "1.0"` in
`rag_orchestrator.config`), which is surfaced by the `/version`, `/metrics`, and
`/v1/pipelines` endpoints so clients can reject an incompatible shape rather than
misinterpret it. A future breaking change to that shape bumps the constant and is
rejected at validation time.

Breaking changes to service APIs require a major version bump and a migration note in the
[CHANGELOG](../../CHANGELOG.md). See [ADR-0006](../../adr/ADR-0006-configuration-system.md)
for the configuration versioning policy.
