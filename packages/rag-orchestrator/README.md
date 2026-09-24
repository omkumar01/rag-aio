# rag-orchestrator
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Strategy and pipeline management, structured concurrency, and the FastAPI service
surface for the rag-aio platform.

`rag-orchestrator` is the glue that binds the contract-bound components from every
other `rag-*` package into a runnable RAG pipeline. It owns **pipeline
specification** (versioned, serializable configs), an in-memory **pipeline registry**,
the **runtime** (`Orchestrator.ask`), stage-cached **ingestion**, and a lazily-loaded
**FastAPI application**. It reimplements no vendor logic — all stages are delegated to
duck-typed components typed against the `rag_core` protocols.

## Installation

```bash
uv pip install rag-orchestrator
```

Core install pulls in only `rag-core` and the other `rag-*` stage packages. The
HTTP service runtime is optional:

```bash
uv pip install "rag-orchestrator[fastapi]"      # FastAPI + uvicorn
uv pip install "rag-orchestrator[pydantic-ai]"  # optional query-generation strategy
```

## Architecture / Design Principles

- **Declarative specs, not live components.** A `Pipeline` pairs a `PipelineConfig`
  with a lightweight map of *component-identifier strings* (e.g. `"DenseRetriever"`).
  It is cheap to serialize and audit; no backend is constructed at spec time.
- **Versioned configuration.** `PipelineConfig.schema_version` (currently `"1.0"`)
  lets a future breaking shape be rejected instead of silently misread. Each stage is a
  `StageConfig` subclass (`enable`, `strategy`, `policy`, `overrides`), validated with
  `extra="forbid"`.
- **Structured concurrency for retrieval.** The orchestrator fans queries out through
  `rag_query.retrieval_exec.ParallelRetrievalExecutor` (bounded by
  `config.concurrency`) using `asyncio.TaskGroup`, and enforces an end-to-end
  `asyncio.timeout(config.timeout_s)`. Timeouts surface as `OperationTimeout`; client
  disconnects surface as `asyncio.CancelledError`.
- **Dependency-light imports.** `import rag_orchestrator` never pulls FastEmbed, Qdrant,
  the doc-handler parsers, or FastAPI. Heavy backends are imported lazily *inside*
  `load_local_services()`; the FastAPI surface is gated behind the package's
  `__getattr__` and the `app` submodule so the `fastapi` extra is truly optional.
- **One protocol, many implementations.** `OrchestratorServices` is a typed *bag* of
  components typed only against `rag_core` protocols — every field is swappable.

## Public API

### Pipeline configuration

```python
from rag_orchestrator import PipelineConfig, pipeline_schema_version

config = PipelineConfig.local_default()  # sensible single-node local profile
print(pipeline_schema_version)  # "1.0"

# Stage overrides fold into stage.overrides at wiring time.
config = config.merge_overrides({"top_k": 25, "model": "gpt-4o-mini", "temperature": 0.0})

config.retrieval.strategy  # "rrf" | "dense" | "sparse" | ...
config.retrieval.policy  # "hybrid" | "dense" | ...
config.generation.policy  # "lm_studio" | "openai" | ...
```

| Field            | Default | Meaning |
| ---------------- | ------- | ------- |
| `schema_version` | `"1.0"` | Monotonic version tag; rejects drift. |
| `name`           | `"default"` | Human-readable pipeline name. |
| `ingestion`      | `IngestionStage` (`policy`, `strategy`, `overrides`) | Parser/strategy (`"pymupdf"`, `"fast"`). |
| `embedding`      | `EmbeddingStage` (`policy`, `strategy`, `overrides`) | Chunker + embedders (`"fastembed"`, `"mock"`). |
| `retrieval`      | `RetrievalStage` (`policy`, `strategy`, `overrides`) | `"hybrid"`/`"dense"`/`"sparse"`, `"rrf"`/`"weighted"`. |
| `rerank`         | `RerankStage` (`policy`, `strategy`, `overrides`) | `"rerank"`/`"none"`, `"heuristic"`. |
| `context`        | `ContextStage` (`policy`, `strategy`, `overrides`) | ordering policy (`"relevance_first"`, …). |
| `generation`     | `GenerationStage` (`policy`, `strategy`, `overrides`) | `"lm_studio"`/`"openai"`, etc. |
| `timeout_s`      | `None` | End-to-end `ask` budget (seconds). |
| `token_budget`   | `None` | Generation answer budget; propagated as the context budget when unset. |
| `cost_budget_usd`| `None` | Soft generation spend cap; enforced via the usage callback. |
| `concurrency`    | `4` (≥1) | Parallelism for independent retrieval branches. |
| `cache_reads`    | `True` | Honor per-stage cache reads. |
| `cache_writes`   | `True` | Write per-stage cache entries. |

`merge_overrides` routes recognized flat keys to their stage (`top_k`,
`candidate_k`, `model`, `temperature`, `max_tokens`, `token_budget`, `timeout_s`, …);
unknown keys land in `generation.overrides` so experimenters can pass
provider-specific options without fighting validation.

### Pipeline spec + registry

```python
from rag_orchestrator import (
    Pipeline,
    PipelineRegistry,
    default_pipeline,
    default_registry,
)

pipeline = default_pipeline()  # Pipeline from local_default()
pipeline.save_yaml("pipelines/local.yaml")  # persist spec

registry = default_registry()
registry.add(pipeline, version="1.0")
registry.add(Pipeline.from_config(custom_config), version="1.1")
registry.get("local_fast")  # first match across versions
registry.list()  # all specs (cheap: identifiers only)
registry.remove("local_fast", version="1.0")
registry.save("pipelines/all.yaml")  # multi-doc YAML list

loaded = PipelineRegistry.load("pipelines/all.yaml")  # round-trip back in
Pipeline.load_yaml("pipelines/local.yaml")
```

A `Pipeline`'s `components` map is best-effort (populated from `services` when
supplied) and is intended for audit, not for reconstructing live objects.

### Orchestrator

```python
from rag_orchestrator import Orchestrator, OrchestratorServices

orchestrator = Orchestrator(services, pipeline_config)  # config optional -> local_default()

# Non-streaming: a fully-assembled AskResult.
result = await orchestrator.ask(
    "What are the auth requirements?", query_id="abc", correlation_id="abc"
)
result.answer, result.citations, result.timings_ms, result.metrics

# Streaming: an async iterator of text deltas (same pipeline, SSE-friendly).
async for delta in await orchestrator.ask("Tell me a story", stream=True):
    print(delta, end="", flush=True)

# Per-request overrides are folded into a fresh PipelineConfig copy.
result = await orchestrator.ask("...", overrides={"top_k": 50, "temperature": 0.0})
```

`ask` runs `_process_query → _retrieve → _fuse → _rerank → _build_context →
_generate → _build_result`, each timed into `AskResult.timings_ms`
(`query`, `retrieve`, `rerank`, `context`, `generate`, `total`/`total_ms`).

#### `AskResult`

| Field        | Type | Notes |
| ------------ | ---- | ----- |
| `answer`     | `str` | Generated text (may be empty on streaming). |
| `citations`  | `list[Citation]` | Numeric `[n]`-style, built from the assembled context. |
| `context`    | `Context \| None` | Token-budgeted evidence (for inspection/debugging). |
| `timings_ms` | `dict[str, float]` | Per-stage + total wall-clock timings. |
| `query_id`   | `str` | Stable within one `ask`. |
| `metrics`    | `dict[str, Any]` | Usage/cost from generation + `cached`, `correlation_id`, `query_id`. |

`metrics.cached` reflects whether the generation result was served from the response
cache (when wired); it is always present and defaults to `False`.

### OrchestratorServices (the component bag)

`OrchestratorServices` is a dataclass of already-constructed, protocol-typed components.
All fields default (to `None` or `NoOpObserver`) so fixtures can build partial bags:

```python
from rag_orchestrator import OrchestratorServices

# Only the components this path needs:
services = OrchestratorServices(generator=my_generator, query_engine=my_engine, ...)
```

Key fields: `query_engine`, `retrievers`, `hybrid_retriever`, `fusion`, `reranker`,
`context_builder`, `generator`, `fallback_generators` (list), `model_router`,
`provider_registry`, `loader`, `parser_registry`, `ingestion_pipeline`, `chunker`,
`embedder`, `sparse_embedder`, `vector_store`, `document_store`, `embedding_pipeline`,
`cache`, `observer`. `ensure_retrievers()` returns the hybrid retriever first (when
present) followed by the dense/sparse list. `NoOpObserver` is the default `Observer`
implementation — a safe discard for components that don't opt into observability.

`_ServiceGenerator` adapts a `rag_generation.GenerationService` to the `Generator`
protocol: `generate` delegates to the service (reusing its router + fallback chain
for free), while `stream` resolves the effective provider and forwards to its native
`stream`.

### Local wiring

```python
from rag_orchestrator import load_local_services, PipelineConfig

services = load_local_services(
    qdrant_path="./data/qdrant",
    db_url="sqlite+aiosqlite:///./data/rag.db",
    lm_studio_url="http://localhost:1234/v1",
)
orchestrator = Orchestrator(services)  # picks up PipelineConfig.local_default()
```

`load_local_services` wires the `local_default` profile against local backends and
imports every heavy backend lazily inside the function. It returns a bag with FastEmbed
dense + sparse embedders, a recursive chunker, a dense retriever, a heuristic Jaccard
reranker, a token-budgeted context builder, and an LM Studio `OpenAICompatibleProvider`
wrapped in a `GenerationService`. (`rag-aio`'s `build_services` mirrors this for the
`"fastembed"` backend and substitutes `_StubGenerator` only for the `"mock"` backend.)

### Ingestion (stage-cached)

```python
from rag_orchestrator import ingest, ingest_directory

document = await ingest(services, "./contracts.pdf")  # load -> parse -> dedup -> embed -> index
docs = await ingest_directory(services, "./folder", recursive=True)
```

Ingestion is cached per `source + ingestion-config hash` (via
`rag_core.ids.config_hash` and `rag_cache.keys.cache_key`): re-ingesting an unchanged
source short-circuits from cache when `PipelineConfig.cache_reads` is enabled, and
writes are gated by `cache_writes`. Unsupported formats raise
`UnsupportedFormatError` (skipped in directory mode); other failures raise
`IngestionError`.

### FastAPI application

```python
from rag_orchestrator import create_app  # resolved lazily via __getattr__
from fastapi import FastAPI

app: FastAPI = create_app(services, pipeline_config)
```

`create_app` is **lazily** re-exported: `import rag_orchestrator` does *not* import
FastAPI; accessing `rag_orchestrator.create_app` / `.app` triggers the import only when
the `fastapi` extra is installed.

| Method   | Path                | Description |
| -------- | ------------------- | ----------- |
| `GET`    | `/health`           | `{"status": "ok"}`. |
| `GET`    | `/ready`            | Per-component readiness; `ready` is the AND of all checks. |
| `GET`    | `/metrics`          | Pipeline name, schema version, vector point count, cache stats. |
| `GET`    | `/version`          | Package version, schema version, pipeline name. |
| `GET`    | `/v1/pipelines`     | Serialized pipeline specs from the registry. |
| `POST`   | `/v1/ask`           | `AskRequest{query, stream, overrides, query_id, correlation_id}`. JSON when `stream=false`; SSE (`data: <delta>` + `data: [DONE]`) when `stream=true`. |
| `POST`   | `/v1/ingest`        | `IngestRequest{source, recursive}`. Returns the indexed document; tracks a `PipelineJob` for async polling. |
| `GET`    | `/v1/jobs/{job_id}` | Polls a previously-started ingest job (`JobStatus.completed|failed|…`). |

## Usage Guides

### Beginner: run a query against local services

```python
import asyncio
from rag_orchestrator import Orchestrator, load_local_services


async def main():
    services = load_local_services()  # LM Studio + local Qdrant + FastEmbed
    orch = Orchestrator(services)
    result = await orch.ask("What are the authentication requirements?")
    print(result.answer)
    for c in result.citations:
        print(f"[{c.citation_id}] {c.source_uri or c.document_id}")
    print(result.timings_ms)  # per-stage wall-clock (ms)


asyncio.run(main())
```

### Intermediate: streaming + per-request overrides

```python
result = await orchestrator.ask(
    "Write a 3-sentence summary.",
    stream=True,
    overrides={"top_k": 8, "temperature": 0.0, "max_tokens": 128},
)
async for delta in result:
    print(delta, end="", flush=True)
```

### Intermediate: inspecting explainability

`timings_ms` carries stage names (and `"<strategy>_failed"` markers from retrieval),
while `metrics` carries generation usage and the `cached` flag — enough to build a
latency/cost dashboard without instrumenting the stages yourself.

### Advanced: custom pipeline + serialized registry

```python
from rag_orchestrator import PipelineConfig, Pipeline, PipelineRegistry

cfg = PipelineConfig.local_default().model_copy(
    update={
        "name": "research-heavy",
        "retrieval": {
            "strategy": "rrf",
            "policy": "hybrid",
            "overrides": {"top_k": 25, "candidate_k": 100},
        },
    }
)
spec = Pipeline.from_config(cfg, services=services)  # records component identifiers
registry = PipelineRegistry()
registry.add(spec, version="1.0")
registry.save("pipelines/research.yaml")

# Reload later — specs deserialize without constructing anything:
registry = PipelineRegistry.load("pipelines/research.yaml")
orch = Orchestrator(services, registry.get("research-heavy").config)
```

### Advanced: custom FastAPI host

```python
import uvicorn
from rag_orchestrator import create_app, load_local_services

services = load_local_services()
app = create_app(services)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Configuration

`PipelineConfig` is the canonical orchestration config and is embedded verbatim as
`cfg.pipeline` inside `rag_aio.RAGConfig`. It is loaded from TOML by `RAGConfig.from_file`
(`tomllib`) and passed straight through, so the same per-stage `overrides` table that
the orchestrator consumes is what users edit in TOML:

```toml
[pipeline]
name = "local_fast"
schema_version = "1.0"
timeout_s = 30.0
token_budget = 256
cost_budget_usd = 0.10
concurrency = 4
cache_reads = true
cache_writes = true

[pipeline.retrieval]
policy = "hybrid"
strategy = "rrf"
[pipeline.retrieval.overrides]
top_k = 10
candidate_k = 50

[pipeline.generation]
policy = "lm_studio"
strategy = "openai_compatible"
[pipeline.generation.overrides]
base_url = "http://localhost:1234/v1"
model = "mistralai/ministral-3-3b"
temperature = 0.2

[pipeline.ingestion]
policy = "fast"
strategy = "pymupdf"
[pipeline.ingestion.overrides]
ocr = true
```

Validation: `RagBaseModel` (`extra="forbid"`) rejects unknown stage fields;
`concurrency` is constrained `>= 1`; `merge_overrides` routes known per-request keys
and funnels the rest into `generation.overrides`. See ADR-0006.

## Testing

Tests wire in stub components (see `tests/fakes.py`) and run fully offline — no LM
Studio, Qdrant, or cloud provider is required.

```bash
uv run ruff format packages/rag-orchestrator && uv run ruff check packages/rag-orchestrator
uv run mypy packages/rag-orchestrator/src
uv run pytest packages/rag-orchestrator -q
```

Test layout:

```
tests/
  fakes.py              # StubGenerator, StubVectorStore, StubCache, make_services()
  conftest.py           # makes fakes importable under importlib mode
  test_smoke.py         # import / version guard
  test_config.py        # local_default, schema_version, merge_overrides routing
  test_pipeline.py      # Pipeline + PipelineRegistry (add/get/list/remove, save/load YAML)
  test_orchestrator.py  # ask, streaming, timeout, fallback, metrics, cache flag
  test_ingest.py        # stage caching, dedup, UnsupportedFormatError skipping
  test_app.py           # FastAPI endpoints (/health, /ready, /metrics, /v1/ask, /v1/ingest, /v1/jobs)
```

## Dependencies

Core wiring (always installed):

- `rag-core` — domain models, protocols (`Generator`, `Retriever`, …), error taxonomy.
- `rag-doc-handler`, `rag-embedder`, `rag-db-handler`, `rag-retrieval`, `rag-rerank`,
  `rag-query`, `rag-context`, `rag-generation`, `rag-llm-provider`, `rag-cache`,
  `rag-observe`.

Optional extras:

- `fastapi` — `fastapi>=0.115`, `uvicorn[standard]>=0.30` (drives `create_app`).
- `pydantic-ai` — optional query-generation strategy.

Heavy backends (`qdrant_client`, `fastembed`, `bm25s`, NumPy) are imported lazily inside
`load_local_services()` and are **not** hard dependencies of this package.

## Cross-Package Relationships

- **Depends on every stage package**, consuming each through its `rag_core` protocol
  contract. `OrchestratorServices` is the typed bag that holds those components.
- **Bridges `rag-generation` ↔ `rag-llm-provider`** via `_ServiceGenerator`: it adapts a
  `rag_generation.GenerationService` to the `Generator` protocol while resolving the
  effective provider through `rag_llm_provider.ModelRouter`/`ProviderRegistry`.
- **Is consumed by `rag-aio`**: `RAG.build_services` either calls `load_local_services`
  (fastembed backend) or assembles the mock bag; `get_app`/`app` delegate to
  `create_app`. The orchestrator owns the runtime; `rag-aio` owns the CLI + facade +
  module-level app instance.
- **Ingestion** reuses `rag_doc_handler`'s `default_pipeline` (loader + parser registry)
  and `rag_embedder`'s `EmbeddingPipeline` (chunk → embed → index), adding stage caching
  keyed on content + config hashes.
- **Retrieval/execution** reuses `rag_query.retrieval_exec.ParallelRetrievalExecutor`
  (structured concurrency, `asyncio.TaskGroup`) and `rag_retrieval` fusion strategies.
- **Context assembly** is delegated wholesale to `rag_context` (`ContextBuilderImpl`,
  `build_citations`, `render_context`) and **routing** to `rag_query` (`QueryEngine`).
- See ADR-0002 (protocol-based contracts), ADR-0003 (Qdrant local-mode default),
  ADR-0005 (FastAPI service APIs), ADR-0006 (typed configuration).

## License

MIT
