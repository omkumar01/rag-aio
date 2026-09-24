# Modules

rag-aio ships as a uv workspace of 18 packages under `packages/` (ADR-0001). The
per-package READMEs under `packages/<name>/README.md` are the source of truth for each
module's purpose, public API, configuration, endpoints, failure modes, and operational
guidance — this page maps the workspace: one row per package, grouped by layer, with the
in-repo dependencies verified against each package's `pyproject.toml`.

Layer assignments follow
[docs/development/final-report.md](../development/final-report.md); the data flow between
them is described in [docs/architecture/README.md](../architecture/README.md).

| Package | Layer | Responsibility | Depends on (workspace) | README |
| --- | --- | --- | --- | --- |
| `rag-core` | Infrastructure | Canonical domain models, runtime-checkable Protocols, errors, IDs/hashes | — | [README](../../packages/rag-core/README.md) |
| `rag-observe` | Infrastructure | OpenTelemetry-API instrumentation, redacted logging, correlation IDs | rag-core | [README](../../packages/rag-observe/README.md) |
| `rag-cache` | Infrastructure | Multi-level caching (memory/disk/sqlite) with config-hash keys | rag-core | [README](../../packages/rag-cache/README.md) |
| `rag-db-handler` | Infrastructure | SQL, document, KV, vector and cache store abstractions + adapters | rag-core | [README](../../packages/rag-db-handler/README.md) |
| `rag-doc-handler` | Processing | Load, type-detect, parse (PDF/HTML/Office/email/Markdown, web crawl) into canonical `Document`s | rag-core | [README](../../packages/rag-doc-handler/README.md) |
| `rag-ocr` | Processing | Hybrid mechanical + semantic OCR with provenance-preserving regions | rag-core | [README](../../packages/rag-ocr/README.md) |
| `rag-embedder` | Processing | Chunking, tokenization, dense/sparse embedding, indexing orchestration | rag-core | [README](../../packages/rag-embedder/README.md) |
| `rag-mass-inject` | Processing | High-throughput bulk ingestion jobs: backpressure, checkpoints, dead letters, resume | rag-core, rag-doc-handler, rag-ocr, rag-embedder, rag-db-handler | [README](../../packages/rag-mass-inject/README.md) |
| `rag-query` | Intelligence | Query normalization, classification, strategies; parallel retrieval execution | rag-core, rag-retrieval | [README](../../packages/rag-query/README.md) |
| `rag-retrieval` | Intelligence | Dense/sparse(BM25)/hybrid retrieval, RRF and weighted fusion, explainability | rag-core, rag-db-handler | [README](../../packages/rag-retrieval/README.md) |
| `rag-rerank` | Intelligence | Second-stage ranking on small candidate sets (≤50) | rag-core | [README](../../packages/rag-rerank/README.md) |
| `rag-context` | Intelligence | Token-budgeted evidence selection, dedup, citation mapping | rag-core | [README](../../packages/rag-context/README.md) |
| `rag-llm-provider` | Intelligence | Provider/model registry, capabilities, aliases, secrets-by-reference, fallbacks | rag-core | [README](../../packages/rag-llm-provider/README.md) |
| `rag-generation` | Intelligence | Provider-agnostic generation: streaming, retries, structured output | rag-core | [README](../../packages/rag-generation/README.md) |
| `rag-orchestrator` | Service | Pipeline definitions (YAML round-trip), budgets/cancellation, `OrchestratorServices` wiring, FastAPI app | rag-core + 11 packages below (see note) | [README](../../packages/rag-orchestrator/README.md) |
| `rag-eval` | Operations | Retrieval and generation metrics, datasets, reproducible runs, comparison | rag-core | [README](../../packages/rag-eval/README.md) |
| `rag-ui` | Operations | Streamlit management console: config, health/ingestion dashboard, ask tab | rag-aio | [README](../../packages/rag-ui/README.md) |
| `rag-aio` | Facade | `RAG` facade, Typer CLI (`ingest`/`ask`/`serve`/`config`), FastAPI service composition | all of the above except `rag-ui` | [README](../../packages/rag-aio/README.md) |

Note on `rag-orchestrator`: its `pyproject.toml` depends on `rag-core`, `rag-cache`,
`rag-observe`, `rag-db-handler`, `rag-doc-handler`, `rag-embedder`, `rag-retrieval`,
`rag-rerank`, `rag-query`, `rag-context`, `rag-llm-provider`, and `rag-generation` — every
layer it drives, wired through the `OrchestratorServices` bag.

## Infrastructure

Four packages with no RAG logic. `rag-core` depends on nothing in-repo (pydantic only) and
defines every cross-package `typing.Protocol` plus the Pydantic boundary models, the error
taxonomy, and ID/hash helpers (`new_id`, `content_hash`, `config_hash`). `rag-observe`
implements the `Observer` protocol with OpenTelemetry API spans/metrics and mechanically
redacted logs; `rag-cache` and `rag-db-handler` turn persistence into swappable adapters —
backends are chosen through factories (`create_cache`, `create_vector_store`,
`create_sql_store`), never by importing vendors at call sites. Extension point: add a
backend by implementing the relevant `rag-core` protocol and adding a factory branch;
consumers keep seeing only the protocol.

## Processing

The ingestion path. `rag-doc-handler` owns format detection and the plugin parser registry,
delegating OCR through the `OCRFallback` hook; `rag-ocr` supplies mechanical
(RapidOCR-class) extraction with VLM escalation, returning provenance-preserving regions.
`rag-embedder` chunks, embeds dense and sparse, and indexes — writing only through
`rag-db-handler` contracts — with incremental reindexing driven by
`content_hash:config_hash` pairs. `rag-mass-inject` scales this to corpora with bounded
queues, per-stage workers, and resumable jobs. Extension point: register a new parser in
the `ParserRegistry` or implement a `Chunker`/`Embedder`; the pipelines accept any
protocol-conformant object. Note that `rag-mass-inject` does not depend on
`rag-orchestrator` at runtime: it accepts any object structurally compatible with
`OrchestratorServices` (the import is `TYPE_CHECKING`-only), which keeps its runtime
surface to the five packages listed above.

## Intelligence

The query path. `rag-query` turns one question into de-duplicated variants
(original/rewrite/expansion/decomposition/HyDE) and executes retrieval branches
concurrently; `rag-retrieval` runs dense, sparse, and hybrid search with
`FusionStrategy`-pluggable merging; `rag-rerank` reorders the ≤50-candidate shortlist;
`rag-context` fits evidence into a token budget and maps numeric citations. Providers are
decoupled twice over: `rag-llm-provider` holds the registry/routing/alias control plane,
and `rag-generation` talks to providers only through its configs. Extension point:
implement rag-core's `Retriever` protocol and drop it into a `HybridRetriever`, or
implement a `QueryStrategy`/`FusionStrategy`/`Reranker` and inject it via
`OrchestratorServices`. `rag-query` is the one intelligence package that depends on a
sibling (`rag-retrieval`) because its `ParallelRetrievalExecutor` drives the retrievers
directly.

## Service and facade

`rag-orchestrator` is the only package that wires the whole graph: it sequences
query → retrieval → fusion → rerank → context → generation inside structured concurrency
(`asyncio.TaskGroup`, deadline propagation, `OperationTimeout`), defines pipelines that
round-trip via YAML (`Pipeline.save_yaml`/`load_yaml`), and exposes the FastAPI app
factory. `rag-aio` is the composition surface: the `RAG` facade
(`RAG.from_config(...)` → `ingest` → `ask`), the Typer CLI, and the service composition.
Heavy backends (`fastembed`, `qdrant_client`) are imported lazily inside
`load_local_services()`, so importing either package stays cheap. Independently of this
layer split, every module with expensive state (doc-handler, ocr, embedder, retrieval,
rerank, generation) exposes an identical FastAPI service API with `/health`, `/ready`,
`/metrics`, `/version` (ADR-0005), and long operations run as jobs with IDs (ADR-0007).

## Operations

`rag-eval` measures retrieval and generation quality with an in-house metrics registry —
register a custom `EvaluationMetric` and it appears in run reports and `compare` — with
reproducible runs keyed by config hash. `rag-ui` is the Streamlit console; it depends on
`rag-aio` alone and talks to the service APIs with the same Pydantic config models
(secrets masked), so anything configurable in TOML is editable in the UI.

## Seeing it in code

Runnable per-stage scripts live in [examples/stages/](../../examples/stages/):
[ingest.py](../../examples/stages/ingest.py) (parse + chunk),
[retrieve.py](../../examples/stages/retrieve.py) (dense/hybrid + RRF),
[rerank.py](../../examples/stages/rerank.py), [context.py](../../examples/stages/context.py)
(token budget + citations), and [generate.py](../../examples/stages/generate.py) — plus the
end-to-end [quickstart.py](../../examples/quickstart.py) and
[fully_local.py](../../examples/fully_local.py) covered in the
[examples README](../../examples/README.md). For the full request/response walkthrough,
read [docs/architecture/README.md](../architecture/README.md).
