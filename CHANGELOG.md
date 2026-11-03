# Changelog

All notable changes to rag-aio are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/).

<!-- Add new entries here, above the latest release. -->

## [Unreleased]

### Added

- **OpenAI-compatible remote embedder** (`rag-embedder`): new
  `OpenAICompatibleEmbedder` turns any `POST {base_url}/embeddings` endpoint
  (LM Studio, Ollama's OpenAI mode, vLLM, OpenAI, ...) into a
  `rag_core.protocols.Embedder`, interchangeable with the local `FastEmbedDense`
  across ingestion and query-time pipelines. Follows the existing remote-client
  conventions (`RemoteReranker`, OCR `VLMSemanticExtractor`): secrets by
  reference via `api_key_ref` (env-var name, resolved at request time, never
  stored on the instance), request batching with semaphore-bounded concurrency,
  index-aligned response parsing, optional `dim` declaration/learning and
  L2-normalization, and uniform `rag_core.EmbeddingError` failures. Adds a
  runtime `httpx` dependency to `rag-embedder`; tests use `httpx.MockTransport`
  (no network).
- **`RAG_AIO_` environment-variable overrides** (`rag-aio`): the env tier of the
  ADR-0006 precedence chain (defaults → TOML → env) is now implemented in
  `RAGConfig.from_file`. `RAG_AIO_SECTION__FIELD=VALUE` maps to nested config
  (double underscore splits, segments lowercased); values are parsed with
  `json.loads` (raw-string fallback); unknown top-level keys are dropped. See
  `docs/configuration/README.md` and `rag_aio.config.env_overrides`.
- **`Document.text_hash`** (`rag-core`): additive field hashing the text alone
  (`content_hash` keeps its `source_uri + text` semantics), enabling content-level
  dedup across paths.
- **Lifecycle helpers** (`rag-core`): `Closable` protocol and
  `aclose_component()` in `rag_core.lifecycle` — close sync-or-async `close`/
  `aclose` methods idempotently without touching existing protocol signatures.
- **`OrchestratorServices.aclose()`** (`rag-orchestrator`): idempotent shutdown of
  all owned components (vector store → document store → cache → other closables).
  `examples/fully_local.py` and the facade `RAG` close path now use it.
- **`load_local_services(document_db_url=...)`** (`rag-orchestrator`): explicit
  name for the document-store connection; the old `db_url` parameter keeps working
  but emits a `DeprecationWarning`.
- **Public step/streaming API** (`rag-orchestrator`): `Orchestrator.prepare() ->
  PreparedQuery`, `ask_stream() -> AsyncIterator[str]`, and
  `stream_events() -> AsyncIterator[StreamDelta | StreamFinal]` — streaming no
  longer discards citations/timings (the terminal `StreamFinal` carries answer,
  citations, timings, and query_id), and integrators no longer need private
  internals. `ask(stream=True)` still works.
- **Per-request namespace threading** (`rag-orchestrator`): `namespace` accepted on
  `Orchestrator.ask/ask_stream/stream_events/prepare` and on `/v1/ask` +
  `/v1/ingest` request bodies; stores/retrievers already supported namespaces.
- **Optional auth/tenant hooks** (`rag-orchestrator.app.hooks`):
  `AuthContext`, `RequestAuthenticator`, and `NamespaceResolver` protocols;
  `create_app(..., authenticator=..., namespace_resolver=..., on_job_complete=...)`.
  With an authenticator set, `/v1/*` requires auth (401 otherwise); the resolver's
  namespace wins over the request body for tenant isolation; `on_job_complete`
  fires a webhook-style callback on terminal ingest jobs. Default behavior with no
  hooks is unchanged. See `docs/deployment/README.md`.
- **Opt-in typed SSE events** (`rag-orchestrator.app`): `AskRequest.events=True`
  switches `/v1/ask` streaming to `event: delta` / `event: done` frames (done
  carries answer/citations/timings/query_id); the default JSON-deltas + `[DONE]`
  protocol is unchanged.
- **Mock services in the orchestrator** (`rag-orchestrator`): `build_mock_services`
  (and `StubGenerator`) moved from the facade into
  `rag_orchestrator.mock_services`, so consumers depending on members directly can
  assemble the offline stack; the facade delegates.
- **Sparse feature detection** (`rag-retrieval`): `supports_sparse(store)`
  (`TypeGuard[SparseSearchStore]`); `load_local_services` auto-wires
  `SparseRetriever` when the store implements `search_sparse` (Qdrant today keeps
  the documented dense-only wiring). Corrected the README claim that
  `QdrantVectorStore` supports learned-sparse search.
- **SSE progress bridge** (`rag-observe`): `rag_observe.sse.sse_stream(hub)` turns
  `ObservabilityHub` events into SSE frames (`data: {"event": ..., "attributes": ...}`)
  with optional filtering and leak-free unsubscribe.
- **Docs**: deployment guide gained "Authentication & multi-tenancy" (incl. the
  `user_{id}` + `global` merge recipe and the wrap-`create_app` pattern),
  "Shutdown ordering", "Sizing on process RSS", and SSE/job-callback sections;
  configuration docs document the now-implemented env tier; reserved
  (`postgres`/`pgvector`/`faiss`) extras and the facade-only `qdrant`/`fastembed`
  extras story are documented in the READMEs.

### Changed

- **`InMemoryVectorStore.count` is now `async`** (`rag-db-handler`), matching
  `QdrantVectorStore.count` for protocol consistency. Call sites that awaited the
  result are unaffected; direct sync callers must now await (the FastAPI metrics
  endpoint already guards with `isawaitable`).

### Removed

- **Web example UI** (`web-example/`): the standalone FastAPI + Jinja2 + HTMX
  web application was removed from the workspace (commits `1944ad2` / `c13a01a`)
  before release; the server/app layer it exercised is now covered by the
  first-class auth hooks, namespace threading, and typed SSE shipped in this
  release (see Added), and the web app itself will live in its own repository.

### Fixed

- **`rag-aio-db-handler` / `rag-aio` now declare `sqlalchemy[asyncio]>=2.0`**: the
  SQL stores import `sqlalchemy.ext.asyncio` eagerly, which failed with
  "SQLAlchemy asyncio module requires greenlet" in consumer environments that did
  not happen to install `greenlet`. The asyncio extra pins the dependency
  correctly (was bare `sqlalchemy>=2.0`).
- **Ingestion stage cache invalidated by source changes** (`rag-orchestrator`):
  the cache key now includes a cheap source fingerprint (local-file `mtime_ns:size`),
  so re-ingesting a *changed* file re-indexes instead of short-circuiting on the
  stale URI-keyed entry (which previously survived out-of-band vector purges).
  Non-file sources behave as before.
- **Rerank probe outcome is logged** (`rag-rerank`): `RemoteReranker` emits one
  INFO line (`probe <base_url> -> mode=<rerank|chat> (status=...)`) when it
  auto-detects the provider contract, so consumers can verify which path was
  taken. `DECISIONS.md` entry moved from "Open questions" to Accepted.

## [v0.1.0]

Initial build of the rag-aio platform: an asynchronous, modular RAG system delivered as a
uv workspace of 18 packages. Verification at close: full gate green (872 package tests +
6 benchmark-harness unit tests, 91% coverage), `uv build --all-packages` produced all 18
packages with a clean wheel import smoke test, a live e2e run over LM Studio
(ingest → ask → cited answer), and a clean Mimosa security scan. See the
[final report](docs/development/final-report.md) for the full record.

### Added

- **Single-project PyPI install**: everything publishes as one `rag-aio`
  distribution — a single wheel bundles all 18 packages' code. Base install
  pulls only light third-party deps and runs the fully-offline mock profile;
  heavy backends are extras (`qdrant`, `fastembed`, `sentence-transformers`,
  `faiss`, `postgres`, `pgvector`, `redis`, `docling`, `msg`, `rtf`,
  `pydantic-ai`, `streamlit`) plus an `all` catch-all. Development stays a
  multi-package uv workspace; a custom hatch build hook bundles the members'
  sources into the release wheel (wheel-only builds).
- **Lazy backend imports**: `qdrant_client` in `rag-db-handler` loads on first
  store construction with an actionable `pip install "rag-aio[qdrant]"` hint,
  and `rag_orchestrator` exposes its heavy surfaces via PEP 562 lazy attributes
  — keeping `import rag_aio` functional with the base install alone. The
  declarative pipeline configuration (`PipelineConfig` and stage models) moved
  to `rag_core.pipeline_config` (re-exported from `rag_orchestrator.config`
  for compatibility).
- **Workspace**: 18-package uv monorepo under `packages/`, layered as infrastructure
  (`rag-core`, `rag-observe`, `rag-cache`, `rag-db-handler`), processing (`rag-doc-handler`,
  `rag-ocr`, `rag-embedder`, `rag-mass-inject`), intelligence (`rag-query`,
  `rag-retrieval`, `rag-rerank`, `rag-context`, `rag-llm-provider`, `rag-generation`),
  service (`rag-orchestrator`), operations (`rag-eval`, `rag-ui`), and facade (`rag-aio`).
  All cross-module interaction via `rag-core` Protocols (ADR-0002).
- **Facade and entry points**: `rag-aio` package with the `RAG` facade
  (`from rag_aio import RAG`), a Typer CLI (`ingest`/`ask`/`serve`/`config`), and FastAPI
  service composition reusing the orchestrator's app factory.
- **Management console**: `rag-ui` Streamlit app (Config / Dashboard / Ask tabs) sharing
  the backend's Pydantic `RAGConfig` models, with secret masking and a fault-tolerant
  HTTP `Dashboard` client.
- **Bulk ingestion and evaluation**: `rag-mass-inject` (checkpointed high-throughput
  ingestion jobs with backpressure, dead-letter handling, and resume) and `rag-eval`
  (retrieval/generation metrics, datasets, reproducible runs, run comparison).
- **Orchestrator**: YAML round-trippable pipeline definitions, strategy composition,
  budgets, cancellation, structured concurrency, and the `OrchestratorServices` wiring
  bag with lazy loading of heavy backends.
- **Benchmarks**: pytest-based harness in `benchmarks/` (nine scenarios; percentile and
  cold/warm methodology documented in `benchmarks/README.md`) plus measured wave-8
  results in `docs/performance/README.md`.
- **Examples**: offline quickstart, full-local-stack example, per-stage building blocks
  (`examples/`), and end-to-end e2e tests that auto-skip when LM Studio is unreachable.
- **Documentation**: `architecture.md`, `design.md`, ADR-0001..0008, and the `docs/` tree
  (api, architecture, configuration, deployment, development, modules, performance,
  testing).
- **Tooling**: `scripts/check.py` quality gate (ruff format/lint, mypy strict, pytest,
  benchmark-harness unit pass), GitHub Actions CI (ubuntu/windows, Python 3.12/3.13,
  coverage status) and PyPI release via trusted publishing, optional Qdrant server
  compose file (`docker-compose.qdrant.yml`).
- **Security**: Mimosa deep scan of 275 source files and 120 dependency packages — 0
  findings (recorded in `docs/development/security.md`); secrets-by-reference throughout.
