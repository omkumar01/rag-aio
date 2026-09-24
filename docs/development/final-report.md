# rag-aio — Final Report (Wave 8 / project close)

Date: 2026-09-24. This report closes the build of the rag-aio monorepo: an asynchronous,
modular RAG platform delivered as a uv workspace of 18 independently testable, independently
deployable packages (ADR-0001). It covers modules and boundaries, the configuration model,
supported backends, benchmark methodology and measured results, limitations, extension
points, recommended topologies, and the verification record.

## 1. Modules and boundaries

Packages are layered; each layer may only depend on contracts from `rag-core` and on
lower/equal layers — never on a concrete sibling implementation (ADR-0002). All cross-module
interaction goes through runtime-checkable Protocols in `rag-core`; boundary data is
Pydantic v2, hot-path internals are dataclasses/NumPy.

**Infrastructure layer** (foundation, no RAG logic):

- `rag-core` — canonical domain models, Protocol contracts, errors, IDs/hashes. Depends on
  nothing in-repo; everything else depends on it.
- `rag-observe` — OpenTelemetry spans/metrics, structured logging, correlation IDs, redaction.
  Depends on `rag-core` only; exports to pluggable OTel backends.
- `rag-cache` — multi-level caching (memory / disk / sqlite; redis extra) with config-hash
  keys. Depends on `rag-core`; consumed optionally by any layer.
- `rag-db-handler` — persistence abstractions and adapters: SQL, document, KV, vector, sparse,
  cache stores. Depends on `rag-core`; owns no pipeline logic.

**Processing layer** (ingestion path):

- `rag-doc-handler` — format detection and parsing (PDF, HTML, web/sitemap crawling, Office,
  email, Markdown) into canonical `Document`s. Delegates OCR to `rag-ocr`.
- `rag-ocr` — hybrid mechanical + semantic OCR with provenance-preserving regions; VLM calls
  delegated to providers.
- `rag-embedder` — chunking, tokenization, dense/sparse embedding, indexing orchestration.
  Writes only through `rag-db-handler` contracts.
- `rag-mass-inject` — high-throughput bulk ingestion jobs with checkpoints, backpressure,
  dead-letter handling and resume. Uses the processing modules via contracts.

**Intelligence layer** (query path):

- `rag-query` — query normalization, rewriting, expansion, routing; parallel strategy
  execution. Rewrites delegate to `rag-generation`.
- `rag-retrieval` — dense/sparse (BM25)/hybrid retrieval, RRF and weighted fusion, filters,
  explainability. Reads vector stores via `rag-db-handler` contracts.
- `rag-rerank` — second-stage ranking on small candidate sets (≤50); heuristic,
  cross-encoder, and remote (LM Studio) scorers.
- `rag-context` — token-budgeted evidence selection, dedup, citation mapping; tokenizer via
  `rag-embedder` contracts.
- `rag-llm-provider` — provider/model registry, capabilities, aliases, secrets-by-reference,
  fallback chains. Depends on `rag-core` only.
- `rag-generation` — provider-agnostic generation with streaming, retries, structured output;
  talks to providers via `rag-llm-provider` configs.

**Service layer:**

- `rag-orchestrator` — pipeline definitions (YAML round-trip), strategy composition, budgets,
  cancellation (`OperationTimeout`), structured concurrency; wires everything through the
  `OrchestratorServices` bag. Heavy backends (`fastembed`, `qdrant_client`) are imported
  lazily in `load_local_services()`, never at module load.

**Operations layer:**

- `rag-eval` — retrieval and generation quality metrics (in-house, no ranx), datasets,
  reproducible runs with config hashes, run comparison.
- `rag-ui` — Streamlit management console: config editing (same Pydantic models, secrets
  masked), health/ingestion dashboard, and ask tab; hits the service APIs.

**Facade layer:**

- `rag-aio` — `RAG` facade (`from rag_aio import RAG`), Typer CLI (`ingest`/`ask`/`serve`/
  `config`), FastAPI service composition reusing the orchestrator's app factory.

Every module with expensive state (doc-handler, ocr, embedder, retrieval, rerank, generation)
additionally exposes an identical FastAPI service API with `/health`, `/ready`, `/metrics`,
`/version` (ADR-0005), and long operations run as jobs with IDs (ADR-0007).

## 2. Configuration model

- One typed Pydantic v2 hierarchy (ADR-0006). Top-level `RAGConfig` composes `pipeline`,
  `embedder`, `storage`, `generation` sub-configs; all models inherit `RagBaseModel` with
  `extra="forbid"`, so unknown TOML keys fail loudly instead of being silently ignored.
- Loading: `RAGConfig.from_file(path)` via Python 3.12+ `tomllib`; the facade is
  `RAG.from_config(...)`. Precedence (lowest → highest): built-in defaults → config file →
  environment (`RAG_AIO_` prefix) → persisted UI configuration → per-request override.
- Secrets are references, never values: provider credentials are stored as environment-
  variable names (`api_key_ref`) and resolved/masked mechanically in logs, traces, UI, and
  API responses (ADR-0004/0006).
- `RAGConfig.mock()` returns a fully-offline wiring: `embedder.backend="mock"` plus
  `pipeline.embedding.policy="mock"` so downstream components use deterministic, hash-based
  mock implementations — this powers CI, tests, and the benchmark scenarios.
- Every execution records a non-secret configuration hash for reproducibility.

## 3. Supported databases and providers

Verified against actual package exports (not just doc claims):

- **Vector stores** (`rag-db-handler`): `InMemoryVectorStore`, `QdrantVectorStore`
  (Qdrant local mode by default at `./data/qdrant`, per ADR-0003, or Qdrant server via URL;
  see `docker-compose.qdrant.yml`). Construction via `create_vector_store`.
- **KV / document stores** (`rag-db-handler`): `SQLStore`, `SQLDocumentStore`,
  `SQLKeyValueStore` (default `sqlite+aiosqlite:///./data/rag.db`), `InMemoryKeyValueStore`.
  **Caches** (`rag-cache`): memory, disk, sqlite backends via `create_cache(CacheConfig)`
  (redis extra available).
- **Embedders** (`rag-embedder`): `FastEmbedDense` / `FastEmbedSparse` (ONNX; default dense
  model `BAAI/bge-small-en-v1.5`) and `MockEmbedder` (deterministic, hash-based, offline).
- **Rerankers** (`rag-rerank`): `HeuristicReranker` (lexical baseline),
  `CrossEncoderReranker`, `RemoteReranker` (OpenAI-compatible; auto-detects LM Studio
  `/rerank` vs chat fallback).
- **Generation providers** (`rag-generation`): `OpenAICompatibleProvider` (default; LM
  Studio at `http://localhost:1234/v1`, also vLLM or any compatible server — ADR-0004),
  `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`. Created via
  `create_provider` and orchestrated by `GenerationService` with `ModelRouter` fallbacks.
- **Retrieval/fusion** (`rag-retrieval`): `DenseRetriever`, `BM25Retriever`/`SparseRetriever`,
  `HybridRetriever`, `ReciprocalRankFusion`, `WeightedScoreFusion`.

## 4. Benchmark methodology and results

Harness: pure pytest (`benchmarks/`), deliberately not pytest-benchmark to stay
dependency-light. Nine scenarios run under `-m benchmark`; each performs one cold run
(lazy init: parser registries, BM25 index build, model load), 1–2 discarded warmups, then
5–20 timed iterations. Wall-clock via `time.perf_counter`; percentiles via linear
interpolation at rank `(n-1)·p/100` (same as `statistics.quantiles(method="inclusive")`);
mean, median, min, max and sample stdev recorded alongside P50/P90/P95/P99. Workloads are
deterministic (synthetic documents from a fixed word bank; hash-based mock embedders).
Peak RAM is measured with `tracemalloc` and is a **lower bound**: Python-level allocations
only — native buffers (ONNX runtime, model weights, qdrant storage) are invisible; VRAM is
not measured. Reports land in `benchmarks/results/` (gitignored) as JSON + markdown with
platform and config hash. Absolute numbers are machine-dependent — use relative comparisons
and trend lines, not the raw figures.

Measured run — 2026-09-24, dev workstation (Windows 11, 12 CPUs, Python 3.14 interpreter),
mock/offline scenarios plus local LM Studio; config hash `6183dc5d35a518bb`:

| Scenario | Workload | Result |
| --- | --- | ---: |
| doc parse + chunk | 30 synthetic docs | 114 ms |
| mock embedding | 200 chunks | 82 ms |
| FastEmbed (`bge-small-en-v1.5`) | 32 texts (model cached) | 513 ms |
| dense retrieval | 200 mock-embedded chunks, candidate_k=50 | 68.7 ms mean |
| hybrid retrieval (dense + BM25 + RRF) | 200 chunks | 64.7 ms mean |
| rerank (heuristic) | 50 candidates | 1.8 ms |
| context build | 50 hits, 2048-token budget | 5.9 ms |
| full mock query (`RAGConfig.mock()`) | ingest 10 docs + ask | 25.5 ms |
| generation | local LM Studio, small `max_tokens` | ~35 ms mean |

All 8 documented scenario families ran (generation auto-skips when `localhost:1234` is
unreachable; here it answered). Not yet measured (planned per docs/performance/README.md):
OCR throughput, standalone chunking/indexing throughput, sparse-only retrieval latency,
concurrency scaling, cache-hit performance, VRAM.

## 5. Limitations

- `tracemalloc` RAM figures cover Python allocations only (see §4); native/model memory and
  VRAM are unreported.
- The Mimosa security scan notes parts of the call graph are dynamic dispatch (Pydantic
  models, duck-typed provider protocols), so static cross-file reachability is partially
  incomplete; the e2e suite is the runtime complement.
- Benchmarks cover the documented metrics only; the "planned" list above is unmeasured, and
  absolute numbers do not transfer across machines.
- LM Studio–dependent features (generation and remote-rerank benchmarks, e2e tests) auto-skip
  when the local endpoint is unreachable, so results depend on the environment.
- The default topology is single-node and local-first: Qdrant local mode + sqlite + localhost
  model servers. Scaled multi-service deployment (Qdrant server, PostgreSQL, Redis behind
  FastAPI services) is supported by design but not exercised by the benchmark suite.
- Property-based and failure-mode tests exist per design.md, but no distributed-load or
  long-soak testing was performed.

## 6. Extension points

- **Retrieval**: implement `Retriever.retrieve(query) -> RetrievalResult` and drop it into a
  `HybridRetriever`; implement `FusionStrategy` for new fusion (RRF and weighted fusion ship
  as references).
- **Generation providers**: implement the provider protocol and register it, or point
  `OpenAICompatibleProvider` (or `create_provider`) at any OpenAI-compatible server — no core
  changes needed.
- **Storage**: add a `VectorStore`/`DocumentStore`/`KeyValueStore` implementation and a
  `create_vector_store` factory branch; consumers only see `rag-core` protocols.
- **Pipelines**: orchestrator `Pipeline` definitions round-trip via YAML
  (`Pipeline.save_yaml`/`load_yaml`) for declarative topologies.
- **Orchestrator wiring**: `OrchestratorServices` is a services bag — swap any component
  (observer, caches, retrievers, generator) without touching orchestration logic; heavy
  backends are injected via `load_local_services()`.
- **Evaluation**: `rag-eval` has a metrics registry — add custom retrieval/generation metrics
  and they appear in run reports and `compare`.
- **Configuration**: `RAGConfig` composition is the single wiring surface; new components are
  exposed as new optional sub-config fields (validated, `extra="forbid"`), so downstream
  code stays config-driven.

## 7. Recommended topologies

1. **Fully-offline mock** (development, CI): `RAGConfig.mock()` — MockEmbedder +
   InMemoryVectorStore + HeuristicReranker + SimpleTokenizer. Zero network, deterministic,
   runs in ~25 ms per full query. Used by the unit suite and most benchmarks.
2. **Single-machine local** (default profile, per architecture.md): PyMuPDF parse →
   token-aware chunking → FastEmbed dense + BM25 sparse → Qdrant local mode → hybrid RRF →
   heuristic/cross-encoder rerank → token-budgeted context → LM Studio for generation.
   In-process platform + localhost model server; matches docs/deployment topologies 1–2.
3. **OpenAI-compatible server farm** (docs/deployment topology 3): the same
   `OpenAICompatibleProvider` pointed at LM Studio/vLLM behind a shared endpoint, with Qdrant
   server (`docker-compose.qdrant.yml`), PostgreSQL for metadata, Redis for cache; selected
   modules (doc-handler, ocr, embedder, retrieval, rerank, generation) deployed as FastAPI
   services behind the orchestrator when scale demands it.

## 8. Verification summary

- `uv run python scripts/check.py` fully green: ruff format + lint, mypy strict across
  `packages/*/src`, 872 package tests + 6 benchmark-harness unit tests, 91% coverage.
- Packaging: `uv build --all-packages` produced all 18 packages (36 artifacts in `dist/`);
  wheel import smoke-tested from a fresh venv.
- Security: Mimosa deep scan on 2026-09-24 — 275 source files, 120 dependency packages,
  **0 findings** (scan ID `scan-2026-09-24T15-32-00.389Z-3f46166e9990`, seal
  `sha256:b5d42d4b81a722d8ab225be214a4a382f21a9c885c7bef2b63934bb8b65016c2`); see
  docs/development/security.md.
- E2E: live run over LM Studio — ingest a real document → ask → answer with citations
  (marked `e2e`, auto-skips when `localhost:1234` is down); quickstart from the README
  verified offline (mock path) and live.
- Reproduction from a clean environment: `uv sync && uv run python scripts/check.py`.
