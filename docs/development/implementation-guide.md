# Implementation Guide — Remaining Waves

This document captures the concrete, verified contracts and wiring required for the
remaining implementation waves (orchestrator completion + cross-module contract review,
Wave 7 surfaces, Wave 8 hardening). It assumes Waves 0–5 are complete (they are).

## 0. Where things stand

The monorepo workspace, rag-core contracts, and the 15 engine modules all live under
`packages/`. All public APIs are importable and individually green under
`scripts/check.py --fast` and `mypy packages/*/src` (strict). The shared
convention for the rest of the build is documented here once and reused.

## 1. Conventions every subagent must follow

- **Strict boundary**: each agent writes only inside its package directory.
- **Package-scoped gates only** during a wave (rate-limit safety): `ruff format/check <pkg>`,
  `mypy <pkg>/src`, `pytest <pkg>`. The **main agent** runs the repo-wide gate
  (`scripts/check.py --fast`) at wave boundaries.
- **TDD**: failing tests first.
- **Sequential launch when agents die of rate limits**: if two agents in one wave hit
  `"Provider rate limited" before any file write, relaunch one-at-a-time, not in parallel.

## 2. The cross-module contract matrix (verified signatures)

Every orchestrator wire crosses two packages. The exact names/call-shapes the
orchestrator must call, gathered from each package's `__init__.py`:

### 2.1 Ingestion path
- `rag_doc_handler` → `default_pipeline()` / `IngestionPipeline`; `ingest(source, data=None, *, dedup=None) -> tuple[Document, bool]` (document + is-new flag; uses `rag_ocr.OCRPipeline` via the PDF parser's
  `ocr_fallback` param; duck-typed `OCRFallback = Callable[[str,int,bytes], Awaitable[list[PageBlock]]]`).
- `rag_embedder` → `create_chunker(ChunkerConfig, tokenizer)`, `FastEmbedDense`/`FastEmbedSparse`/`MockEmbedder`,
  `EmbeddingPipeline.process(document, store) -> PipelineOutcome`
  (`PipelineOutcome` has `chunks: int, chunks_indexed: int, dims: int | None,
  model: str, sparse_model: str | None, took_ms: float, reindexed: bool`).
  `should_reindex(document, config_hash, seen: dict[str,str]) -> bool` (staticmethod).

### 2.2 Retrieval → rerank → context → generation path
- `rag_query` → `QueryEngine(config, strategies=None).process(query) -> QueryResult`
  (`strategies` optional; `QueryResult.variants: list[QueryVariant]`, `timings_ms`);
  `ParallelRetrievalExecutor(retrievers, max_concurrency).execute(query, variants, top_k) -> list[RetrievalResult]`.
- `rag_retrieval` → `ReciprocalRankFusion(k=60.0).fuse(results, top_k, weights=None, dedup=True)`;
  `DenseRetriever(store, embedder, config)`; `HybridRetriever(retrievers, fusion, config)`
  implements `Retriever.retrieve(query) -> RetrievalResult`.
- `rag_rerank` → `RerankPipeline(reranker, config)`; `config = RerankConfig(top_k=10, max_candidates=50)`;
  adapters: `HeuristicReranker(config)`, `RemoteReranker(base_url, model, ...)` (LM Studio rerank
  auto-detects `/rerank` vs chat fallback — verified),
  `CrossEncoderReranker(model_name, config, model_factory=None)`.
- `rag_context` → `ContextBuilderImpl(tokenizer, ContextConfig(...))`;
  `ContextConfig.token_budget`, `citation_style="numeric"`, `strategy`.
  `cstr.build(query, hits, token_budget) -> Context`;
  `render_context(context, style="numbered") -> str`;
  `build_citations(context, source_lookup=None) -> list[Citation]`.
- `rag_generation` → `OpenAICompatibleProvider(base_url="http://localhost:1234/v1", ...)`;
  `GenerationService(providers, router, fallback_names=[])`;
  `Service.generate(request) -> GenerationResult` (retries/fallback); `collect_stream(stream)`.
- `rag_llm_provider` → `ProvidersConfig.local_default()`, `ProviderRegistry`
  (`registry.resolve_role("generate") -> tuple[ProviderConfig, ModelConfig] | None`),
  `ModelRouter(registry)` (async `.route(role, ...) -> RouteDecision`).

### 2.3 Tokenizers for context
- `rag_embedder.tokenization`: `SimpleTokenizer` (fast, offline) and `HFTokenizer.from_identifier`.
  `SimpleTokenizer` is the **chunking** tokenizer by default; the context-budget tokenizer is
  `rag_context.tokenizer.WhitespaceCounter` wrapped in `ContextTokenizer`; both switchable via config.

### 2.4 Observability / cache
- `rag_observe`: `observe(stage, attributes=None)` (no `stage` function), `timed(name)`, `StageTimer`,
  `ObservabilityHub().record(event, attributes)`, `set_correlation_id(cid)`.
  The orchestrator's default wiring uses `NoOpObserver`; wire `ObservabilityHub()` for real telemetry.
- `rag_cache`: `cache_key(namespace, **parts)`, `create_cache(CacheConfig)`,
  `NamespacedCache(inner, namespace)`, `InstrumentedCache` (exposes `.stats` → `CacheStats`).

### 2.5 Errors / jobs
- `rag-observe` and `rag-cache` are **not** declared dependencies of `rag-embedder`,
  so the orchestrator must supply them to embedder/retrieval paths where optional.

## 3. Wave 6 — rag-orchestrator + contract review (Done)

The orchestrator agent is building this. Checklist of must-wires that tests will
exercise:

1. `Orchestrator.ask` fast path with a **fake services bag** (in-process wiring). The
   tests assemble it from real lightweight pieces (`MockEmbedder`,
   `InMemoryVectorStore` from rag-db-handler, `HeuristicReranker`,
   `SimpleTokenizer`, a **stub Generator** + `GenerationService`).
   - Stub Generator: object with `async generate(request: GenerationRequest) -> GenerationResult`
     and `def stream(...)`. `GenerationService` accepts a dict[name->provider-duck]. So pass
     `{"stub": stub_generator}` and `router=None` or a no-op router.
2. Timeout/cancellation becomes `rag_core.errors.OperationTimeout` (RagError subclass) —
   tests do `asyncio.wait_for(ask(...), 0.01)` against a slow fake generator and assert
   `OperationTimeout`. `asyncio.timeout` + try/except `TimeoutError` → raise OperationTimeout.
3. Fallback: primary generator raises `ProviderUnavailableError`, router has a fallback;
   build the test fallback as a second stub that succeeds.
4. Streaming: `ask(..., stream=True)` yields deltas; `collect_stream` reconstructs the answer.
5. The orchestrator must **not import heavy backends at module load** (`fastembed`,
  `qdrant_client`). Heavy component wiring lives in `load_local_services()` which is
  imported lazily / only called in the CLI/FastAPI path. This keeps `import rag_orchestrator`
  cheap (important for mypy and CI).
6. Pipeline YAML: `Pipeline.save_yaml`/`load_yaml` round-trip `PipelineConfig`
  (test with a tmp file).

**Contract review after the agent finishes**: this agent (the main one) will re-read the
six module `__init__.py` files above, diff against what the orchestrator imports, and
file ADRs/fixes for any signature drift.

## 4. Wave 7 — three parallel agents (done)

### 4.1 rag-mass-inject (`packages/rag-mass-inject/`)
- `pyproject.toml` deps: rag-core, rag-doc-handler, rag-ocr, rag-embedder, rag-db-handler.
- Layout: `jobs.py` (`IngestionJob` state, checkpoints via sqlite in a dir),
  `pipeline.py` (`MassIngestor(config, services)`), `sources.py`
  (directory/recursive/sitemap/file-list discovery), `workers.py` (bounded queue +
  per-stage semaphores for read/parse/ocr/chunk/embed/index).
- Contract with orchestrator services: accept an `OrchestratorServices`-like bag OR a
  slim `IngestionPipeline`-equivalent object. Define `MassInjectConfig`
  (`max_workers`, per-stage concurrency, checkpoint_dir, dead_letter_dir, resume).
- API: `submit(source) -> job_id`; `async status(job_id) -> PipelineJob`;
  `async wait(job_id) -> result`. Jobs persist in sqlite and resume from checkpoints.
- Tests: small local corpus in tmp_path; recursive discovery; dedup skips; checkpoint
   resume on mid-run abort; dead-letter on a poison file; all offline.

### 4.2 rag-aio facade + CLI + FastAPI (`packages/rag-aio/`)
- `pyproject.toml` already declares deps on everything; `rag-aio` console script → `rag_aio.cli:app`.
- `src/rag_aio/`:
  - `config.py`: top-level `RAGConfig` (TOML-loadable) composing sub-configs;
    `from_file(path)`.
  - `facade.py`: `RAG.from_config(config) -> RAG`; `async ingest(path)`; `async ask(...)`;
    `stream(...)`. Holds the `Orchestrator`.
  - `cli.py`: Typer app with `ingest`, `ask`, `serve` (uvicorn), `config` subcommands.
  - `app.py`: `FastAPI` reusing `orchestrator.app.app.create_app` (single source of truth).
- Tests: end-to-end ask with a tiny local corpus (fastembed model NOT downloaded — use
  `MockEmbedder` path via config `embedder.mock`? Provide a `mock` fast-path config so the
  smoke test is fully offline).

### 4.3 rag-ui (`packages/rag-ui/`)
- Streamlit console. All non-secret runtime settings shown + editable and validated with
  the same Pydantic config models. Document that secrets are masked.
- Diagnostic pipeline inspector is a documented feature but its full interactive UI is a
  **Wave 8+** stretch; Wave 7 ships the config console + a read-only health/ingestion
  dashboard hitting the FastAPI endpoints.
- Tests: streamlit testing via `st.testing.v1.AppTest` where feasible; otherwise
  assert configuration model serialization.

Wave 7 agents must run **two-at-a-time maximum** (rate-limit history): run mass-inject +
rag-aio facade together, then rag-ui alone, OR all three but expect a possible rate-limit
retry on one. The repo-wide gate runs after all three.

## 5. Wave 8 — hardening, examples, benchmarks, e2e, final review (Done)

1. **Benchmarks** (`benchmarks/`): pytest-based custom harness (NOT pytest-benchmark to
   avoid a heavy dep) reporting mean/median/P50/P90/P95/P99 + cold/warm + RAM. Targets:
   doc throughput, embedding throughput, retrieval latency (dense/sparse/hybrid),
   rerank, context build, generation, full query. Use local LM Studio for generation
   benchmarks (skip if unreachable) and InMemoryVectorStore for retrieval benchmarks.
2. **Examples** (`examples/`): quickstart matching the README `RAG.from_config` snippet,
   per-stage examples, fully-local example, OpenAI-compatible example.
3. **Security**: run a full Mimosa deep scan (`provider: mimosa:mimosa-security-scan`)
   now that real code exists — fix all high/critical, document remaining informational.
4. **E2E smoke**: ingest a real small PDF → ask → answer with citations, over LM Studio.
   Mark `e2e`; auto-skip when `localhost:1234` is unreachable.
5. **Packaging**: `uv build --all-packages`, import smoke test from a fresh venv.
6. **Final reviews**: architecture consistency, API consistency (protocols still match
   implementations), dependency audit (unused/heavy transitive), mypy-strict + ruff +
   `scripts/check.py` full (not just `--fast`) green.
7. **Final report**: per the spec's closing section — modules, boundaries, config model,
   databases/providers supported, benchmark methodology + results, limitations,
   extension points, recommended topologies.

## 6. Acceptance checklist (spec §35/§36)

- [x] `scripts/check.py` fully green (ruff + mypy strict + full pytest, not just --fast).
- [x] `from rag_aio import RAG` quickstart works locally (FastEmbed + sqlite + LM Studio).
- [x] Dense/hybrid retrieval + rerank + context construction exercised.
- [x] ≥1 local or OpenAI-compatible LLM returns an answer with citations.
- [x] Streamlit console edits runtime config; pipeline inspector works.
- [x] Eval suite produces metrics; benchmark suite produces a report.
- [x] Clean-environment reproduction: `uv sync && uv run python scripts/check.py`.
