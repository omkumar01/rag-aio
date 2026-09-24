# Architecture — as-built deep dive

The canonical design doc is the root [architecture.md](../../architecture.md); individual
decisions live in [adr/](../../adr/). This page is the as-built view: the actual layering,
the ingestion and query paths with real function and contract names, the cross-package
contract matrix, observability/caching hooks, and the job lifecycle for long-running
operations. It reflects the wave-8 closing state recorded in
[docs/development/final-report.md](../development/final-report.md).

rag-aio is a uv workspace of 18 packages (ADR-0001). Each layer depends on contracts from
`rag-core` and on lower or equal layers — never on a concrete sibling implementation
(ADR-0002). Boundary data is Pydantic v2; hot-path internals are dataclasses and NumPy.

## Layers

```
                ┌────────────────────────────────────────────────────────────┐
      Facade    │  rag-aio  (RAG facade · Typer CLI · FastAPI composition)   │
                └─────────────────────────────┬──────────────────────────────┘
                ┌─────────────────────────────┴──────────────────────────────┐
    Operations  │  rag-eval (metrics, datasets, run compare)                 │
                │  rag-ui   (Streamlit console → service APIs)               │
                └─────────────────────────────┬──────────────────────────────┘
                ┌─────────────────────────────┴──────────────────────────────┐
      Service   │  rag-orchestrator (pipelines, budgets, services bag,       │
                │  structured concurrency, FastAPI app factory)              │
                └─────────────────────────────┬──────────────────────────────┘
                ┌─────────────────────────────┴──────────────────────────────┐
  Intelligence  │  query path:  rag-query · rag-retrieval · rag-rerank       │
                │               rag-context · rag-generation                 │
                │  control:     rag-llm-provider (registry, routing, aliases)│
                └─────────────────────────────┬──────────────────────────────┘
                ┌─────────────────────────────┴──────────────────────────────┐
   Processing   │  ingestion path:  rag-doc-handler · rag-ocr                │
                │                   rag-embedder · rag-mass-inject           │
                └─────────────────────────────┬──────────────────────────────┘
                ┌─────────────────────────────┴──────────────────────────────┐
Infrastructure  │  rag-core (models, Protocols, errors, IDs/hashes)          │
                │  rag-observe · rag-cache · rag-db-handler                  │
                └────────────────────────────────────────────────────────────┘
```

Every module with expensive state (doc-handler, ocr, embedder, retrieval, rerank,
generation) additionally exposes an identical FastAPI service API with `/health`, `/ready`,
`/metrics`, `/version` (ADR-0005); long operations run as jobs with IDs (ADR-0007).

## Ingestion path

Single documents flow through `rag_doc_handler.IngestionPipeline` and
`rag_embedder.EmbeddingPipeline`; bulk work goes through `rag_mass_inject.MassIngestor`.
All persistence happens through `rag-db-handler` contracts (ADR-0003: Qdrant local mode is
the default vector store).

1. **Load** — `SourceLoader` fetches bytes (local paths, web with SSRF guards, sitemap/site
   crawl) plus basic metadata.
2. **Detect** — `detect_extension` / `guess_mime` / `sniff_mime` pick the type; `auto_detect`
   can be tuned via `IngestionPipelineConfig`.
3. **Parse** — the `ParserRegistry` selects a parser implementing the rag-core
   `DocumentParser` protocol (PDF, HTML, Office, email, Markdown, …) and produces a
   canonical `Document`.
4. **OCR fallback** — the PDF parser accepts an optional `OCRFallback`; pages whose
   extracted text is too short are routed to `rag-ocr` (`OCRPipeline`: RapidOCR-class
   mechanical engine, escalating to a VLM semantic extractor via providers), returning
   provenance-preserving `OCRRegion`s that become `PageBlock`s.
5. **Dedup** — `DedupIndex` keys on `rag_core.ids.content_hash`; `ingest()` returns
   `(document, is_new=False)` for a repeated document and the index is left untouched.
   Incremental reindexing: `EmbeddingPipeline.should_reindex(document, config_hash, seen)`
   skips documents whose `content_hash:config_hash` pair is unchanged.
6. **Chunk** — a rag-core `Chunker` (recursive chunker by default) splits the document,
   preserving provenance into `Chunk`/`ChunkMetadata`.
7. **Embed + index** — `IngestionIndexer.index_document(document, chunks, embedder,
   sparse_embedder, store)` batches dense `Embedder.embed` (FastEmbed `BAAI/bge-small-en-v1.5`
   by default) and sparse `SparseEmbedder.embed_sparse` (BM25-style) and upserts into the
   `VectorStore`. `PipelineOutcome` reports chunk counts, dims, model names and timing.
8. **Bulk variant** — `MassIngestor` discovers sources (directory, file list, sitemap),
   streams them through bounded queues (`BoundedQueue`, `StageWorker` per-stage concurrency),
   and tracks everything in a `JobTracker` (see [Job lifecycle](#job-lifecycle)).

## Query path

`rag_orchestrator.Orchestrator.ask()` sequences the components wired in the
`OrchestratorServices` bag. Every stage is timed and the whole call runs under
`asyncio.timeout(pipeline_config.timeout_s)`, surfacing as `rag_core.OperationTimeout`.

1. **Query intelligence** — `QueryEngine.process(Query)` normalizes, classifies, and runs
   enabled `QueryStrategy`s (rewrite, expansion, decomposition, HyDE) concurrently in an
   `asyncio.TaskGroup`. The result (`QueryResult`) always contains a `kind="original"`
   variant; variants are de-duplicated case-insensitively and capped at
   `QueryConfig.max_variants`. With default flags no LLM call is made.
2. **Parallel retrieval** — `ParallelRetrievalExecutor(retrievers, max_concurrency)` runs the
   retrievers (`DenseRetriever`, `BM25Retriever`, or a `HybridRetriever`) over all variants
   at `top_k * 2` breadth for rerank/context headroom.
3. **Fusion** — `services.fusion.fuse(results, top_k, dedup=True)` merges result lists
   (`ReciprocalRankFusion` or `WeightedScoreFusion`), capped at the rerank candidate limit
   (default 50).
4. **Rerank** — `services.reranker.rerank(query, candidates, top_k)` reorders the small
   candidate set (`HeuristicReranker`, `CrossEncoderReranker`, or `RemoteReranker`). A
   `RerankError` degrades gracefully to retrieval order (recorded as `rerank.unavailable`).
5. **Context build** — `ContextBuilder.build(query, hits, token_budget)` selects evidence
   under the token budget (default 2048, with `reserve_for_answer=64`), deduplicates, and
   maps citations. `build_citations(context)` and `render_context(context)` prepare the
   prompt payload.
6. **Generation** — a `GenerationRequest` (fixed `SYSTEM_PROMPT` requiring numeric `[n]`
   citations) goes to `services.generator`. Transient failures
   (`ProviderUnavailableError`, `RateLimitError`) walk `services.fallback_generators`;
   non-transient `ProviderError`s are re-raised. `stream=True` yields text deltas instead.
7. **Result** — `AskResult` carries `answer`, `citations`, the `context` (for inspection),
   `timings_ms` (`query`, `retrieve`, `rerank`, `context`, `generate`, `total_ms`),
   `query_id`, and `metrics` (token usage plus `cached` and `correlation_id`).

## Contract matrix

All cross-package capabilities are runtime-checkable `typing.Protocol`s owned by
`rag-core` (ADR-0002) — see
[packages/rag-core/src/rag_core/protocols.py](../../packages/rag-core/src/rag_core/protocols.py).
The hot ones:

| Protocol | Key method(s) | Implemented by |
| --- | --- | --- |
| `Retriever` | `retrieve(query) -> RetrievalResult` | `DenseRetriever`, `BM25Retriever`/`SparseRetriever`, `HybridRetriever` |
| `FusionStrategy` | `fuse(results, top_k)` | `ReciprocalRankFusion`, `WeightedScoreFusion` |
| `Reranker` | `rerank(query, candidates, top_k)` | `HeuristicReranker`, `CrossEncoderReranker`, `RemoteReranker` (via `RerankPipeline`) |
| `ContextBuilder` | `build(query, hits, token_budget) -> Context` | `rag_context.ContextBuilderImpl` |
| `Generator` | `generate(request)`, `stream(request)` | provider adapters behind `GenerationService` (`_ServiceGenerator` adapts it) |
| `LLMProvider` | `complete(request)` | `OpenAICompatibleProvider`, `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider` |
| `Embedder` / `SparseEmbedder` | `embed` / `embed_sparse` | `FastEmbedDense`, `FastEmbedSparse`, `MockEmbedder` |
| `VectorStore` | `upsert`, `search`, `delete`, `health` | `InMemoryVectorStore`, `QdrantVectorStore` (`create_vector_store`) |
| `DocumentParser` / `DocumentLoader` | `parse`, `load` | `rag-doc-handler` plugin registry |
| `OCRProcessor` | `process_page` | `rag-ocr` (`OCRPipeline`, `OCRRouter`) |
| `Chunker` / `Tokenizer` | `chunk`, `count_tokens` | recursive chunker, `SimpleTokenizer` |
| `QueryStrategy` | `transform(query) -> list[QueryVariant]` | rewrite, expansion, decompose, HyDE strategies |
| `Observer` | `record(event, attributes)` | `NoOpObserver`, `rag-observe` `ObservabilityHub` |
| `Cache` | `get`, `set`, `delete` | `rag-cache` backends |
| `EvaluationMetric` | `compute(ground_truth, predictions, k)` | `rag-eval` metric registry |

`OrchestratorServices` is the composition point: a typed, behavior-free bag of
already-constructed components (query engine, retrievers, fusion, reranker, context
builder, generator + fallbacks, ingestion/embedding pipelines, stores, cache, observer).
`load_local_services()` wires the default profile — heavy backends (`fastembed`,
`qdrant_client`) are imported lazily there, never at module load.

## Observability and caching

- **Observability** — `rag-core` owns the `Observer` protocol; `rag-observe` is the
  OpenTelemetry-API implementation (no SDK required; unconfigured spans are no-ops).
  Hooks: `observe` (span context), `timed` / `StageTimer` (stage duration),
  `record_counter` / `record_histogram` (metrics), plus redacted structured logging
  (`JSONFormatter`, `RedactingFilter`, `REDACTED`) and correlation IDs via
  `set_correlation_id` (set per `ask()` call). The orchestrator records events such as
  `ask.start`, `ask.end`, `rerank.unavailable`, and `generate.fallback`.
- **Caching** — `rag_core.ids.config_hash` canonicalizes values, and
  `rag_cache.keys.cache_key(namespace, **parts)` builds deterministic
  `namespace:<sha256>` keys, so keys embed content/config hashes by construction.
  `NamespacedCache` prefixes keys so stages share one backend; `InstrumentedCache` wraps
  any cache with `CacheStats` (hits/misses/sets/deletes/evictions/errors, `hit_rate`).
  Backends are memory, disk, and sqlite via `create_cache(CacheConfig)` (redis behind an
  extra). Caching is opt-in per stage; the local default profile uses an
  `InstrumentedCache(MemoryCache(...))`.

## Job lifecycle

Long-running operations become jobs with IDs instead of blocking HTTP requests (ADR-0007).
`rag_core.PipelineJob` is the status record: `job_id`, `kind`, `status`
(`queued → running → completed | failed | cancelled`), `progress`, `stage`,
`total_items`/`processed_items`, `error`/`error_code`, timestamps, and a `checkpoints` map.

`rag-mass-inject` implements the pattern end to end:

- `MassIngestor.submit(source)` synchronously creates a queued job and returns its ID
  immediately; work proceeds asynchronously.
- `JobTracker` persists job rows, per-file checkpoints, and dead-letter records in a
  SQLite `jobs.db`. Checkpoints are content-hash based, so resume skips files already
  ingested with the same config.
- `BoundedQueue` + `StageWorker` give backpressure and per-stage concurrency; permanently
  failing files land in the dead-letter table instead of stalling the job.
- In-process callers may await jobs directly; HTTP clients poll or subscribe to structured
  status. Cancellation is cooperative and propagated.

## Key decisions

- [ADR-0001](../../adr/ADR-0001-uv-workspace-monorepo.md) — uv workspace monorepo; heavy
  deps live in package extras, never in `rag-core`.
- [ADR-0002](../../adr/ADR-0002-protocol-based-contracts.md) — protocol contracts with
  adapter registries; vendor types never cross a boundary.
- [ADR-0003](../../adr/ADR-0003-qdrant-local-mode-default.md) — Qdrant local mode default
  (`./data/qdrant`), server mode is a config change.
- [ADR-0004](../../adr/ADR-0004-lm-studio-default-provider.md) — OpenAI-compatible provider
  defaulting to LM Studio at `http://localhost:1234/v1`.
- [ADR-0005](../../adr/ADR-0005-fastapi-service-apis.md) — FastAPI/HTTP+JSON only for
  service APIs; no gRPC until profiling demands it.
- [ADR-0006](../../adr/ADR-0006-configuration-system.md) — one typed Pydantic config
  hierarchy, documented precedence, secrets by reference.
- [ADR-0007](../../adr/ADR-0007-jobs-for-long-running-operations.md) — jobs with IDs and
  persisted checkpoints.
- [ADR-0008](../../adr/ADR-0008-tdd-and-quality-gates.md) — strict TDD behind
  `scripts/check.py`.
