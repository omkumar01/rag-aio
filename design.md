# rag-aio Design Principles

This document records implementation-level design principles that apply across modules.
Architecture and boundaries live in [architecture.md](architecture.md); individual decisions
in [adr/](adr/).

## Contracts first

- `rag-core` defines runtime-checkable `typing.Protocol` interfaces; implementations depend
  on abstractions, never on vendors or concrete siblings (ADR-0002).
- Boundary data is Pydantic v2; hot-path internals use dataclasses, tuples, and NumPy.
  Avoid Pydantic model construction inside tight loops where profiling shows it matters.
- Public APIs stay small and stable; provider-specific features live behind optional
  extension interfaces and never leak into core models.

## Async and concurrency discipline

- `async def` for all I/O paths. CPU/GPU inference never blocks the loop: `asyncio.to_thread`
  or process pools, or a dedicated service when the model deserves its own process.
- `asyncio.TaskGroup` for structured concurrency; deadlines propagate; client disconnects
  cancel downstream work; partial results are returned with explicit status.
- Bounded everything: semaphores around remote calls, bounded queues between pipeline
  stages, connection pooling for HTTP and databases.
- Batch at every expensive boundary: embeddings, reranking, DB writes, remote requests.

## Performance

- Measure before optimizing; every micro-optimization is benchmark-backed or isolated.
- Prefer ONNX-optimized runtimes (FastEmbed) for embeddings; keep models warm; never reload
  per request.
- Avoid large copies: pass references/arrays internally, serialize only at true boundaries.
- Incremental work: unchanged documents are not reparsed/re-embedded (content + config
  hashes drive reindexing).

## Persistence

- Capability-specific store interfaces (`SQLStore`, `DocumentStore`, `KeyValueStore`,
  `VectorStore`, `SparseStore`, `CacheStore`, `SearchStore`) instead of one false universal
  repository. Hybrid persistence is normal: Postgres metadata + file/object sources +
  Qdrant vectors + Redis cache.
- Transactions and consistency semantics are explicit per store; bulk operations everywhere.

## Security posture

- Every document, page, email, and chunk is untrusted: isolate retrieved content from
  system instructions, sanitize HTML, guard against SSRF in crawling, enforce upload/
  size/page/token limits, defend against decompression bombs.
- Secrets are references, never values, outside a narrowly scoped credential layer; logs,
  traces, UI, and API responses redact them mechanically.
- Fail closed on unsafe inputs; record structured errors instead of swallowing exceptions.

## Testing

- Strict TDD (ADR-0008): failing test first, minimal implementation, refactor.
- Unit tests mock providers; integration tests use ephemeral local backends (Qdrant local
  mode, sqlite); e2e tests run against local LM Studio and skip when unreachable.
- Property-based tests for parsers, chunkers, IDs/hashes, serialization, and config.
- Test failure modes deliberately: timeouts, retries, cancellation, partial failure,
  malformed input, provider outage, concurrent ingestion/retrieval.

## Documentation

- Every module documents purpose, public API, configuration, dependencies, endpoints,
  data flow, performance characteristics, failure modes, extension points, examples,
  testing strategy, and operations.
- Trade-offs are recorded where the choice is not obvious (e.g., why FastEmbed is the
  default embedder, why reranking is restricted to small candidate sets, why LangChain/
  LlamaIndex are not mandatory dependencies — they may exist only as optional adapters
  when they add concrete value).
