# rag-aio System Architecture

## Overview

rag-aio separates a **control plane** (configuration, strategies, models, pipelines, jobs,
tenants, health, UI state) from a **data plane** (ingestion, OCR, embedding, retrieval,
reranking, context construction, generation). Modules are library packages that can run
in-process for latency or as FastAPI services for scale; every module that can be remote
exposes identical Python and HTTP APIs.

## Logical architecture

```
                ┌──────────────────────────────────────────────────┐
                │                  Control plane                   │
                │   rag-orchestrator · rag-llm-provider · jobs ·   │
                │   configuration · strategy routing · health      │
                └───────────────┬──────────────────────────────────┘
                                │
  UI/API ──────────────► Orchestrator ──────────────► Generation
  (rag-ui / FastAPI)         │                          (rag-generation)
                ┌────────────┼───────────────┐
                ▼            ▼               ▼
          rag-query    rag-retrieval    rag-context
                        ▲        ▲             │
                  dense │        │ sparse      ▼
                        └──► rag-rerank    Generation
                                ▲
                ┌───────────────┴────────────────┐
                │            Data plane          │
                │  rag-mass-inject → rag-doc-    │
                │  handler → rag-ocr → rag-      │
                │  embedder → rag-db-handler     │
                └────────────────────────────────┘

  Shared infrastructure: rag-core (contracts) · rag-observe · rag-cache
```

## Module responsibilities

| Module | Owns | Delegates |
| --- | --- | --- |
| `rag-core` | Canonical models, Protocols, errors, IDs/hashes | — |
| `rag-doc-handler` | Format detection, parsing, web crawling, canonical `Document` output | OCR to `rag-ocr` |
| `rag-ocr` | Mechanical OCR + semantic extraction escalation, provenance regions | VLM calls to providers |
| `rag-embedder` | Chunking, tokenization, embedding, indexing orchestration | Storage to `rag-db-handler` |
| `rag-db-handler` | SQL/document/KV/vector/sparse/cache store abstractions + adapters | — |
| `rag-retrieval` | Dense/sparse/hybrid execution, fusion, filtering | Vector stores via `rag-db-handler` contracts |
| `rag-rerank` | Second-stage ranking on small candidate sets | Models via providers |
| `rag-query` | Query intelligence, parallel strategy execution | Rewrites to `rag-generation` |
| `rag-context` | Token-budgeted evidence selection, citation mapping | Tokenizer via `rag-embedder` contracts |
| `rag-generation` | Provider-agnostic generation, streaming, structured output | Providers via `rag-llm-provider` |
| `rag-llm-provider` | Model registry, capabilities, aliases, fallbacks | — |
| `rag-orchestrator` | Pipeline definitions, strategy composition, budgets, cancellation | Everything else via contracts |
| `rag-mass-inject` | Bulk ingestion jobs, backpressure, checkpoints | Pipeline modules |
| `rag-eval` | Metrics, datasets, reproducible runs | — |
| `rag-cache` | Multi-level caching with config-hash keys | — |
| `rag-observe` | Spans, metrics, redacted logs | OTel backends (pluggable) |
| `rag-ui` | Streamlit console, pipeline inspector | Service APIs |
| `rag-aio` | Facade, CLI, service composition | All modules |

## Execution model

- **Async-first**: all I/O is async; inference (OCR, embeddings, reranking) runs off the
  event loop via thread/process pools.
- **Structured concurrency**: `asyncio.TaskGroup` with deadline propagation, cancellation,
  and bounded concurrency (semaphores, bounded queues) — never unbounded task creation.
- **Parallel strategies**: dense/sparse/metadata retrieval branches run concurrently and
  are fused (RRF or weighted).
- **Batching**: embeddings, reranking, database writes, and remote calls are batched.

## Service boundaries

Every module has a local implementation; modules with expensive state or models
(doc-handler, ocr, embedder, retrieval, rerank, generation) additionally expose FastAPI
service APIs with `/health`, `/ready`, `/metrics`, `/version`, typed domain endpoints,
correlation IDs, and job endpoints for long operations (see ADR-0005 and ADR-0007).

## Default profile (local high-performance)

PyMuPDF fast PDF → token-aware chunking → FastEmbed dense + BM25 sparse → Qdrant local
mode → hybrid RRF fusion → CrossEncoder-style rerank on ≤50 candidates → token-budgeted
context → OpenAI-compatible generation (LM Studio). Every element is replaceable through
configuration and dependency injection (ADR-0003, ADR-0004).

## Cross-cutting concerns

- **Observability** (`rag-observe`): OpenTelemetry API spans/metrics per stage, redaction
  by default, pluggable exporters.
- **Caching** (`rag-cache`): per-stage opt-in, keys include model/config/content hashes.
- **Security**: documents are untrusted (injection isolation, SSRF guards, size limits);
  secrets are by-reference and redacted everywhere.
- **Configuration** (ADR-0006): one versioned Pydantic hierarchy, documented precedence.

## Documented trade-offs

Key choices and their rationale are recorded as ADRs in [adr/](adr/): workspace layout
(0001), protocol contracts (0002), Qdrant local mode (0003), LM Studio default provider
(0004), FastAPI-only service APIs (0005), configuration precedence (0006), job model
(0007), TDD gates (0008).
