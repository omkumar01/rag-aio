# Changelog

All notable changes to rag-aio are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Initial build of the rag-aio platform: an asynchronous, modular RAG system delivered as a
uv workspace of 18 packages. Verification at close: full gate green (872 package tests +
6 benchmark-harness unit tests, 91% coverage), `uv build --all-packages` produced all 18
packages with a clean wheel import smoke test, a live e2e run over LM Studio
(ingest → ask → cited answer), and a clean Mimosa security scan. See the
[final report](docs/development/final-report.md) for the full record.

### Added

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
