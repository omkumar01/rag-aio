# ADR-0001: uv workspace monorepo layout

**Status:** Accepted | **Date:** 2026-09-19

## Context
rag-aio consists of ~17 independently testable, independently deployable modules that share contracts (rag-core). Consumers must be able to install the core alone or any subset of capabilities.

## Decision
One `uv` workspace monorepo with member packages under `packages/`. Each package has its own `pyproject.toml`, extras, tests, and docs. The root workspace provides shared dev tooling (ruff, mypy, pytest) and `scripts/check.py`. A top-level `rag-aio` package provides the `RAG` facade, CLI, and service app that compose the modules.

## Consequences
- Single `uv.lock` for reproducibility; per-package dependency isolation via extras.
- Heavy dependencies (torch, docling, faiss, redis) live in package extras, never in `rag-core`.
- Cross-package changes run through the workspace's single `scripts/check.py` gate.
