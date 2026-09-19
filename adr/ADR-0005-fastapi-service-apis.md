# ADR-0005: FastAPI + async HTTP for service APIs; no gRPC until benchmarked

**Status:** Accepted | **Date:** 2026-09-19

## Context
Modules can run in-process (library) or as services. Internal communication needs to be fast in Python and simple to operate.

## Decision
Every remotely deployable module exposes the same operations via a Python API and a FastAPI service API (HTTP/JSON + SSE streaming). gRPC or another binary protocol may be introduced later only for an internal high-throughput path that profiling proves is bottlenecked by JSON serialization.

## Consequences
- One communication stack to operate; OpenAPI documentation for free.
- Internal high-volume paths (bulk embedding/indexing) minimize copies and use batching to keep serialization costs negligible.
