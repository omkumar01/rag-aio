# ADR-0007: Jobs with IDs for long-running operations

**Status:** Accepted | **Date:** 2026-09-19

## Context
Mass ingestion, OCR of large files, reindexing, and evaluation runs must not block HTTP requests, and the UI must show progress cheaply.

## Decision
Long-running operations become jobs: the API returns a job ID immediately and clients poll or subscribe to structured status (queued/running/completed/failed/cancelled, stage progress, error taxonomy). Job state persists in the configured SQL store with checkpoints so mass ingestion is resumable. In-process usage may await jobs directly.

## Consequences
- HTTP timeouts never truncate work; cancellation is cooperative and propagated.
- A message broker can later replace the in-process job queue without changing business logic (queue is behind an interface).
