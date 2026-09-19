# Testing Strategy

Markers: `unit` (default, no external services), `integration` (real local backends:
Qdrant local mode, sqlite, filesystem), `e2e` (full pipeline against local LM Studio;
auto-skips when the endpoint is unreachable).

- Unit tests mock all providers; fast and deterministic; no network.
- Integration tests use ephemeral local state (tmp dirs, sqlite, Qdrant local mode with
  in-memory or temp-path persistence).
- E2E smoke test ingests a small fixture corpus and answers a question with citations
  through the running LM Studio endpoint.
- Property-based tests (hypothesis) cover parsers, chunkers, ID/hash generation,
  serialization round-trips, and configuration parsing.
- Failure-mode tests are mandatory: timeouts, retries, cancellation, partial failure,
  malformed files, corrupt PDFs, provider outage, concurrent ingestion/retrieval.

Run:

```bash
uv run python scripts/check.py --unit
uv run python scripts/check.py --integration
uv run python scripts/check.py --e2e
```
