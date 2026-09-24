# Testing Strategy

Every package carries its own test suite under `packages/<name>/tests/`; pytest collects
them through the workspace (`testpaths = ["packages"]`). Four markers are registered in
the root `pyproject.toml`:

- `unit` (default, no external services) — mocks all providers; fast and deterministic;
  no network.
- `integration` (real local backends) — uses ephemeral local state (tmp dirs, sqlite,
  Qdrant local mode with in-memory or temp-path persistence).
- `e2e` (full pipeline; may require local LM Studio/Qdrant) — Tier 1 runs fully offline
  against the mock profile (real parse → chunk → embed → retrieve → rerank → context →
  generate chain with a stub generator); Tier 2 targets a local LM Studio server and
  ingests a small fixture corpus, then answers a question with citations.
- `benchmark` (performance scenarios in `benchmarks/`) — never run by default; executed
  explicitly via `pytest benchmarks -m benchmark` (see
  [benchmarks/README.md](../../benchmarks/README.md) and
  [performance](../performance/README.md)).

## Running the modes

Through the quality gate (preferred — same entry point CI calls):

```bash
uv run python scripts/check.py --unit         # unit tests only (-m unit)
uv run python scripts/check.py --integration  # integration tests only (-m integration)
uv run python scripts/check.py --e2e          # e2e tests only (-m e2e)
uv run python scripts/check.py --fast         # -m "not integration and not e2e" (skips mypy too)
uv run python scripts/check.py                # full: everything collected in packages/
uv run python scripts/check.py --coverage     # any test mode with --cov term-missing report
```

Raw pytest invocations:

```bash
uv run pytest                          # all tests under packages/
uv run pytest -m unit                  # unit only
uv run pytest -m integration           # integration only
uv run pytest -m e2e                   # e2e only (LM Studio tiers auto-skip)
uv run pytest benchmarks -m benchmark  # the benchmark suite (explicit only)
uv run pytest benchmarks -m unit       # benchmark harness math unit tests (CI-safe)
```

`benchmarks/` sits outside `testpaths`, so the harness unit tests get an explicit
`pytest benchmarks -m unit` pass in the gate and CI; that pass is skipped in the
marker-specific `--integration` / `--e2e` gate modes.

## CI behaviour

`.github/workflows/ci.yml` runs on pushes to `main` and all pull requests, on a matrix of
ubuntu-latest + windows-latest and Python 3.12 + 3.13:

1. `uv sync --frozen`
2. `ruff format --check .` and `ruff check .`
3. `mypy packages/*/src`
4. `pytest -m "not integration and not e2e" --cov` — **integration and e2e are always
   deselected in CI**; they require local services (Qdrant, LM Studio) that CI does not
   provide.
5. `pytest benchmarks -m unit` — the benchmark harness math tests only; the measured
   scenarios (`-m benchmark`) never run in CI.
6. Coverage total is published as a commit status (`coverage <n>%`) on pushes to `main`
   (ubuntu / Python 3.12 leg only).

The release workflow (`.github/workflows/release.yml`) runs `scripts/check.py --fast`
before building artifacts on version tags.

## E2E auto-skip contract

E2E Tier 2 tests probe `http://localhost:1234/v1` before running:

- Unreachable endpoint → `pytest.skip("LM Studio not reachable at localhost:1234")`.
- Endpoint up but no served model can generate → skipped as well.

Tier 1 (mock profile) always runs offline. The same probe-and-skip pattern is used by the
generation benchmark scenario and the endpoint-based examples, so no command in this
repository ever hangs on a missing local model server.

## Fixtures and fakes conventions

- Shared offline fakes live in `packages/rag-aio/tests/aio_fakes.py` (stub query engine,
  retriever, generator, etc. wired into an `OrchestratorServices` bag), mirroring
  `packages/rag-orchestrator/tests/fakes.py`. Package-level `conftest.py` files provide
  event-loop and temp-path fixtures (`asyncio_mode = "auto"`).
- HTTP boundaries are tested with `httpx.MockTransport` (e.g. the `rag-ui` `Dashboard`
  suite) — no real server is ever started in unit tests.
- Streamlit tabs are exercised with the Streamlit `AppTest` harness.
- Mock embeddings are deterministic and hash-based, so retrieval assertions are stable
  without semantic models.

## Failure-mode coverage

Failure-mode tests are mandatory (design.md): timeouts, retries, cancellation, partial
failure, malformed files, corrupt PDFs, provider outage, and concurrent
ingestion/retrieval. These are covered in-suite — e.g. `rag-generation` tests exercise
retry/fallback and provider errors, `rag-ocr` tests exercise malformed/corrupt inputs,
and orchestrator tests cover timeout (`OperationTimeout`) and cancellation paths.

## Property-based testing

Not yet implemented. `hypothesis` is not currently a dependency. Property-based testing
of parsers, chunkers, ID/hash generation, serialization round-trips, and configuration
parsing is a planned extension; the strict type checks, the mock-profile determinism, and
the failure-mode suites are the current defence.
