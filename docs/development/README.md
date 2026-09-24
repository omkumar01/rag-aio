# Development

Developer setup, the quality-gate matrix, and the conventions every change must follow.

## Setup

Requires Python >= 3.12 (CI matrix covers 3.12 and 3.13) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install the 18-package workspace + dev tools from uv.lock
uv run python scripts/check.py           # full gate: format, lint, types, tests
uv run python scripts/check.py --fix     # autofix formatting/lint first
```

Reproduction from a clean environment is exactly `uv sync && uv run python scripts/check.py`.

## The quality gate (`scripts/check.py`)

`scripts/check.py` is the only validation entry point CI calls; it runs checks in a
deterministic order and fails on the first non-zero exit. Flags:

| Flag | Effect |
| --- | --- |
| *(none)* | `ruff format --check .` → `ruff check .` → `mypy packages/*/src` → `pytest` (everything under `packages/`) → `pytest benchmarks -m unit` |
| `--fix` | applies `ruff format .` and `ruff check --fix .` first, then runs the default checks |
| `--fast` | skips the type check **and** deselects integration/e2e tests (`-m "not integration and not e2e"`); format/lint and the benchmark-harness unit pass still run |
| `--unit` | `pytest -m unit` only |
| `--integration` | `pytest -m integration` only |
| `--e2e` | `pytest -m e2e` only |
| `--coverage` | adds `--cov --cov-report=term-missing` to the pytest run (combinable with the mode flags) |
| `--skip-format` | skips the ruff format/lint checks |
| `--skip-types` | skips the mypy check |

Details worth knowing:

- `benchmarks/` is outside pytest's `testpaths`, so the harness unit tests get an explicit
  `pytest benchmarks -m unit` pass. That pass runs in the default, `--fast`, `--unit`, and
  `--coverage` modes, and is skipped in `--integration` / `--e2e` modes.
- The measured benchmark scenarios are never part of the gate; run them explicitly via
  `uv run pytest benchmarks -m benchmark`.
- The mode flags are mutually exclusive in practice: `--unit` takes precedence over
  `--integration` over `--e2e` over `--fast` for the pytest selection.
- As of the wave-8 close ([final-report.md](final-report.md)) the full gate is green:
  872 package tests + 6 benchmark-harness unit tests, 91% coverage.

## Day-to-day commands

```bash
uv run pytest packages/rag-retrieval -q           # one package's tests
uv run pytest packages/rag-aio/tests/test_e2e.py  # single file (e2e tiers auto-skip)
uv run ruff format . && uv run ruff check --fix . # format + autofix everything
uv run python examples/quickstart.py              # offline end-to-end demo
uv run pytest benchmarks -m benchmark             # measure, not verify
```

Workspace notes: the root `pyproject.toml` is not a published package
(`[tool.uv] package = false`); all 18 members live in `packages/*` and are wired as
workspace sources. Add a dependency to a package in that package's own `pyproject.toml`
(heavy/backends as extras), never at the root.

## Lint and types

- **ruff**: `line-length = 100`, `target-version = "py312"`. Rule sets: pycodestyle
  (`E`/`W`), pyflakes (`F`), isort (`I`), pyupgrade (`UP`), bugbear (`B`), simplify
  (`SIM`), comprehensions (`C4`), async hygiene (`ASYNC`), and ruff-specific (`RUF`).
  `E501` is ignored (the formatter owns code lines). Tests ignore `B011`/`SIM117`;
  `scripts/` ignores `T201` (prints are the point there).
- **mypy**: `strict = true` with `warn_unreachable`, run over every
  `packages/*/src` directory. Third-party heavy dependencies (fastembed, qdrant_client,
  streamlit, sqlalchemy, numpy, ...) are declared `ignore_missing_imports` — code inside
  `packages/*/src` is still fully typed.
- Both must stay clean on every change; do not weaken checks to go green
  ([ADR-0008](../../adr/ADR-0008-tdd-and-quality-gates.md)).

## Code layout conventions

- **src layout**: each package keeps code under `packages/<name>/src/<name>/` and tests
  under `packages/<name>/tests/`; every package ships a `py.typed` marker.
- **Layering**: packages are layered (infrastructure → processing → intelligence →
  service → operations → facade); a layer may only depend on `rag-core` contracts and
  lower/equal layers, never on a concrete sibling implementation (ADR-0002).
- **Heavy dependencies go in package extras**; `rag-core` stays dependency-light.
- **Lazy heavy imports**: `fastembed`, `qdrant_client`, Streamlit, and uvicorn are
  imported inside functions (`load_local_services()`, `rag_aio.app.get_app()`, the
  Streamlit `app` attribute), never at module load — import time stays fast and the mock
  profile never touches them.
- **Contracts in `rag-core`**: all cross-module interaction goes through runtime-checkable
  Protocols; boundary data is Pydantic v2, hot-path internals are dataclasses/NumPy.

## TDD and ADRs

Strict TDD per [ADR-0008](../../adr/ADR-0008-tdd-and-quality-gates.md): tests first, and
failure-mode tests (timeouts, retries, cancellation, partial failure, malformed inputs)
are mandatory. See [../testing/README.md](../testing/README.md) for the test strategy.

An ADR is required for decisions affecting public APIs, contracts, service boundaries,
persistence, protocols, or model lifecycle. Accepted ADRs:

| ADR | Decision |
| --- | --- |
| [ADR-0001](../../adr/ADR-0001-uv-workspace-monorepo.md) | uv workspace monorepo layout of independently installable packages |
| [ADR-0002](../../adr/ADR-0002-protocol-based-contracts.md) | Protocol-based contracts with adapter registries instead of inheritance |
| [ADR-0003](../../adr/ADR-0003-qdrant-local-mode-default.md) | Qdrant local mode as the default vector store; server mode when scaling |
| [ADR-0004](../../adr/ADR-0004-lm-studio-default-provider.md) | OpenAI-compatible (LM Studio) as the default local model provider |
| [ADR-0005](../../adr/ADR-0005-fastapi-service-apis.md) | FastAPI + async HTTP for service APIs; no gRPC until benchmarked |
| [ADR-0006](../../adr/ADR-0006-configuration-system.md) | Typed configuration system with precedence and secrets-by-reference |
| [ADR-0007](../../adr/ADR-0007-jobs-for-long-running-operations.md) | Jobs with IDs for long-running operations |
| [ADR-0008](../../adr/ADR-0008-tdd-and-quality-gates.md) | Strict TDD and quality gates |

## Security and releasing

- Security posture, the wave-8 Mimosa deep scan record, and standing mitigations are
  documented in [security.md](security.md). Re-run the scan after dependency bumps.
- The release process (versioning, changelog validation, artifact build, clean-environment
  smoke test, PyPI trusted publishing) is documented in [releasing.md](releasing.md); the
  release workflow runs the gate with `--fast` on version tags before building
  `uv build --all-packages` artifacts.
