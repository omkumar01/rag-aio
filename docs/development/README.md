# Development

## Setup

```bash
uv sync                                  # install workspace + dev tools from lockfile
uv run python scripts/check.py           # full gate: format, lint, types, tests
uv run python scripts/check.py --fix     # autofix formatting/lint
uv run python scripts/check.py --fast    # skip integration/e2e
```

## Conventions

- Strict TDD ([ADR-0008](../../adr/ADR-0008-tdd-and-quality-gates.md)): tests first.
- `ruff` + `mypy --strict` clean on every change; `scripts/check.py` is the only entry
  point CI calls.
- Heavy dependencies go in package extras; `rag-core` stays dependency-light.
- ADR required for decisions affecting public APIs, contracts, service boundaries,
  persistence, protocols, or model lifecycle.

## Releasing

The release process (versioning, changelog validation, artifact build, smoke test,
PyPI trusted publishing) is documented in [releasing.md](releasing.md).
