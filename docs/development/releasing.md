# Releasing

## Versioning

Semantic versioning per package. The workspace root is not published. Cross-package
contract changes are released in lockstep: `rag-core` first, then dependents.

## Process

1. Ensure `scripts/check.py` passes fully (including integration tests).
2. Update `CHANGELOG.md` entries for every changed package.
3. Bump versions (`uv run python scripts/bump_version.py <version>` when introduced).
4. Build artifacts: `uv build --all-packages`.
5. Verify artifacts from a clean environment (fresh `uv venv`, install wheels, import
   smoke test).
6. Tag and push; the `release.yml` workflow publishes to PyPI via **trusted publishing
   (OIDC)** — no long-lived API tokens. The workflow can also be run manually from the
   GitHub Actions tab (`workflow_dispatch`) — e.g. to re-publish after a build-job fix —
   but normal releases are tag-driven.
7. The release workflow runs only on version tags (or manual dispatch), never on normal
   CI runs.

## Install surface

The `rag-aio` facade is the PyPI umbrella package. Its base install is
dependency-light (`rag-aio-core` + CLI/service deps); every workspace package is
exposed as a pip extra of the facade (extras use the import-style name, e.g.
`rag-ocr`) plus an `all` extra:

```bash
pip install rag-aio             # base only
pip install "rag-aio[all]"      # all 16 backend packages
pip install "rag-aio[rag-ocr]"  # single package + its own dependencies
```

**Distribution naming:** sub-packages publish as `rag-aio-<name>`
(`rag-aio-core`, `rag-aio-ocr`, `rag-aio-orchestrator`, …) — the generic
`rag-retrieval`, `rag-orchestrator`, and `rag-eval` names were already taken on
PyPI by unrelated projects. Import names (`rag_core`, …) and directory names
under `packages/` are unchanged; only the `[project.name]` in each pyproject
carries the prefix. Do not drop the prefix for new packages.

**Contract:** every package under `packages/` except the facade itself and
`rag-ui` must have a matching extra in `packages/rag-aio/pyproject.toml` (and be
listed in `all`). The release workflow validates this before publishing and
fails the build otherwise — when adding a new package, add its extra to the
facade in the same change. `rag-ui` is intentionally not part of `all` (it
depends on `rag-aio` and pulls Streamlit); users install it standalone as
`rag-aio-ui`.

## Checklist

- [ ] Changelog updated
- [ ] Versions bumped consistently
- [ ] Wheels + sdist built and smoke-tested clean
- [ ] CI green on the release commit
