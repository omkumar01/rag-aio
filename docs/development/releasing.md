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
   (OIDC)** — no long-lived API tokens.
7. The release workflow runs only on version tags, never on normal CI runs.

## Checklist

- [ ] Changelog updated
- [ ] Versions bumped consistently
- [ ] Wheels + sdist built and smoke-tested clean
- [ ] CI green on the release commit
