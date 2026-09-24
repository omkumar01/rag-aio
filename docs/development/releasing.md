# Releasing

## Versioning

Semantic versioning per package. The workspace root is not published. Cross-package
contract changes are released in lockstep: `rag-core` first, then dependents.

## Process

1. Ensure `scripts/check.py` passes fully (including integration tests).
2. Update `CHANGELOG.md` entries for every changed package.
3. Bump versions (`uv run python scripts/bump_version.py <version>` when introduced).
4. Build artifacts: `uv build --package rag-aio --wheel` (single distribution,
   wheel only — see [Install surface](#install-surface)).
5. Verify artifacts from a clean environment (fresh `uv venv`, install wheels, import
   smoke test).
6. Tag and push; the `release.yml` workflow publishes to PyPI via **trusted publishing
   (OIDC)** — no long-lived API tokens. The workflow can also be run manually from the
   GitHub Actions tab (`workflow_dispatch`) — e.g. to re-publish after a build-job fix —
   but normal releases are tag-driven.
7. The release workflow runs only on version tags (or manual dispatch), never on normal
   CI runs.

## Install surface

Everything publishes under a **single PyPI project: `rag-aio`**. One wheel
bundles all 18 workspace packages' code (`rag_aio`, `rag_core`, … `rag_ui`).

```bash
pip install rag-aio                     # all packages, mock/light backends
pip install "rag-aio[all]"              # + every optional backend
pip install "rag-aio[qdrant,fastembed]" # pick individual backends
```

**Distribution shape:**

- Base dependencies are the union of the members' light third-party deps
  (pydantic, fastapi, numpy, tokenizers, sqlalchemy, pymupdf, …). Heavy
  backend SDKs are extras keyed by backend name: `qdrant`, `fastembed`,
  `sentence-transformers`, `faiss`, `postgres`, `pgvector`, `redis`,
  `docling`, `msg`, `rtf`, `pydantic-ai`, `streamlit`, plus `all`.
- **Wheel-only.** Build with `uv build --package rag-aio --wheel`. The build
  hook (`packages/rag-aio/build_hook.py`) copies the sibling members' sources
  into the wheel; an sdist build cannot resolve those sibling paths, so no
  sdist is published.
- When adding a new workspace package, its code is bundled automatically on
  the next build; the release workflow fails if the wheel is missing any
  `packages/` module. Decide explicitly whether its third-party deps belong
  in the facade's base dependencies or in an extra, and keep heavy imports
  lazy so the base install stays functional offline.

## Checklist

- [ ] Changelog updated
- [ ] Versions bumped consistently
- [ ] Wheels + sdist built and smoke-tested clean
- [ ] CI green on the release commit
