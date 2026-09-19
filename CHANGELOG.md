# Changelog

All notable changes to rag-aio are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: `uv` workspace with 17 packages under `packages/`.
- Architecture documentation (`architecture.md`, `design.md`, `docs/`), ADR-0001..0008.
- `scripts/check.py` quality gate (ruff format/lint, mypy strict, pytest).
- GitHub Actions CI and PyPI release (trusted publishing) workflows.
- Optional Qdrant server compose file (`docker-compose.qdrant.yml`).
