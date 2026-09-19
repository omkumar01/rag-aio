# ADR-0006: Typed configuration system and precedence

**Status:** Accepted | **Date:** 2026-09-19

## Context
Every module needs typed, validated, overridable configuration; secrets must never leak through inspection endpoints, logs, or the UI.

## Decision
A single versioned Pydantic configuration hierarchy shared by all modules. Precedence (lowest to highest): built-in defaults -> config file (TOML/YAML) -> environment variables (`RAG_AIO_` prefix) -> persisted UI configuration -> per-request override. Secrets are stored by reference (env var name / secret file / OS keyring handle) and are redacted from every API/UI/log surface. Config models carry `schema_version` and migration hooks; every execution records a non-secret config hash for reproducibility.

## Consequences
- All RAG-affecting settings are explicit and inspectable.
- Adding a setting requires a model change, a default, and documentation - no hidden magic values in code.
