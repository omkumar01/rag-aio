# ADR-0008: Strict TDD and quality gates

**Status:** Accepted | **Date:** 2026-09-19

## Context
The platform must remain correct under asynchronous concurrency and across many adapters; retrofitting tests onto async code is expensive.

## Decision
Every feature is implemented test-first: failing test, minimal implementation, refactor. `scripts/check.py` (ruff format, ruff lint, mypy strict, pytest) must pass at every wave boundary and is the only entry point CI calls. Unit tests mock external providers; integration tests use ephemeral local backends; e2e tests require local services and skip when unavailable. Property-based tests cover parsers, chunkers, ID/hash generation, serialization, and config parsing.

## Consequences
- No check is ever disabled or weakened to go green; root causes are fixed.
- Failing to keep the gate green blocks the wave, not the checklist.
