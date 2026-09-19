# Configuration

The configuration model, precedence rules, secret handling, and versioning are defined in
[ADR-0006](../../adr/ADR-0006-configuration-system.md).

- Precedence (lowest → highest): built-in defaults → config file (TOML/YAML) →
  environment (`RAG_AIO_` prefix) → persisted UI configuration → per-request override.
- Secrets are references (env var names / files), never values; inspection endpoints,
  logs, traces, and the UI redact them.
- Every execution records a non-secret configuration hash for reproducibility.

The full reference of settings per module is generated into this section as modules are
implemented.
