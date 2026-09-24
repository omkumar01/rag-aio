# Security posture

Wave 8 hardening included a full static security scan of the repository. This
page records the methodology, the results, and the standing mitigations already
designed into the codebase (see `adr/` for the underlying decisions).

## Mimosa deep scan (Wave 8)

| Field | Value |
| --- | --- |
| Date | 2026-09-24 |
| Tool | Mimosa deep scan (static analysis + offline dependency advisories, no runtime execution) |
| Scope | 275 source files parsed (0 failures), 120 dependency packages scanned |
| Result | **0 findings** — 0 high, 0 medium, 0 low, 0 informational, 0 business-logic candidates |
| Scan ID | `scan-2026-09-24T15-32-00.389Z-3f46166e9990` |
| Seal | `sha256:b5d42d4b81a722d8ab225be214a4a382f21a9c885c7bef2b63934bb8b65016c2` |

The scan's own run status is `inconclusive` for one documented reason: parts of
the call graph are dynamic dispatch (Pydantic-validated models, duck-typed
provider protocols), so cross-file reachability is partially incomplete. All
analysis phases (threat model, finding discovery, validation) completed; the
path-analysis phase had no findings to trace. No high/critical findings
required fixes.

## Standing mitigations (by design)

These are architectural properties already enforced in waves 0–7, not scan
findings:

- **Secrets by reference only.** Provider credentials are stored as
  *environment-variable names* (`api_key_ref`), never as values, in every
  config model (ADR-0004/0006). The Streamlit console masks secret fields.
- **Local-first defaults.** The default generation/rerank/embedding endpoints
  are `localhost` services (LM Studio at `http://localhost:1234/v1`, ADR-0004);
  nothing ships credentials or telemetry off-host by default.
- **Strict Pydantic configs.** All config models use `extra="forbid"`, so
  unexpected keys in TOML files fail loudly instead of being ignored.
- **No code execution from documents.** Parsing (PDF/OCR) treats documents as
  untrusted input; the ingestion pipeline never evaluates document content.

## Dependency advisories

The scanner's offline advisory pass matched 1 advisory across the 120 scanned
packages at informational level; it did not surface as a finding (no vulnerable
version ranges confirmed in `uv.lock`). For continuous coverage, run
`uv lock --upgrade` periodically and re-run the scan after dependency bumps.

## Known limitations

- The static scan cannot reach dynamic-dispatch call sites (see above); the
  e2e suite (`scripts/check.py --e2e`) provides the runtime complement by
  exercising the full ingestion → retrieval → generation path.
- `benchmarks/` and `examples/` are development-only code and run with the
  same trust assumptions as the test suite; do not point examples at untrusted
  OpenAI-compatible endpoints without TLS.
