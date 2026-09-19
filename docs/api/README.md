# APIs

## Python API

Every module is importable and usable in-process. The composition facade is:

```python
from rag_aio import RAG
```

## Service APIs

Modules that can run as services expose FastAPI applications with `/health`, `/ready`,
`/metrics`, `/version`, and typed domain endpoints. Each service publishes an OpenAPI
schema at `/docs`. Endpoint references are added here as services are implemented.

## Versioning

Boundary schemas are versioned (see ADR-0006 and the `schema_version` fields in
`rag-core`). Breaking changes to service APIs require a major version bump and a
migration note in [CHANGELOG](../../CHANGELOG.md).
