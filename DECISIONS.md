# Decisions Index

Nontrivial architectural assumptions are recorded as ADRs in [adr/](adr/). This index
tracks them plus open questions.

## Accepted ADRs

| ADR | Decision |
| --- | --- |
| [0001](adr/ADR-0001-uv-workspace-monorepo.md) | uv workspace monorepo layout |
| [0002](adr/ADR-0002-protocol-based-contracts.md) | Protocol-based contracts with adapter registries |
| [0003](adr/ADR-0003-qdrant-local-mode-default.md) | Qdrant local mode as default vector store |
| [0004](adr/ADR-0004-lm-studio-default-provider.md) | OpenAI-compatible (LM Studio) as default local provider |
| [0005](adr/ADR-0005-fastapi-service-apis.md) | FastAPI + async HTTP service APIs; no gRPC until benchmarked |
| [0006](adr/ADR-0006-configuration-system.md) | Typed configuration system and precedence |
| [0007](adr/ADR-0007-jobs-for-long-running-operations.md) | Jobs with IDs for long-running operations |
| [0008](adr/ADR-0008-tdd-and-quality-gates.md) | Strict TDD and quality gates |

## Conventional decisions (no ADR required)

- Python `>=3.11`; CI matrix 3.11–3.13; local development on 3.13.
- Pydantic v2, FastAPI, httpx, Typer, pytest/pytest-asyncio, ruff, mypy strict.
- In-house evaluation metrics (no ranx) to keep `rag-eval` dependency-light.
- LangChain/LlamaIndex not dependencies; optional adapters only if a concrete benefit
  is demonstrated.

## Open questions

- Reranking via LM Studio's qwen3-reranker: confirm the endpoint shape exposed by the
  local server (native `/rerank` vs scoring-template over `/v1/chat/completions`) during
  `rag-rerank` implementation.
