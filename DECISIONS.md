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

- Python `>=3.12` (numpy stubs use 3.12+ syntax under mypy); CI matrix 3.12–3.13; local
  development on 3.13.
- Pydantic v2, FastAPI, httpx, Typer, pytest/pytest-asyncio, ruff, mypy strict.
- In-house evaluation metrics (no ranx) to keep `rag-eval` dependency-light.
- LangChain/LlamaIndex not dependencies; optional adapters only if a concrete benefit
  is demonstrated.
- Rerank endpoint auto-detection (`rag-rerank`): the LM Studio / qwen3-reranker
  endpoint-shape question is resolved by runtime probing rather than configuration.
  `RemoteReranker` probes `POST {base_url}/rerank` once with a one-document probe body
  (`{model, query: "probe", documents: ["probe"]}`): HTTP 200 selects native `/rerank`
  mode (Jina/Cohere-shaped `results[i].score`, index-aligned); 404/405/422 selects
  chat-template scoring over `/chat/completions` (a JSON array of 0-10 relevance scores
  mapped to 0-1); any other status raises `RerankError`. The probe outcome is logged at
  INFO and the detected mode is cached on the instance for its lifetime (`.mode`
  property).

## Open questions

- None currently.
