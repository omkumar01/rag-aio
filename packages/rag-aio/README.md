# rag-aio

The composition surface of the rag-aio platform: the `RAG` facade
(`RAG.from_config("config.toml")` → `await rag.ingest(...)` → `await rag.ask(...)`), a
Typer CLI (`rag-aio`), and the FastAPI service application (`/health`, `/ready`,
`/metrics`, `/version`, typed domain endpoints, job endpoints).

`rag-aio` is the top-level package users reach for first. It does not implement
RAG stages itself — it selects and wires backends from the other `rag-*` packages
into an `Orchestrator`, then exposes that wiring through three equivalent surfaces:
a Python facade, a command-line tool, and an HTTP service.

## Installation

```bash
uv pip install rag-aio
```

This pulls in the full set of `rag-*` packages plus `fastapi`, `uvicorn[standard]`,
`typer`, and `httpx`. The console-script entry point is registered automatically:

```bash
rag-aio --help
```

## Architecture / Design Principles

- **Batteries-in-or-batteries-out.** A single `RAGConfig` drives everything.
  `RAGConfig.mock()` produces a *fully offline* configuration (deterministic mock
  embedder, in-memory stores, a stub generator) so the facade works with zero
  external services; any TOML config swaps in real backends (FastEmbed + Qdrant +
  LM Studio) with no code change.
- **Lazy heavy imports.** `import rag_aio` is dependency-light. FastEmbed,
  Qdrant, the doc-handler parsers, and NumPy are only imported inside
  `build_services` / `_build_mock_services` / `get_app` — never at module import.
- **Thin over composition.** The facade owns three things only: (1) selecting mock
  vs. local wiring based on `config.embedder.backend`, (2) adapting the
  orchestrator's output to the user-facing `AskResult`, and (3) exposing the FastAPI
  app. Every actual stage lives in its own package (see Cross-Package Relationships).
- **Config is a typed contract, not a free dict.** `RAGConfig` and all sub-configs
  use `rag_core.base.RagBaseModel` (`extra="forbid"`), so typos in a TOML file fail
  loudly at load time rather than silently degrading behavior. See ADR-0006.
- **Secrets by reference.** Provider credentials are named via env-var references
  (e.g. `api_key_ref`), never inlined into config. See ADR-0004 / ADR-0006.

## Public API

### `RAG` — the facade

```python
import asyncio
from rag_aio import RAG, RAGConfig


async def main() -> None:
    # mock() -> fully offline, deterministic, no services running.
    async with RAG.from_config(RAGConfig.mock()) as rag:
        await rag.ingest("documents/meeting-notes.md")
        result = await rag.ask("Who led the quarterly review?")
        print(result.answer)
        for c in result.citations:
            print(f"[{c.citation_id}] {c.source_uri or c.document_id}")


asyncio.run(main())
```

The facade surface:

| Method / property | Signature | Notes |
| ----------------- | --------- | ----- |
| `from_config`     | `(config: RAGConfig) -> RAG` | Builds services from config; the primary constructor. |
| `from_config` (alt) | `("config.toml")` via `RAGConfig.from_file` | See Configuration. |
| `ingest`          | `(path, *, recursive=False) -> Document` | One file through the stage-cached ingest pipeline. |
| `ingest_directory`| `(dir, recursive=False) -> list[Document]` | All supported files under a directory. |
| `ask`             | `(query, *, stream=False, overrides=None) -> AskResult \| AsyncIterator[str]` | Non-streaming returns `AskResult`; `stream=True` yields text deltas. |
| `app`             | `FastAPI` (property) | A FastAPI service backed by this facade's services. |
| `__aenter__` / `__aexit__` | async context manager | Best-effort `close()` on vector + document stores on exit. |

`overrides` is folded into the resolved `PipelineConfig` by the orchestrator
(`top_k`, `model`, `temperature`, `max_tokens`, `token_budget`, `timeout_s`,
`cost_budget_usd`, `concurrency`, …).

### `RAGConfig` + sub-configs

```python
from rag_aio import RAGConfig

cfg = RAGConfig.mock()                 # fully offline profile (backend="mock")
cfg = RAGConfig.from_file("config.toml")  # your tuned profile (backend="fastembed")
print(cfg.model_dump_json(indent=2))
```

| Sub-config        | Key fields | Meaning |
| ----------------- | ---------- | ------- |
| `EmbedderConfig`  | `backend: Literal["mock","fastembed"]`, `dense_model`, `sparse_model`, `dim` | Embedder backend + dim hint (mock uses `dim` for the in-memory store). |
| `StorageConfig`   | `qdrant_path`, `db_url`, `cache_backend: Literal["memory","sqlite"]`, `cache_path` | Vector DB location, SQL document store URL, cache backend. |
| `GenerationConfig`| `provider`, `base_url`, `model`, `api_key_ref`, `temperature`, `max_tokens` | Live-generation settings; `api_key_ref` names an env var. |
| `PipelineConfig`  | (see `rag-orchestrator`) | Embedded verbatim as `cfg.pipeline`; stages, budgets, concurrency. |

### `build_services`

```python
from rag_aio import RAG, RAGConfig
from rag_orchestrator import load_local_services

cfg = RAGConfig.from_file("config.toml")
services = build_services(cfg)           # mock wiring OR load_local_services(...)
rag = RAG(services, cfg.pipeline)
```

`build_services` selects `_build_mock_services` when `cfg.embedder.backend == "mock"`,
otherwise delegates to `rag_orchestrator.load_local_services(...)` with paths taken from
`cfg.storage`. The mock profile substitutes `_StubGenerator` for live generation (see below).

### `_StubGenerator`

Used by the offline profile. It is deterministic and performs *no* HTTP: the answer is
derived from the retrieved context that the orchestrator renders into the user message.
When the prompt contains no context, it returns a fixed fallback string. It records every
`GenerationRequest` it receives (handy for tests) and implements both `generate` and
`stream`.

```python
from rag_aio.facade import _StubGenerator
from rag_core.generation import GenerationRequest

stub = _StubGenerator()
stub.requests        # list[GenerationRequest] captured so far
result = await stub.generate(request)   # text derived from rendered prompt
```

### FastAPI service

```python
from rag_aio.app import get_app, app        # app is a ready module-level instance

application = get_app("config.toml")       # or get_app() -> mock profile
```

`get_app(config_path=None)` loads a `RAGConfig` (mock defaults when `None`), wires
services, and delegates to `rag_orchestrator.create_app(...)`. The module-level
`app = get_app()` instance lets a server boot with no code:

```bash
uvicorn rag_aio.app:app --host 0.0.0.0 --port 8000
# or via the CLI:
rag-aio serve
```

## Usage Guides

### Beginner: zero-config, fully offline

```bash
rag-aio ask "What is the difference between supervised and unsupervised learning?"
```

No config file, no services running — `rag-aio` uses `RAGConfig.mock()`. To ingest a
document first:

```bash
rag-aio ingest ./notes.md
rag-aio ask "Summarize the notes."
```

### Intermediate: streaming answers

```bash
rag-aio ask "Explain transformers" --stream
# or -s
```

In Python:

```python
async for delta in await rag.ask("Explain transformers", stream=True):
    print(delta, end="", flush=True)
print()
```

The HTTP `/v1/ask` endpoint mirrors this: send `{"query": "...", "stream": true}` and
receive an SSE `text/event-stream` of `data: <text-delta>` frames terminated by
`data: [DONE]`.

### Intermediate: pointing at real backends

Create `config.toml`:

```toml
[embedder]
backend = "fastembed"
dense_model = "BAAI/bge-small-en-v1.5"
sparse_model = "Qdrant/bm25"
dim = 384

[storage]
qdrant_path = "./data/qdrant"
db_url = "sqlite+aiosqlite:///./data/rag.db"
cache_backend = "memory"

[generation]
provider = "lm_studio"
base_url = "http://localhost:1234/v1"
model = "local-chat"
temperature = 0.2

[pipeline]
name = "local_fast"
# pipeline.* overrides are folded here; see rag-orchestrator for the full schema.
[pipeline.retrieval.overrides]
top_k = 10
[pipeline.generation.overrides]
temperature = 0.2
```

Then:

```bash
rag-aio -c config.toml ingest ./docs
rag-aio -c config.toml ask "What does the architecture diagram show?"
rag-aio -c config.toml serve --port 8000
```

### Advanced: programmatic service reuse

```python
import asyncio
from rag_aio import RAG, RAGConfig
from rag_orchestrator import create_app

async def main():
    cfg = RAGConfig.from_file("config.toml")
    async with RAG.from_config(cfg) as rag:
        await rag.ingest("./documents")
        # Use the orchestrator's per-request overrides:
        result = await rag.ask(
            "What are the auth requirements?",
            overrides={"top_k": 25, "temperature": 0.0, "max_tokens": 256},
        )
        return result

asyncio.run(main())
```

The `RAG.app` property and `rag_aio.app.get_app(...)` both produce a FastAPI app from the
*same* services, so the CLI, the facade, and the service host always agree on wiring.

## Configuration

`RAGConfig` is loaded from TOML via the stdlib `tomllib` (Python 3.12+). The shape is a
flat union of its sub-configs plus an embedded `pipeline` table:

```toml
[embedder]
backend = "mock" | "fastembed"
dense_model = "name"          # FastEmbed model id
sparse_model = "name"
dim = 384                     # hint for the vector store dimensionality

[storage]
qdrant_path = "./data/qdrant"
db_url = "sqlite+aiosqlite:///./data/rag.db"
cache_backend = "memory" | "sqlite"
cache_path = "./data/cache.db"    # required only when cache_backend = "sqlite"

[generation]
provider = "lm_studio"          # informational name; base_url selects the wire format
base_url = "http://localhost:1234/v1"
model = "local-chat"
api_key_ref = "LM_STUDIO_API_KEY"   # env var name read at call time (optional)
temperature = 0.2
max_tokens = 512

[pipeline]
name = "local_fast"
description = "..."
timeout_s = 30.0
token_budget = 256
cost_budget_usd = 0.10
concurrency = 4
cache_reads = true
cache_writes = true

[pipeline.retrieval.overrides]
top_k = 10
candidate_k = 50
...
[pipeline.generation.overrides]
base_url = "http://localhost:1234/v1"
model = "local-chat"
temperature = 0.2
```

Validation rules:

- `RAGConfig` and all sub-configs forbid extra keys (`extra="forbid"`); a typo raises
  `ValidationError` at load time (ADR-0006).
- `RAGConfig.mock()` overrides the pipeline's embedding `policy` to `"mock"` and sets
  `embedder.backend = "mock"` so `build_services` picks the offline wiring.
- `GenerationConfig.api_key_ref` is never a credential value — it names the env var that
  `rag_generation.create_provider` resolves per request.

## Testing

Tests run fully offline; the facade's mock profile requires no LM Studio, Qdrant, or
cloud keys.

```bash
uv run ruff format packages/rag-aio && uv run ruff check packages/rag-aio
uv run mypy packages/rag-aio/src
uv run pytest packages/rag-aio -q
```

Test layout:

```
tests/
  aio_fakes.py            # fake services + _StubGenerator stand-ins
  conftest.py             # fixtures; makes aio_fakes importable (importlib mode)
  test_smoke.py           # import / version guard
  test_config.py          # TOML load, mock(), extra=forbid validation, overrides
  test_facade.py          # RAG.ask/ingest/app wiring, lifecycle, stub generator
  test_cli.py             # ingest / ask (incl. --stream) / config / --help
```

## Dependencies

`rag-aio` is an aggregation package — it depends on every stage package plus the service
runtime:

- Core: `rag-core`, `rag-observe`, `rag-cache`
- Storage: `rag-db-handler`
- Documents: `rag-doc-handler`, `rag-ocr`
- Embeddings: `rag-embedder`
- Retrieval: `rag-retrieval`, `rag-rerank`, `rag-query`, `rag-context`
- Generation: `rag-llm-provider`, `rag-generation`
- Orchestration: `rag-orchestrator`
- Tooling: `rag-mass-inject` (bulk ingestion), `rag-eval` (metrics)
- Service runtime: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `typer>=0.12`,
  `httpx>=0.27`

`import rag_aio` stays light: heavy backends are imported lazily inside `build_services`
and `get_app`.

## Cross-Package Relationships

- **Composes `rag-orchestrator`**: `RAG` constructs an `Orchestrator` and forwards
  `ingest`/`ingest_directory`/`ask` to it. `OrchestratorServices` is built by
  `build_services`, which either assembles the mock bag (`_build_mock_services`) or calls
  `rag_orchestrator.load_local_services(...)`.
- **Embeds `rag-generation`**: in the offline profile the generator slot is
  `_StubGenerator` (no HTTP); in the local profile it is a `GenerationService` wrapping an
  `OpenAICompatibleProvider` pointed at LM Studio.
- **Consumes `rag-llm-provider`**: `ProvidersConfig.local_default()` + `ProviderRegistry`
  + `ModelRouter` are wired into `OrchestratorServices` regardless of profile, so routing
  is consistent across offline and live runs.
- **Owns the FastAPI surface**: `get_app`/`app` delegate to
  `rag_orchestrator.create_app(...)`, which mounts the orchestrator and the stage-cached
  ingest pipeline behind `/health`, `/ready`, `/metrics`, `/version`, `/v1/ask`,
  `/v1/ingest`, `/v1/jobs/{job_id}`, and `/v1/pipelines`.
- **CLI as a thin driver**: the Typer `app` (`rag-aio` console script) mirrors the facade
  API — `ingest`, `ask` (`-s/--stream`), `config` (print resolved JSON), `serve`
  (uvicorn). It is intentionally free of business logic.
- See ADR-0004 (LM Studio default), ADR-0005 (FastAPI service APIs), ADR-0006
  (typed configuration system).

## License

MIT
