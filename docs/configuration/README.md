# Configuration

rag-aio is configured by a single typed model, `RAGConfig`
(`packages/rag-aio/src/rag_aio/config.py`), composed with the versioned
`PipelineConfig` (`packages/rag-orchestrator/src/rag_orchestrator/config.py`). The
policy — precedence, secrets-by-reference, hashing, versioning — is defined in
[ADR-0006](../../adr/ADR-0006-configuration-system.md).

Every model derives from `rag_core.base.RagBaseModel` with `extra="forbid"`, so a typo
or an unknown key in a TOML file raises a `ValidationError` at load time instead of
silently degrading behavior.

## The `RAGConfig` model

`RAGConfig` has four sub-configs. Constructors: `RAGConfig()` (all defaults),
`RAGConfig.from_file(path)` (TOML via stdlib `tomllib`), and `RAGConfig.mock()` (offline
profile — sets `embedder.backend = "mock"` and overrides
`pipeline.embedding.policy` to `"mock"`).

### `pipeline: PipelineConfig`

A declarative spec of the six stages plus execution budgets. Each stage type
(`IngestionStage`, `EmbeddingStage`, `RetrievalStage`, `RerankStage`, `ContextStage`,
`GenerationStage`) shares the `StageConfig` shape:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | `bool` | `True` | When `False` the stage is skipped at runtime. |
| `strategy` | `str \| None` | `None` | Named algorithm within the stage (e.g. `"rrf"`, `"recursive"`). |
| `policy` | `str \| None` | `None` | Named operational policy (e.g. `"fastembed"`, `"mock"`, `"hybrid"`). |
| `overrides` | `dict[str, Any]` | `{}` | Merged into the bound component's own config at wiring time. |

Top-level `PipelineConfig` fields:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `str` | `"1.0"` (`pipeline_schema_version`) | Bumped on breaking shape changes. |
| `name` | `str` | `"default"` | Profile name (`local_default()` produces `"local_fast"`). |
| `description` | `str \| None` | `None` | Human-readable profile description. |
| `ingestion` | `IngestionStage` | `enable=True` | Parsing/OCR stage (e.g. `policy="fast"`, `strategy="pymupdf"`). |
| `embedding` | `EmbeddingStage` | `enable=True` | Chunking + dense/sparse embedding. |
| `retrieval` | `RetrievalStage` | `enable=True` | Retrieval/fusion stage. |
| `rerank` | `RerankStage` | `enable=True` | Reranking stage. |
| `context` | `ContextStage` | `enable=True` | Context assembly stage. |
| `generation` | `GenerationStage` | `enable=True` | Generation provider stage. |
| `timeout_s` | `float \| None` | `None` | End-to-end budget for one `ask`; `None` = unlimited. |
| `token_budget` | `int \| None` | `None` | Generation answer budget; propagated to the context builder. |
| `cost_budget_usd` | `float \| None` | `None` | Soft generation spend cap (USD). |
| `concurrency` | `int` (ge=1) | `4` | Parallelism for independent retrieval branches. |
| `cache_reads` / `cache_writes` | `bool` | `True` | Stage-cache behavior during ingestion. |

The bundled `PipelineConfig.local_default()` profile wires: PyMuPDF parsing with OCR,
`recursive` chunking, FastEmbed dense (`BAAI/bge-small-en-v1.5`) + BM25-style sparse,
hybrid RRF retrieval (`candidate_k=50`, `top_k=10`), heuristic rerank, a 2048-token
`relevance_first` context, and LM Studio generation (`http://localhost:1234/v1`).

### `embedder: EmbedderConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | `Literal["mock", "fastembed"]` | `"fastembed"` | `"mock"` selects the offline wiring (mock fast-path). |
| `dense_model` | `str \| None` | `None` | Dense model override. |
| `sparse_model` | `str \| None` | `None` | Learned-sparse model override. |
| `dim` | `int` | `384` | Embedding dimension; sizes the mock in-memory store. |

### `storage: StorageConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `qdrant_path` | `str` | `"./data/qdrant"` | Local Qdrant persistence path (ADR-0003). |
| `db_url` | `str` | `"sqlite+aiosqlite:///./data/rag.db"` | SQL document store URL. |
| `cache_backend` | `Literal["memory", "sqlite"]` | `"memory"` | Cache backend selection. |
| `cache_path` | `str \| None` | `None` | Location for the sqlite cache backend. |

### `generation: GenerationConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"lm_studio"` | Named provider (ADR-0004). |
| `base_url` | `str` | `"http://localhost:1234/v1"` | OpenAI-compatible endpoint. |
| `model` | `str \| None` | `None` | Model name for the provider. |
| `api_key_ref` | `str \| None` | `None` | **Environment-variable name** holding the key — never the credential itself. |
| `temperature` | `float` | `0.2` | Sampling temperature. |
| `max_tokens` | `int \| None` | `None` | Generation token cap. |

### Fields that matter on the mock fast-path

`build_services` wires the fully-offline stack when `embedder.backend == "mock"`. On
that path only `embedder.dim` (in-memory vector size), `embedder.backend`, and
`pipeline.embedding.policy = "mock"` (tells downstream stages to expect mock
implementations) take effect. `storage.*` and `generation.*` are ignored: vectors live
in an `InMemoryVectorStore`, the cache is in-memory, and generation is the deterministic
`_StubGenerator` (no HTTP).

## TOML loading

`RAGConfig.from_file(path)` reads a TOML file with `tomllib` and validates it through
`model_validate`. Sections mirror the model nesting; unknown keys anywhere fail because
of `extra="forbid"`. A minimal mock profile
([examples/quickstart_config.toml](../../examples/quickstart_config.toml)) overrides just
two keys:

```toml
[embedder]
backend = "mock"

[pipeline.embedding]
policy = "mock"
```

A complete real (local FastEmbed + LM Studio) profile:

```toml
[pipeline]
name = "local_fast"

[pipeline.ingestion]
policy = "fast"
strategy = "pymupdf"
overrides = { ocr = true }

[pipeline.embedding]
policy = "fastembed"
strategy = "recursive"
overrides = { dense = "BAAI/bge-small-en-v1.5", sparse = "Qdrant/bm25" }

[pipeline.retrieval]
policy = "hybrid"
strategy = "rrf"
overrides = { mode = "hybrid", top_k = 10, candidate_k = 50 }

[pipeline.rerank]
policy = "rerank"
strategy = "heuristic"
overrides = { max_candidates = 50, top_k = 10 }

[pipeline.context]
policy = "relevance_first"
strategy = "relevance_first"
overrides = { token_budget = 2048, reserve_for_answer = 64 }

[pipeline.generation]
policy = "lm_studio"
strategy = "openai_compatible"
overrides = { base_url = "http://localhost:1234/v1", temperature = 0.2 }

[embedder]
backend = "fastembed"
dim = 384

[storage]
qdrant_path = "./data/qdrant"
db_url = "sqlite+aiosqlite:///./data/rag.db"
cache_backend = "memory"

[generation]
provider = "lm_studio"
base_url = "http://localhost:1234/v1"
api_key_ref = "LM_STUDIO_API_KEY"  # env var name, never the key value
temperature = 0.2
```

## Precedence and per-request overrides

[ADR-0006](../../adr/ADR-0006-configuration-system.md) defines the precedence
(lowest → highest): built-in defaults → config file (TOML) → environment variables
(`RAG_AIO_` prefix) → persisted UI configuration → per-request override.

As implemented today:

- **Defaults + TOML file** — fully implemented (`RAGConfig`, `from_file`).
- **Environment variables (`RAG_AIO_` prefix)** — specified by ADR-0006 but **not yet
  implemented** in the loader; the only env interaction is secret *resolution*, below.
- **Persisted UI configuration** — the Streamlit console (`packages/rag-ui`, Config tab)
  edits every non-secret `RAGConfig` field and saves the validated config back to a TOML
  path, so the file on disk is the persistence layer.
- **Per-request override** — fully implemented: `rag.ask(..., overrides={...})` and
  `POST /v1/ask` `overrides` are folded into the pipeline by
  `PipelineConfig.merge_overrides`. Recognized flat keys: `top_k`, `candidate_k`,
  `mode`, `fusion`, `dense_weight`, `sparse_weight` (retrieval); `max_candidates`
  (rerank); `token_budget` (top-level + context); `base_url`, `model`, `temperature`,
  `max_tokens`, `top_p` (generation); `timeout_s`, `cost_budget_usd`, `concurrency`
  (top-level). Unknown keys land in `generation.overrides` so provider-specific options
  pass without fighting validation.

## Secrets by reference

Secrets are never stored by value. `generation.api_key_ref` names an environment
variable that is read at call time; `rag_llm_provider.secrets.SecretRef` generalizes the
pattern to `kind: "env" | "file" | "none"` and resolves the credential on demand. Because
models only ever hold the reference string, `model_dump`, logs, and API payloads are
inherently safe; the UI additionally masks references on screen (`mask_secret` reveals
only the tail, e.g. `•••••_KEY`).

## Configuration hash recording

Two reproducibility mechanisms record a non-secret hash of the effective configuration:

- `rag_core.ids.config_hash` — an order-insensitive SHA-256 over a canonical JSON
  serialization. The orchestrator's stage cache keys ingest results by
  `content_hash + config_hash`, so a config change invalidates cached stages.
- `benchmarks/harness.py` — each benchmark run records a 16-hex `config_hash` (with
  timestamp, platform, CPU count) into the results JSON and report; runs with different
  hashes must not be compared directly.

## Reference

- Minimal real example: [examples/quickstart_config.toml](../../examples/quickstart_config.toml)
- Policy: [ADR-0006 — configuration system](../../adr/ADR-0006-configuration-system.md)
- Package overview: [packages/rag-aio README](../../packages/rag-aio/README.md)
