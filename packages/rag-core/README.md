# rag-core
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Canonical domain models and runtime-checkable Protocol contracts for the
[rag-aio](https://github.com/omkumar01/rag-aio) platform. Every other package
depends on `rag-core` — never the reverse.

## Overview

`rag-core` is the single source of truth for the rag-aio type system. It defines:

* **Domain models** -- boundary Pydantic v2 models for documents, chunks,
  embeddings, retrieval hits, rerank hits, context items, generation requests/
  responses, queries, evaluation results, pipeline jobs, and provider/model
  metadata.
* **Protocol contracts** -- 22 `@runtime_checkable` `typing.Protocol` interfaces
  (`DocumentLoader`, `DocumentParser`, `OCRProcessor`, `Tokenizer`, `Chunker`,
  `Embedder`, `SparseEmbedder`, `Indexer`, `VectorStore`, `DocumentStore`,
  `KeyValueStore`, `Cache`, `Retriever`, `HybridRetriever`, `FusionStrategy`,
  `Reranker`, `QueryStrategy`, `ContextBuilder`, `Generator`, `LLMProvider`,
  `EvaluationMetric`, `Observer`). Implementations depend on these abstractions,
  never on vendors or on each other's concrete classes (see
  [ADR-0002](https://github.com/omkumar01/rag-aio/blob/main/adr/ADR-0002-protocol-based-contracts.md)).
* **Utility functions** -- deterministic identifiers and content hashes
  (`new_id`, `content_hash`, `stable_id`, `config_hash`).
* **SerDe helpers** -- compact JSON round-trip helpers (`to_json`, `from_json`).
* **Error taxonomy** -- a single `RagError` base with a structured `code` and
  `details` dict, plus 23 typed subclasses organized by pipeline stage.

By design, installing `rag-core` pulls in **only** `pydantic`. There are no
indirect transitive dependencies.

## Installation

```bash
pip install rag-core
# or, within the uv workspace:
uv pip install rag-core
```

Import everything from the top-level package:

```python
from rag_core import (
    Document,
    Chunk,
    Query,
    RetrievalHit,
    Context,
    GenerationRequest,
    GenerationResult,
    RagError,
    ConfigError,
    RetrievalError,
    new_id,
    content_hash,
    config_hash,
    to_json,
    from_json,
)
import rag_core

print(rag_core.__version__)  # 0.1.0
```

Protocol classes are available under the `rag_core.protocols` namespace:

```python
from rag_core import protocols
# protocols.Embedder, protocols.VectorStore, protocols.Generator, ...
```

## Architecture / Design Principles

### Contracts first (ADR-0002)

Every cross-module capability is a `typing.Protocol` defined here. Implementations
are adapters that structurally satisfy these interfaces — no inheritance required.
This lets `rag-embedder` accept any `VectorStore` duck-typed against
`rag_core.protocols.VectorStore`, and lets `rag-retrieval` plug in a replacement
`FusionStrategy` without coordination.

### Runtime-checkable protocols

All protocols are decorated with `@runtime_checkable`, enabling `isinstance(obj,
protocols.Embedder)` checks. Contract tests assert structural conformance for
every adapter. See `tests/test_protocols.py` for minimal reference implementations
of each protocol.

### Strict boundary models

`RagBaseModel` enforces `model_config = ConfigDict(extra="forbid")` so that
schema drift across module versions fails loudly instead of silently dropping
fields. All boundary data flowing between stages is a `RagBaseModel` subclass;
hot-path internals in consuming modules use dataclasses and NumPy arrays.

### Deterministic identifiers

`new_id` produces a random UUID4 hex (32 chars). `content_hash` is SHA-256 over
raw bytes. `stable_id` and `config_hash` canonicalize inputs via sorted-key JSON
so the same content always produces the same identifier across processes — this
drives incremental ingestion, cache keys, and reproducible evaluation runs.

### Error taxonomy

Every exception is a `RagError` subclass carrying a stable machine-readable
`code` and an optional `details` dict. Exceptions are never swallowed silently;
the orchestrator catches `RagError` (and its subclasses) to produce structured
error responses. The severity hierarchy is:

```
RagError
  ├── ConfigError
  ├── IngestionError
  │     ├── ParseError
  │     ├── UnsupportedFormatError
  │     ├── CrawlError
  │     ├── OCRError
  │     └── ChunkingError
  ├── EmbeddingError
  ├── IndexError_        (index write/indexing failures)
  ├── StorageError
  │     └── CacheError
  ├── RetrievalError
  │     └── FusionError
  ├── RerankError
  ├── ContextError
  ├── ProviderError
  │     ├── ProviderUnavailableError
  │     ├── RateLimitError
  │     └── GenerationError
  ├── OperationTimeout
  ├── BudgetExceededError
  ├── JobError
  └── EvaluationError
```

## Source tree

```
src/rag_core/
  __init__.py       Public API re-exports; version = "0.1.0"
  base.py           RagBaseModel (strict, extra="forbid")
  protocols.py      22 runtime-checkable Protocol contracts
  documents.py      Document, DocumentMetadata, DocumentPage, PageBlock,
                    DocumentAsset, BlockKind, BoundingBox
  chunks.py         Chunk, ChunkMetadata (deterministic ids)
  embeddings.py     Embedding, SparseEmbedding, EmbeddingKind
  types.py          SparseVector
  retrieval.py      RetrievalHit, RetrievalResult (explainable scores)
  rerank.py         RerankHit (preserves pre-rerank signal)
  context.py        Context, ContextItem, Citation
  generation.py     Message, Role, Usage, FinishReason,
                    GenerationRequest, GenerationResult
  queries.py        Query, QueryVariant, QueryVariantKind
  evaluation.py     MetricResult, QueryEvaluation, EvaluationResult
  jobs.py           PipelineJob, JobStatus
  models_info.py    ModelInfo, ProviderInfo, ProviderKind, HealthState
  ids.py            new_id, content_hash, stable_id, config_hash
  serde.py          to_json, from_json
  errors.py         RagError + 23 subclasses
```

## Public API

### Domain models

#### Documents (`documents.py`)

```python
from rag_core import Document, DocumentMetadata, DocumentPage, PageBlock, BoundingBox

doc = Document(
    source_uri="s3://bucket/report.pdf",
    metadata=DocumentMetadata(title="Q3 Report", mime_type="application/pdf", author="Analytics"),
    text="Executive summary ...",
    pages=[
        DocumentPage(
            page_number=1,
            text="Executive summary",
            blocks=[PageBlock(page_id="p1", kind="text", text="...")],
        ),
    ],
)
# content_hash is auto-derived from source_uri + text for stable deduplication
assert doc.content_hash
```

| Model | Key fields |
|---|---|
| `Document` | `id`, `source_uri`, `text`, `metadata`, `pages`, `assets`, `tenant`, `namespace`, `content_hash`, `created_at`, `updated_at` |
| `DocumentMetadata` | `title`, `mime_type`, `language`, `author`, `created_at`, `modified_at`, `custom` |
| `DocumentPage` | `page_number`, `text`, `width`, `height`, `image_ref`, `language`, `blocks` |
| `PageBlock` | `page_id`, `kind` (text/heading/table/figure/...), `text`, `bbox`, `confidence`, `order` |
| `DocumentAsset` | `asset_id`, `kind`, `page_number`, `bbox`, `mime_type`, `content_ref`, `description` |
| `BoundingBox` | `x0`, `y0`, `x1`, `y1` (enforces `x0 <= x1`, `y0 <= y1`) |
| `BlockKind` | Literal: `"text"`, `"heading"`, `"table"`, `"figure"`, `"caption"`, `"list"`, `"header"`, `"footer"`, `"other"` |

#### Chunks (`chunks.py`)

```python
from rag_core import Chunk, ChunkMetadata

meta = ChunkMetadata(
    document_id=doc.id,
    document_hash=doc.content_hash,
    chunker="recursive",
    chunker_version="1.0",
    page_numbers=[1, 2],
    section_path=["Executive summary"],
    char_start=0,
    char_end=500,
    token_count=120,
)
chunk = Chunk(document_id=doc.id, text="...", index=0, metadata=meta)
# id is deterministic: stable_id(document_id, chunker, chunker_version, index, text)
# Reconstructing with the same inputs always yields the same id:
assert chunk.id == Chunk(document_id=doc.id, text="...", index=0, metadata=meta).id
```

| Field | Description |
|---|---|
| `Chunk.id` | Deterministic -- same content + chunker + index always yields the same id |
| `ChunkMetadata.parent_chunk_id` | Set for child chunks in parent/child strategies |

#### Embeddings (`embeddings.py`)

```python
from rag_core import Embedding, SparseEmbedding

dense = Embedding(chunk_id="c1", vector=[0.1, 0.2, 0.3], model="fastembed-BAAI")
sparse = SparseEmbedding(chunk_id="c1", indices=[12, 45, 78], values=[0.9, 0.3, 0.7], model="bm25")
# dimension is auto-inferred from vector length
assert dense.dimension == 3
```

#### Retrieval (`retrieval.py`)

```python
from rag_core import RetrievalHit, RetrievalResult

hit = RetrievalHit(
    chunk_id="c1",
    document_id="d1",
    score=0.92,
    normalized_score=0.88,
    rank=0,
    strategy="dense",
    model="fastembed-BAAI",
    strategy_scores={"dense": 0.92},
    filters_applied={"tenant": "acme"},
)
result = RetrievalResult(
    query_id="q1",
    hits=[hit],
    strategies=["dense", "sparse"],
    timings_ms={"dense": 12.3, "sparse": 4.1},
)
```

#### Rerank (`rerank.py`)

```python
from rag_core import RerankHit

reranked = RerankHit(
    chunk_id="c1",
    document_id="d1",
    score=0.95,  # post-rerank score
    rank=0,
    original_score=0.88,  # preserved pre-rerank signal
    original_rank=2,
)
```

#### Context (`context.py`)

```python
from rag_core import Context, ContextItem, Citation

item = ContextItem(
    chunk_id="c1",
    document_id="d1",
    text="evidence ...",
    token_count=45,
    citation_id="[1]",
    page_numbers=[3],
    document_title="Report",
    source_uri="s3://bucket/report.pdf",
)
ctx = Context(items=[item], token_budget=2048, strategy="relevance_first")
assert ctx.total_tokens == 45
```

| Model | Key fields |
|---|---|
| `ContextItem` | `chunk_id`, `document_id`, `text`, `token_count`, `citation_id`, `page_numbers`, `section_path`, `score`, `document_title`, `source_uri` |
| `Citation` | `citation_id`, `document_id`, `chunk_id`, `source_uri`, `page_numbers`, `quote` |
| `Context` | `items`, `token_budget`, `strategy`, `truncated` (+ `total_tokens` property) |

#### Generation (`generation.py`)

```python
from rag_core import (
    GenerationRequest,
    GenerationResult,
    Message,
    Role,
    Usage,
    FinishReason,
)

request = GenerationRequest(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What are the authentication requirements?"),
    ],
    temperature=0.2,
    max_tokens=512,
    stream=True,  # set False for non-streaming
)
# model is optional -- routing/fallback resolution happens upstream
# (rag-llm-provider / orchestrator), not in the request itself.

result = GenerationResult(
    text="The authentication requirements are...",
    model="qwen3.8-27b",
    finish_reason="stop",
    usage=Usage(prompt_tokens=340, completion_tokens=120, cost_usd=0.001),
)
assert result.usage.total_tokens == 460
```

| Type | Values |
|---|---|
| `Role` | `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `FinishReason` | `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`, `"error"`, `"cancelled"` |

#### Queries (`queries.py`)

```python
from rag_core import Query, QueryVariant, QueryVariantKind

q = Query(text="What are the auth requirements?", filters={"tenant": "acme"}, top_k=10)
variant = QueryVariant(query_id=q.id, text="What authentication is required?", kind="rewrite")
```

| `QueryVariantKind` | Meaning |
|---|---|
| `original` | The untransformed query |
| `rewrite` | Paraphrased / normalized form |
| `expansion` | Added terms (e.g. from a thesaurus) |
| `hyde` | Hypothetical-document embedding query |
| `decomposition` | Breakdown into sub-questions |
| `subquery` | A single sub-question from a decomposition |

#### Evaluation (`evaluation.py`)

```python
from rag_core import EvaluationResult, QueryEvaluation, MetricResult

run = EvaluationResult(
    dataset="cran-v1",
    config_hash="abc123...",
    model_versions={"dense": "fastembed-BAAI", "generator": "qwen3.8-27b"},
    metrics=[MetricResult(name="mrr@10", value=0.82, k=10)],
    per_query=[QueryEvaluation(query_id="q1", metrics=[...])],
)
```

#### Jobs (`jobs.py`)

```python
from rag_core import PipelineJob, JobStatus

job = PipelineJob(job_id="j1", kind="ingest", total_items=100, stage="parsing")
# Progress, errors, checkpoints, and result summaries are tracked on the job.
```

#### Provider / model info (`models_info.py`)

```python
from rag_core import ProviderInfo, ModelInfo, ProviderKind, HealthState

provider = ProviderInfo(
    name="lmstudio",
    kind="openai_compatible",
    base_url="http://localhost:1234/v1",
    auth_ref="LMSTUDIO_API_KEY",  # env var name, never the value itself
    models=[ModelInfo(model_id="qwen3.8-27b", provider="lmstudio", context_window=32768)],
)
# auth_ref is a reference (env var / secret file / keyring handle);
# raw credentials never appear in this model, API responses, logs, or the UI.
```

| `ProviderKind` | |
|---|---|
| `openai`, `anthropic`, `gemini`, `openai_compatible`, `ollama`, `vllm`, `llamacpp` |

| `HealthState` | |
|---|---|
| `unknown`, `healthy`, `degraded`, `down` |

### Protocols (`protocols.py`)

All 22 protocols are `@runtime_checkable` and grouped by pipeline stage:

| Protocol | Method(s) | Stage |
|---|---|---|
| `DocumentLoader` | `async load(source) -> (bytes, dict)` | Ingestion |
| `DocumentParser` | `supported_types()`, `async parse(source, data) -> Document` | Ingestion |
| `OCRProcessor` | `async process_page(document_id, page, image) -> dict` | OCR |
| `Tokenizer` | `count_tokens(text)`, `encode(text)`, `decode(tokens)` | Chunking |
| `Chunker` | `async chunk(document) -> list[Chunk]` | Chunking |
| `Embedder` | `async embed(texts) -> list[list[float]]` | Embedding |
| `SparseEmbedder` | `async embed_sparse(texts) -> list[SparseVector]` | Embedding |
| `Indexer` | `async index(embeddings, sparse)` | Indexing |
| `VectorStore` | `upsert`, `search`, `delete`, `health` | Storage |
| `DocumentStore` | `put`, `get`, `delete`, `find_by_hash` | Storage |
| `KeyValueStore` | `get`, `set`, `delete` | Storage |
| `Cache` | `get`, `set`, `delete` (TTL-aware) | Caching |
| `Retriever` | `async retrieve(query) -> RetrievalResult` | Retrieval |
| `HybridRetriever` | `async retrieve(query) -> RetrievalResult` | Retrieval |
| `FusionStrategy` | `fuse(results, top_k) -> RetrievalResult` | Retrieval |
| `Reranker` | `async rerank(query, candidates, top_k) -> list[RetrievalHit]` | Reranking |
| `QueryStrategy` | `async transform(query) -> list[QueryVariant]` | Query |
| `ContextBuilder` | `async build(query, hits, token_budget) -> Context` | Context |
| `Generator` | `async generate(request)`, `stream(request)` | Generation |
| `LLMProvider` | `async complete(request) -> GenerationResult` | Generation |
| `EvaluationMetric` | `compute(ground_truth, predictions, k) -> float` | Eval |
| `Observer` | `record(event, attributes)` | Observability |

```python
# Implementing a protocol -- no inheritance needed:
from rag_core import protocols


class MyEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# Runtime structural check:
assert isinstance(MyEmbedder(), protocols.Embedder)
```

### Utility functions (`ids.py`, `serde.py`)

```python
from rag_core import new_id, content_hash, stable_id, config_hash, to_json, from_json

# Random IDs
assert len(new_id()) == 32  # UUID4 hex

# Content hashing (str vs bytes differ by design)
assert content_hash("hello") == content_hash("hello")
assert content_hash("hello") != content_hash("hellö")

# Deterministic IDs and config hashes (order-insensitive for dicts)
assert stable_id("doc1", "chunk", 0, "text") == stable_id("doc1", "chunk", 0, "text")
assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

# Compact JSON round-trip
serialized = to_json(doc)  # no whitespace
restored = from_json(serialized, Document)
assert restored == doc
```

### Error handling

```python
from rag_core import RagError, ConfigError, RetrievalError

try:
    raise RetrievalError("vector search failed", code="qdrant_down", details={"status_code": 503})
except RagError as e:
    print(e.code)  # qdrant_down
    print(e.details)  # {'status_code': 503}
```

## Usage Guides

### Beginner: constructing a pipeline input

The most common first step is creating a `Document` from a file path or URL. The
document is the canonical artifact passed through ingestion:

```python
from pathlib import Path
from rag_core import Document, DocumentMetadata, content_hash

raw = Path("report.pdf").read_bytes()
doc = Document(
    source_uri="file:///report.pdf",
    text="Extracted text content ...",
    metadata=DocumentMetadata(mime_type="application/pdf", title="Report"),
)
# The content_hash is used by the orchestrator to decide whether re-ingestion
# is needed. Storing it in a KV store lets you skip unchanged documents.
```

### Beginner: building a Query

```python
from rag_core import Query

q = Query(text="What are the authentication requirements?", filters={"tenant": "acme"}, top_k=5)
# downstream stages read q.text, q.filters, q.top_k, q.deadline_s
```

### Intermediate: chunked ingestion with provenance

```python
from rag_core import Chunk, ChunkMetadata, stable_id

meta = ChunkMetadata(
    document_id=doc.id,
    document_hash=doc.content_hash,
    chunker="recursive",
    chunker_version="1.0",
    page_numbers=[1],
    section_path=["Section 2"],
    token_count=200,
)
chunk = Chunk(document_id=doc.id, text="...", index=3, metadata=meta)
# chunk.id is stable: re-chunking the same document with the same chunker
# and index yields the same id, enabling incremental indexing.
```

### Intermediate: hybrid retrieval result fusion

```python
from rag_core import RetrievalHit, RetrievalResult

dense_results = RetrievalResult(query_id=q.id, hits=[...], strategies=["dense"])
sparse_results = RetrievalResult(query_id=q.id, hits=[...], strategies=["bm25"])


class RRF:
    def fuse(self, results, top_k: int) -> RetrievalResult:
        # merge by chunk_id, rank by reciprocal rank
        ...


fused = RRF().fuse([dense_results, sparse_results], top_k=10)
```

### Advanced: custom protocol implementation

```python
from collections.abc import Sequence
from rag_core import Query, Retriever, RetrievalResult, RetrievalHit
from rag_core import protocols
from rag_core.ids import config_hash
import asyncio


class CachedRetriever:
    """A Retriever that memoizes results in a Cache (protocol-based)."""

    def __init__(self, inner: protocols.Retriever, cache: protocols.Cache):
        self._inner = inner
        self._cache = cache

    async def retrieve(self, query: Query) -> RetrievalResult:
        from rag_core import to_json, from_json
        from rag_core.ids import config_hash

        key = f"retrieval:{config_hash({'text': query.text, 'top_k': query.top_k})}"
        cached = await self._cache.get(key)
        if cached is not None:
            return from_json(cached.decode(), RetrievalResult)
        result = await self._inner.retrieve(query)
        await self._cache.set(key, to_json(result).encode(), ttl=300)
        return result


# Still satisfies the Retriever protocol:
assert isinstance(CachedRetriever(...), protocols.Retriever)
```

### Advanced: evaluation metric implementation

```python
from rag_core import protocols, QueryEvaluation, MetricResult


class RecallAtK:
    name = "recall@k"

    def compute(self, ground_truth, predictions, k=None):
        if k is None:
            k = len(predictions[0]) if predictions else 0
        # ... compute recall@k ...
        return 0.75
```

## Configuration

`rag-core` itself has no runtime configuration — it carries no tunable state.
All "configuration" is expressed as Pydantic models (`RagBaseModel`) passed by
value across protocol boundaries. The versioned, precedence-aware configuration
system is owned by the orchestrator layer (see
[ADR-0006](https://github.com/omkumar01/rag-aio/blob/main/adr/ADR-0006-configuration-system.md)).

The models here are consumed by configuration objects in every downstream
package (e.g. `RetrievalConfig`, `ContextConfig`, `IndexerConfig`).

## Testing

```bash
uv run pytest packages/rag-core -q
```

Four test modules cover the package:

| File | Focus |
|---|---|
| `tests/test_smoke.py` | Version string, basic import |
| `tests/test_ids_errors.py` | ID uniqueness, hash determinism, error hierarchy, config hash on Pydantic models |
| `tests/test_models.py` | Document/chunk/embedding/retrieval/context/generation round-trips, validation constraints (BoundingBox ordering, progress bounds, dimension inference) |
| `tests/test_protocols.py` | Structural conformance: minimal reference implementations of every protocol pass `isinstance` checks |

Tests use the `unit` marker and require no external services.

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| `pydantic` | `>=2.7` | Base model framework, validation, serialization |
| *(transitive)* | — | None. `pydantic-core`, `typing-extensions`, etc. come solely via pydantic. |

## Cross-Package Relationships

`rag-core` is the **bottom** of the dependency graph. Every package imports from
it:

```
rag-core        (no internal deps)
  ├── rag-observe      implements protocols.Observer
  ├── rag-cache        implements protocols.Cache; uses ids.config_hash / to_json
  ├── rag-doc-handler  implements DocumentLoader / DocumentParser / OCRProcessor
  ├── rag-ocr          implements OCRProcessor
  ├── rag-embedder     implements Tokenizer / Chunker / Embedder / SparseEmbedder / Indexer
  ├── rag-db-handler   implements VectorStore / DocumentStore / KeyValueStore
  ├── rag-retrieval    implements Retriever / HybridRetriever / FusionStrategy
  ├── rag-rerank       implements Reranker
  ├── rag-query        implements QueryStrategy
  ├── rag-context      implements ContextBuilder
  ├── rag-llm-provider   implements LLMProvider
  ├── rag-generation     implements Generator
  ├── rag-eval         implements EvaluationMetric
  ├── rag-orchestrator   uses PipelineJob, Error taxonomy
  ├── rag-mass-inject    uses PipelineJob, Document, Chunk
  └── rag-aio            the RAG facade composes everything
```

No `rag-core` model or protocol imports from any other package. The separation
is enforced by ADR-0001 (workspace layout) and ADR-0002 (protocol contracts).
