# rag-aio

Asynchronous, modular, high-performance RAG platform and reusable Python SDK.

`rag-aio` is a `uv` workspace monorepo of independently testable, independently deployable
packages that compose into a full Retrieval-Augmented Generation pipeline — from document
ingestion and OCR through chunking, embedding, hybrid retrieval, reranking, context
engineering, and provider-agnostic LLM generation — with observability, caching,
evaluation, and a Streamlit management console built in.

## Quickstart

```python
import asyncio
from rag_aio import RAG


async def main() -> None:
    rag = RAG.from_config("config.toml")
    await rag.ingest_directory("./documents")
    result = await rag.ask("What are the authentication requirements?")
    print(result.answer)
    print(result.citations)


asyncio.run(main())
```

Defaults are optimized for a local high-performance setup: PyMuPDF for fast PDF extraction,
token-aware chunking, FastEmbed embeddings, Qdrant (embedded local mode) for vectors,
BM25 + dense hybrid retrieval with RRF, CrossEncoder-style reranking on a small candidate
set, and OpenAI-compatible generation (LM Studio, vLLM, Ollama, or any compatible server).

## Packages

| Package | Responsibility |
| --- | --- |
| `rag-core` | Canonical domain models and Protocol contracts (dependency-light) |
| `rag-observe` | OpenTelemetry instrumentation, structured logging, redaction |
| `rag-cache` | Multi-level caching (memory/disk/sqlite, redis extra) |
| `rag-db-handler` | Persistence abstractions: SQL, document, KV, vector, sparse, cache stores |
| `rag-doc-handler` | Document loading/parsing: PDF, HTML, web/sitemap, Office, email, Markdown |
| `rag-ocr` | Hybrid mechanical + semantic OCR with provenance-preserving regions |
| `rag-embedder` | Chunking, tokenization, embedding, indexing orchestration |
| `rag-retrieval` | Dense/sparse/hybrid retrieval, fusion, filters, explainability |
| `rag-rerank` | Second-stage relevance optimization |
| `rag-query` | Query normalization, rewriting, expansion, routing |
| `rag-context` | Context engineering: budgets, dedup, expansion, citations |
| `rag-llm-provider` | Provider/model registry, capabilities, fallback chains |
| `rag-generation` | Provider-agnostic generation with streaming and structured output |
| `rag-orchestrator` | Strategy and pipeline management, structured concurrency |
| `rag-mass-inject` | High-throughput bulk ingestion with checkpoints |
| `rag-eval` | Retrieval and generation quality metrics, reproducible runs |
| `rag-ui` | Streamlit management and experimentation console |
| `rag-aio` | `RAG` facade, Typer CLI, FastAPI service |

See [architecture.md](architecture.md) for the full picture and
[docs/](docs/README.md) for module-level documentation.

## Development

```bash
uv sync                    # create environment from lockfile
uv run python scripts/check.py          # format + lint + types + tests
uv run python scripts/check.py --fast   # skip integration/e2e
```

Architecture decisions are recorded in [adr/](adr/). Design principles live in
[design.md](design.md); release history in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
