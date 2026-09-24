<h1 align="center">rag-aio</h1>

<p align="center">
  <strong>Local-first RAG. Eighteen packages. One facade.</strong><br>
  An asynchronous, modular RAG platform and reusable Python SDK — from OCR to citations in one process.
</p>

<p align="center">
<a href="https://github.com/omkumar01/rag-aio/actions/workflows/ci.yml?query=branch%3Amain">
  <img src="https://github.com/omkumar01/rag-aio/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
</a>
<a href="https://github.com/omkumar01/rag-aio/actions/workflows/release.yml">
  <img src="https://github.com/omkumar01/rag-aio/actions/workflows/release.yml/badge.svg?branch=main" alt="Release">
</a>
<a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
</a>
<a href="https://github.com/omkumar01/rag-aio">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?color=%2334D058" alt="Python 3.12+">
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/github/license/omkumar01/rag-aio" alt="License">
</a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#packages">Packages</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#development">Contribute</a>
</p>

`rag-aio` is a `uv` workspace monorepo of independently testable, independently deployable
packages that compose into a full Retrieval-Augmented Generation pipeline — from document
ingestion and OCR through chunking, embedding, hybrid retrieval, reranking, context
engineering, and provider-agnostic LLM generation — with observability, caching,
evaluation, bulk ingestion, a benchmark suite, and a Streamlit management console built in.

**Why rag-aio?**

| Common RAG-platform problem | rag-aio approach |
| --- | --- |
| Vendor lock-in on models and stores | Protocol-based contracts; OpenAI-compatible providers, Qdrant/sqlite/InMemory |
| "Works on my machine" demos | Fully-offline mock profile; the quickstart runs with zero services and zero API keys |
| Opaque pipelines | Stage timings on every answer, citation provenance, explainable retrieval and rerank |
| Heavy frameworks, heavy imports | Lazy loading: `import rag_aio` stays dependency-light; heavy backends load on demand |
| Secrets sprawl | Secrets by reference (env-var names), never values; redacted in logs, traces, and UI |
| Slow bulk onboarding | `rag-mass-inject` with bounded queues, per-stage concurrency, checkpoints, dead letters |

## Installation

Everything ships in a single PyPI project: one `rag-aio` wheel contains all 18
packages' code. The base install pulls only light, pure-Python dependencies;
heavy backends are opt-in via extras:

```bash
pip install rag-aio                     # all packages, mock/light backends
pip install "rag-aio[all]"              # + every optional backend
pip install "rag-aio[qdrant,fastembed]" # the full local RAG stack
pip install "rag-aio[streamlit]"        # management console UI
```

Available extras: `qdrant`, `fastembed`, `sentence-transformers`, `faiss`,
`postgres`, `pgvector`, `redis`, `docling`, `msg`, `rtf`, `pydantic-ai`,
`streamlit`, and the `all` catch-all. Without them, the fully-offline mock
profile still runs end to end — the quickstart below needs nothing extra.

For development, clone and sync the uv workspace instead:

```bash
uv sync
```

## Quickstart

The fully offline path — no model server, no vector database, no network:

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

with a two-key `config.toml` that selects the deterministic mock profile
([examples/quickstart_config.toml](examples/quickstart_config.toml)):

```toml
[embedder]
backend = "mock"

[pipeline.embedding]
policy = "mock"
```

The same code runs against a real local stack by pointing the config at
[FastEmbed](https://qdrant.github.io/fastembed/) embeddings, Qdrant (embedded local mode),
SQLite, and an OpenAI-compatible server such as [LM Studio](https://lmstudio.ai):

```toml
[generation]
provider = "lm_studio"
base_url = "http://localhost:1234/v1"
```

Defaults are optimized for a local high-performance setup: PyMuPDF for fast PDF extraction,
token-aware chunking, FastEmbed embeddings, Qdrant (embedded local mode) for vectors,
BM25 + dense hybrid retrieval with RRF, CrossEncoder-style reranking on a small candidate
set, and OpenAI-compatible generation (LM Studio, vLLM, Ollama, or any compatible server).

### CLI and services

```bash
uv run rag-aio ingest ./documents --config config.toml   # bulk ingest
uv run rag-aio ask "What are the auth requirements?"     # one-shot question
uv run rag-aio serve                                     # FastAPI service (/docs for OpenAPI)
uv run streamlit run packages/rag-ui/src/rag_ui/app.py   # management console
```

See [docs/api/README.md](docs/api/README.md) for the full Python, CLI, and HTTP API
reference, and [docs/configuration/README.md](docs/configuration/README.md) for every
setting, default, and precedence rule.

## Packages

Eighteen packages in six layers — each independently installable, each with its own README:

| Package | Layer | Responsibility |
| --- | --- | --- |
| [rag-core](packages/rag-core/README.md) | Infrastructure | Canonical domain models and Protocol contracts (dependency-light) |
| [rag-observe](packages/rag-observe/README.md) | Infrastructure | OpenTelemetry instrumentation, structured logging, redaction |
| [rag-cache](packages/rag-cache/README.md) | Infrastructure | Multi-level caching (memory/disk/sqlite, redis extra) |
| [rag-db-handler](packages/rag-db-handler/README.md) | Infrastructure | Persistence abstractions: SQL, document, KV, vector stores |
| [rag-doc-handler](packages/rag-doc-handler/README.md) | Processing | Document loading/parsing: PDF, HTML, web/sitemap, Office, email, Markdown |
| [rag-ocr](packages/rag-ocr/README.md) | Processing | Hybrid mechanical + semantic OCR with provenance-preserving regions |
| [rag-embedder](packages/rag-embedder/README.md) | Processing | Chunking, tokenization, embedding, indexing orchestration |
| [rag-retrieval](packages/rag-retrieval/README.md) | Intelligence | Dense/sparse/hybrid retrieval, fusion, filters, explainability |
| [rag-rerank](packages/rag-rerank/README.md) | Intelligence | Second-stage relevance optimization |
| [rag-query](packages/rag-query/README.md) | Intelligence | Query normalization, rewriting, expansion, routing |
| [rag-context](packages/rag-context/README.md) | Intelligence | Context engineering: budgets, dedup, expansion, citations |
| [rag-generation](packages/rag-generation/README.md) | Intelligence | Provider-agnostic generation with streaming and structured output |
| [rag-llm-provider](packages/rag-llm-provider/README.md) | Intelligence | Provider/model registry, capabilities, fallback chains |
| [rag-orchestrator](packages/rag-orchestrator/README.md) | Service | Pipeline management, structured concurrency, FastAPI service |
| [rag-mass-inject](packages/rag-mass-inject/README.md) | Operations | High-throughput bulk ingestion with checkpoints |
| [rag-eval](packages/rag-eval/README.md) | Operations | Retrieval and generation quality metrics, reproducible runs |
| [rag-ui](packages/rag-ui/README.md) | Operations | Streamlit management and experimentation console |
| [rag-aio](packages/rag-aio/README.md) | Facade | `RAG` facade, Typer CLI, FastAPI service composition |

## Documentation

Everything is cross-linked from here; each package README links back to this file.

| Section | Contents |
| --- | --- |
| [docs/architecture/](docs/architecture/README.md) | As-built deep dive: layer diagram, ingestion/query paths, contract matrix, jobs |
| [docs/modules/](docs/modules/README.md) | Per-package index: layer, responsibility, dependencies, extension points |
| [docs/api/](docs/api/README.md) | Python facade, CLI, and HTTP API reference (endpoints, SSE, versioning) |
| [docs/configuration/](docs/configuration/README.md) | Full `RAGConfig` reference: fields, defaults, TOML examples, secrets, precedence |
| [docs/deployment/](docs/deployment/README.md) | Topologies (embedded / workstation / scaled services) and run commands |
| [docs/testing/](docs/testing/README.md) | Test strategy: markers, fixtures/fakes, CI behavior, e2e auto-skip |
| [docs/performance/](docs/performance/README.md) | Benchmark methodology and measured results |
| [docs/development/](docs/development/README.md) | Setup, quality gates, conventions, ADR index, security workflow |
| [docs/development/security.md](docs/development/security.md) | Security posture and Mimosa scan results |
| [docs/development/releasing.md](docs/development/releasing.md) | Release process (versioning, build, PyPI trusted publishing) |
| [docs/development/final-report.md](docs/development/final-report.md) | Project-closing report: modules, config, providers, results, topologies |
| [docs/development/implementation-guide.md](docs/development/implementation-guide.md) | Wave-by-wave build plan (historical, all waves done) |
| [architecture.md](architecture.md) · [design.md](design.md) | System architecture and design principles |
| [adr/](adr/) | Architecture decision records (ADR-0001 … ADR-0008) |
| [DECISIONS.md](DECISIONS.md) · [CHANGELOG.md](CHANGELOG.md) | Decision log and release history |

## Examples

Runnable, verified scripts under [examples/](examples/README.md) — each prints what it
demonstrates and exits cleanly when a model server is unreachable:

- [quickstart.py](examples/quickstart.py) — the README flow, fully offline
- [stages/](examples/stages/) — one script per pipeline stage (ingest, retrieve, rerank, context, generate)
- [fully_local.py](examples/fully_local.py) — real FastEmbed + Qdrant + SQLite + LM Studio end to end
- [openai_compatible.py](examples/openai_compatible.py) — any OpenAI-compatible endpoint, env-configured

## Benchmarks

The pytest-based harness in [benchmarks/](benchmarks/README.md) measures document,
embedding, retrieval (dense/hybrid), rerank, context-build, generation, and end-to-end
query performance with P50/P90/P95/P99, cold/warm splits, and RAM tracking; reports land
in gitignored `benchmarks/results/`. Headline numbers from the reference run: hybrid
retrieval ~65 ms, rerank ~2 ms, context build ~6 ms, full offline query ~26 ms — see
[docs/performance/README.md](docs/performance/README.md) for the full table and caveats.

## Development

```bash
uv sync                                 # create environment from lockfile
uv run python scripts/check.py          # format + lint + types + tests + benchmark units
uv run python scripts/check.py --fast   # skip integration/e2e and mypy
```

CI runs the same gates on Python 3.12/3.13 across Ubuntu and Windows; releases build all
packages via `uv build --all-packages` and publish to PyPI with trusted publishing.
Conventions, the ADR process, and the security-scan workflow are documented in
[docs/development/README.md](docs/development/README.md). Architecture decisions live in
[adr/](adr/); design principles in [design.md](design.md).

## License

MIT — see [LICENSE](LICENSE).
