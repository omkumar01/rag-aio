# rag-aio Examples

Small, runnable scripts that each demonstrate one slice of the platform.
Run them from the repository root with `uv run python examples/<script>.py`.

## Quickstart

| Example | What it shows | Requirements |
| --- | --- | --- |
| [`quickstart.py`](quickstart.py) | The README flow: `RAG.from_config("<file>.toml")` (TOML path form) → `ingest_directory` → `ask` with answer, citations, and per-stage timings. Uses the fully-offline mock profile from [`quickstart_config.toml`](quickstart_config.toml) (`embedder.backend = "mock"`, `pipeline.embedding.policy = "mock"`). | None — fully offline |

## Full local stack

| Example | What it shows | Requirements |
| --- | --- | --- |
| [`fully_local.py`](fully_local.py) | End-to-end with the *real* components — FastEmbed dense + sparse, Qdrant embedded local mode, SQLite, LM Studio — via `rag_orchestrator.load_local_services`. Auto-detects LM Studio at startup and exits with a clear message when unreachable. | LM Studio (or any OpenAI-compatible server) on `http://localhost:1234/v1`; first run downloads the embedding model |
| [`openai_compatible.py`](openai_compatible.py) | Direct use of `rag_generation.OpenAICompatibleProvider` and `GenerationService` (retry/fallback, usage, streaming) against any OpenAI-compatible endpoint. Configured via env vars: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`. Auto-skips when the endpoint is unreachable. | OpenAI-compatible server (default LM Studio on `localhost:1234`) |

## Per-stage building blocks

Each script is standalone and prints a header describing what it demonstrates.

| Example | What it shows | Requirements |
| --- | --- | --- |
| [`stages/ingest.py`](stages/ingest.py) | Parse a Markdown file with `IngestionPipeline`, then chunk it with the recursive chunker and `SimpleTokenizer`; prints chunk stats. | Fully offline |
| [`stages/retrieve.py`](stages/retrieve.py) | Index mock-embedded chunks into `InMemoryVectorStore`, then run `DenseRetriever` and a `HybridRetriever` (dense + `BM25Retriever` fused with `ReciprocalRankFusion`); prints top hits with scores. | Fully offline |
| [`stages/rerank.py`](stages/rerank.py) | Rerank a deliberately mis-ordered candidate list with `HeuristicReranker`; prints before/after ordering with original ranks preserved (`to_rerank_hits`). | Fully offline |
| [`stages/context.py`](stages/context.py) | Build a token-budgeted context from hits with `ContextBuilderImpl`; prints the rendered context (`render_context`) and citation records (`build_citations`). | Fully offline |
| [`stages/generate.py`](stages/generate.py) | Send a context-bearing `GenerationRequest` to an OpenAI-compatible endpoint (`OpenAICompatibleProvider`), non-streaming and streamed. | LM Studio / OpenAI-compatible server; auto-skips when unreachable |

## Notes

- The offline examples use deterministic mock embeddings (hash-based), so
  retrieval scores are stable but not semantically meaningful — they
  demonstrate the mechanics of each stage.
- Endpoint-based examples read configuration from the environment only;
  no secrets are hardcoded. `LM_STUDIO_URL` / `OPENAI_BASE_URL` override the
  default `http://localhost:1234/v1`.
