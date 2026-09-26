# web-example — Web UI over rag-aio

`web-example/` is a standalone example application (not a workspace member and
not part of the published wheel) that composes the rag-aio packages into a
multi-page web app: a ChatGPT-style chat page, a document explorer, a
mass-upload/injection console, a settings console, and cookie-session
authentication.

## Stack

- **FastAPI + Jinja2 + HTMX + Alpine.js** — server-rendered pages with
  HTMX partials for interactivity; no Node toolchain. `htmx.min.js` and
  `alpine.min.js` are vendored under `web-example/src/web_example/static/vendor/`.
- **SQLAlchemy (async) + aiosqlite** for the web database (accounts, settings,
  chat history, the document registry, bulk-job tracking). RAG content itself
  lives in the vector/document stores configured through rag-aio.
- **bcrypt** password hashes; **Starlette `SessionMiddleware`** signed-cookie
  sessions (`itsdangerous`). The signing key comes from `WEB_SESSION_SECRET`
  or is generated once and stored under the data directory.

## Running

From the repo root (the example runs inside the workspace environment; the
frontend deps live in the `dev` dependency group):

```bash
uv sync
uv run uvicorn web_example.main:app --reload --app-dir web-example/src
```

Then open http://127.0.0.1:8000. The first registered account becomes the
admin. Environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `WEB_EXAMPLE_DATA_DIR` | Data directory (sqlite web DB, uploads, session key, global RAG config) | `./data` |
| `WEB_SESSION_SECRET` | Session signing key (otherwise generated + persisted) | — |

## Pages

| Route | Page | Access |
| --- | --- | --- |
| `/` | Landing: project overview + package catalogue | Public (the only unauthenticated page) |
| `/login`, `/signup`, `/logout` | Authentication | Public |
| `/chat` | ChatGPT-style chat with SSE streaming answers and attachments | Authenticated |
| `/documents` | Document explorer: list, preview, delete, promote to global | Authenticated |
| `/upload` | Mass upload / bulk injection (files, directory, URL/sitemap) with live job progress | Authenticated |
| `/settings` | Providers/models for chat, embedding, OCR, rerank; retrieval mode (hybrid vs dense); storage & DB credentials | Authenticated (global section admin-only) |

## Document visibility model

Every document is registered with a visibility:

- **`user`** — ingested into the owner's vector-store namespace (`user_{id}`),
  visible and manageable only by them.
- **`global`** — ingested into the shared `global` namespace, visible to every
  user; only admins can promote/delete global documents.

Chat retrieval fans out to both pools concurrently via a
`CompositeVectorStore` (namespace-scoped searches merged by score), so answers
draw on the user's own corpus and the shared corpus at once. Both the mock
in-memory backend and Qdrant scope by namespace, so the same mechanism works
offline and in production wiring.

## Settings model

- **Global defaults** (admin-only) store a validated `RAGConfig`-shaped JSON
  document (generation, embedding, storage sections) **plus web-only rerank
  and OCR sections**.
- **Per-user overrides** are partial documents merged over the global defaults
  before validation; blank fields inherit the global value.
- Secrets are stored **by reference** (env-var names in `api_key_ref`
  fields), never as values — matching ADR-0004/0006.

Every setting is applied where it actually matters
(`web_example.services.local_services.build_pool_services`):

- **Embedding** — three backends: `mock` (deterministic placeholder, offline),
  `fastembed` (local ONNX models with the configured dense/sparse model ids and
  dimension), or `remote` — an OpenAI-compatible `/embeddings` endpoint
  (LM Studio, Ollama, vLLM, OpenAI) via
  :class:`rag_embedder.OpenAICompatibleEmbedder`, with `base_url`, model id
  (`dense_model`), `api_key_ref`, and a required explicit dimension (the vector
  store is sized before the first response). The same embedder instance embeds
  documents at index time and queries at search time. The `mock` backend has
  no semantic similarity — retrieval quality requires `fastembed` or `remote`.
- **Rerank** — heuristic, local cross-encoder (`sentence-transformers`), or a
  remote OpenAI-compatible rerank endpoint (model + `api_key_ref`).
- **OCR** — engine `vlm` wires mechanical-OCR → VLM escalation into the PDF
  parser's OCR fallback (endpoint URL, model, `api_key_ref`); `none` skips OCR.
- **Generation** — provider base URL and `api_key_ref`; the chosen model,
  temperature, and max tokens are mirrored into the pipeline's generation
  stage so every chat request uses them.
- **Storage** — local Qdrant path or server URL + `api_key_ref` (vector size
  follows the embedder dimension), SQL document-store URL, cache backend.

Per-user RAG instances are built lazily from the resolved settings and cached;
saving settings invalidates the affected user's cached pipeline. Changing the
embedding model or dimension is handled safely:

- The Qdrant collection name is **dimension-scoped**
  (`rag_chunks_d384`, `rag_chunks_d1024`, …) — a new collection is created at
  the right size instead of querying a mismatched one. Old collections are
  left intact; **re-ingest your documents** after a model/dimension switch so
  they become searchable again.
- The stage caches (notably the ingestion source cache) are **namespaced by
  embedder identity** (backend + model + dimension), so re-ingesting after a
  switch re-parses and re-embeds instead of returning stale cached documents.
- Chat failures surface as friendly SSE error events — e.g. a dimension
  mismatch yields "Re-ingest your documents after changing the embedding
  model" rather than a raw numpy traceback.

## Tests

```bash
uv run pytest web-example/tests -q
```

The suite runs fully offline against the mock profile (`RAGConfig.mock()`):
in-memory vector store, mock embedders, and a stub generator, with
`httpx.AsyncClient` + ASGI transport exercising the real HTTP stack.

## Mass ingestion

Directory and sitemap-URL jobs run through the real
:class:`rag_mass_inject.MassIngestor` pipeline (bounded queue, per-stage
concurrency, sqlite checkpoints under `<data dir>/checkpoints/<pool>`), bound
to the target pool's services so indexed chunks land in the correct
namespace. Two adapters make web sources work end to end:

- `SchemeRouterLoader` dispatches http(s) sources to the SSRF-guarded
  `WebLoader` and local paths to `FileLoader` (the stock `default_pipeline()`
  is file-only).
- `MimeAwareRegistry` maps response MIME types (`text/html`, `text/plain`,
  `application/pdf`, …) onto the parser registry's extension keys, since URLs
  like `https://example.com/` carry no extension.

Every successfully ingested document — chat attachments, uploaded files,
directory scans, sitemap URLs, and single pages — is mirrored into the web
document registry (`documents` table) so the explorer lists it with preview,
delete, and (for admins) promote-to-global.

## Limitations

- The example is intentionally simple: no CSRF token beyond SameSite=Lax
  cookies, no rate limiting, no HTTPS enforcement — it demonstrates
  composition of the rag-aio packages, not production hardening.
- MassIngestor discovers *all* files under a submitted directory and
  dead-letters unsupported ones; unsupported entries count toward the job
  total.
- Promoting a document to global updates the registry; re-indexing its
  vectors into the global namespace is left to the admin re-ingest flow.
