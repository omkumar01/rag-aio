# rag-ui

Streamlit management and experimentation console for rag-aio. Provides a
fault-tolerant HTTP client (`Dashboard`) to the FastAPI backend, a
configuration editor that shares the **same** Pydantic `RAGConfig` models as the
backend (so the UI and service can never disagree on what is valid), and a
three-tab Streamlit app: **Config**, **Dashboard**, and **Ask**.

`import rag_ui` is intentionally lightweight — it does **not** import Streamlit
(or any heavy backend). Streamlit is only pulled in when `rag_ui.app` is
accessed (e.g. via `streamlit run` or `rag_ui.app` lazy attribute access).

## Installation

```bash
uv add -e packages/rag-ui
```

This pulls in `rag-aio` (which transitively provides `RAGConfig`, the `RAG`
facade, and `build_services`) plus `streamlit>=1.37` and `httpx>=0.27`.

> **Note:** `rag-aio` is a dependency, not `rag-orchestrator` directly. The UI
> consumes the orchestrator's FastAPI service over HTTP and shares its config
> models through the `rag-aio` facade.

## Architecture / Design Principles

```
                     ┌─────────────────────────────────────┐
                     │            rag-ui (this package)      │
                     │                                       │
  streamlit run ──▶  app.main()                              ├──► Dashboard (httpx client)
                     │   ├── _config_tab()                   │     /health, /ready, /metrics,
                     │   │   render_config_form()            │     /v1/pipelines, /v1/jobs,
                     │   │   validate_config()               │     /v1/ask, /v1/ingest
                     │   │   validate_save_path()            │
                     │   ├── _dashboard_tab()                │
                     │   └── _ask_tab()                      └──► config_view (TOML + masking)
                     │       _render_answer() / _render_stream()
                     └─────────────────────────────────────┘
                                    │ HTTP
                                    ▼
                     ┌─────────────────────────────────────┐
                     │         rag-orchestrator service      │
                     │   (created by rag_aio.app.get_app)    │
                     └─────────────────────────────────────┘
```

**Key design decisions:**

- **Validation parity.** The config form's data dict is validated through
  `RAGConfig.model_validate(data)` — the *exact same* Pydantic model the
  backend uses (via `rag_aio.config`). A field the UI accepts is always a field
  the backend accepts; a field the backend rejects is flagged inline in the
  Streamlit form before the user clicks "Save".
- **Lazy Streamlit import.** `__getattr__` on the package delegates `rag_ui.app`
  to `importlib.import_module("rag_ui.app")`. `config_view.py` imports Streamlit
  *inside* `render_config_form` rather than at module top level, so
  `import rag_ui.config_view` never triggers the Streamlit runtime.
- **Fault tolerance.** Every `Dashboard` method returns a `dict` on failure
  (`{"ok": False, "error": ..., "endpoint": ...}`) instead of raising. An
  unreachable backend, a connection refused, or a non-JSON response never
  crashes a Streamlit session — the error is rendered inline.
- **Secret-by-reference, always masked.** `RAGConfig.generation.api_key_ref`
  stores an environment-variable *name*, never a credential value. The config
  form renders it through `mask_secret()` (revealing only the last 4 characters)
  so the reference is never shown in full by accident.
- **Path safety.** `validate_save_path()` rejects `..` traversal segments,
  requires a `.toml` suffix, and confines writes to the current working
  directory or the user's home directory — so the Streamlit console cannot write
  to arbitrary filesystem locations.
- **Transport injection.** `Dashboard(transport=...)` accepts an
  `httpx.AsyncBaseTransport`, making every method testable with
  `httpx.MockTransport` — no real server needed.
- **SSE streaming.** `Dashboard.ask(query, stream=True)` returns an async
  iterator of decoded SSE `data:` lines, terminated by the `[DONE]` sentinel.
  The app collects these into text deltas and renders them as a streaming
  answer.

## Public API

```python
from rag_ui import __version__          # lazy; Streamlit not loaded

# Heavy module — loaded on demand (Streamlit pulled in only when accessed):
from rag_ui.app import main           # lazily imports streamlit
from rag_ui.dashboard import Dashboard
from rag_ui.config_view import (
    mask_secret,
    dict_to_toml,
    validate_config,
    validate_save_path,
    default_config_path,
    render_config_form,
)
```

### Dashboard — fault-tolerant HTTP client

```python
from rag_ui.dashboard import Dashboard

dash = Dashboard("http://localhost:8000", timeout=5.0)

health = await dash.health()              # {"status": "ok"} or {"ok": False, "error": ...}
ready  = await dash.ready()               # {"ready": True, "components": {...}}
metrics = await dash.metrics()            # {"pipeline": ..., "vector_points": N, "cache": {...}}
pipelines = await dash.list_pipelines()   # {"schema_version": ..., "pipelines": [...]}
job = await dash.list_jobs("job-id")      # pipeline job record or error dict
```

Every method returns a parsed JSON `dict` on success or an error `dict` (with
`ok=False`, `error`, `endpoint`, and optionally `status_code`) on any failure.
No exceptions cross the `Dashboard` boundary.

#### Ask (JSON and streaming)

```python
# Non-streaming — returns the full AskResult payload dict
result = await dash.ask("What are the auth requirements?")
print(result["answer"])
print(result["citations"])
print(result["timings_ms"])
print(result["metrics"])                   # includes "cached" flag

# With per-request overrides
result = await dash.ask("What is RAG?", overrides={"top_k": 5, "temperature": 0.5})

# Streaming — returns an async iterator of delta dicts
stream = await dash.ask("Explain RAG", stream=True)
async for delta in stream:
    if "error" in delta:
        break
    print(delta.get("delta", ""), end="", flush=True)
```

The streaming path consumes SSE `data:` lines, decodes each as JSON, and yields
it. The `[DONE]` sentinel terminates iteration.

#### Ingest

```python
result = await dash.ingest("/path/to/docs", recursive=True)
print(result)   # {"content_hash": "...", "pages": 3, ...} or error dict
```

### config_view — TOML serialization and secret masking

```python
from rag_aio.config import RAGConfig
from rag_ui.config_view import mask_secret, dict_to_toml, validate_save_path

# Mask a secret-by-reference env var name for display
mask_secret("OPENAI_API_KEY")     # "•••••_KEY"
mask_secret("ABCDE")              # "•••••"  (len <= keep+1)
mask_secret(None)                 # "<unset>"

# Serialize a config dict to TOML
data = RAGConfig.mock().model_dump(mode="json")
toml_text = dict_to_toml(data)

# Validate a save path (rejects traversal, requires .toml in CWD or home)
path = validate_save_path("./my_config.toml")   # returns Path or raises ValueError
```

### App entry points

```python
from rag_ui.app import main, get_api_url, API_URL

# Run the full Streamlit console:
main()

# Resolve the backend URL from env / st.secrets (with fallback to API_URL):
url = get_api_url()   # checks API_URL env var, then st.secrets, then constant
```

The console app renders three tabs:

| Tab       | Purpose                                                                 |
|-----------|-------------------------------------------------------------------------|
| **Config** | Load/save a TOML config file, edit every non-secret `RAGConfig` field via widgets, validate inline, persist to `rag_config.toml` in the CWD, reset to defaults. |
| **Dashboard** | Health/ready probes, pipeline metrics (vector points, cache hit rate), registered pipelines, and job-status lookup by ID. |
| **Ask** | Text input for a query, optional streaming toggle, answer rendering with citations and per-stage timings. |

## Usage Guides

### Beginner — run the console locally

```bash
# 1. Start the rag-aio backend (mock / offline by default)
uv run rag-aio serve
#    → uvicorn on http://127.0.0.1:8000

# 2. In another terminal, run the Streamlit console
uv run streamlit run packages/rag-ui/src/rag_ui/app.py
#    → opens http://localhost:8501
```

Or via the CLI:

```bash
uv run rag-aio ui
```

### Beginner — query the backend from Python

```python
import asyncio
from rag_ui.dashboard import Dashboard

async def main():
    dash = Dashboard("http://localhost:8000", timeout=10.0)

    # Check health
    health = await dash.health()
    if "error" in health:
        print(f"backend down: {health['error']}")
        return

    # Ask a question
    result = await dash.ask("What are the authentication requirements?")
    print(result["answer"])
    for cite in result.get("citations", []):
        print(f"  [{cite['citation_id']}] {cite.get('source_uri', '')}")

    # Look up a job
    job = await dash.list_jobs("some-job-id")
    print(f"status: {job.get('status')}")

asyncio.run(main())
```

### Intermediate — customize the backend URL

The `Dashboard` base URL and timeout are fully configurable:

```python
from rag_ui.dashboard import Dashboard

# Point at a remote host with a longer timeout
dash = Dashboard("https://rag.example.com", timeout=30.0)

# Or inject a custom transport (e.g. for tests or a proxy)
import httpx
dash = Dashboard(
    "http://localhost:8000",
    timeout=5.0,
    transport=httpx.AsyncHTTPTransport(retries=3),
)
```

You can also set `API_URL` as an environment variable before launching:

```bash
export API_URL="https://staging.rag.example.com"
uv run streamlit run packages/rag-ui/src/rag_ui/app.py
```

The app resolves the URL in order: `API_URL` env var → `st.secrets["API_URL"]`
→ `API_URL` module constant (default `http://localhost:8000`).

### Intermediate — stream a response

```python
from rag_ui.dashboard import Dashboard

dash = Dashboard("http://localhost:8000", timeout=30.0)

async def main():
    stream = await dash.ask("Explain retrieval-augmented generation", stream=True)
    async for delta in stream:
        if "error" in delta:
            print(f"\n[streaming error: {delta['error']}]")
            break
        print(delta.get("delta", ""), end="", flush=True)
    print()

asyncio.run(main())
```

### Advanced — test the Dashboard with MockTransport

```python
import httpx
from rag_ui.dashboard import Dashboard

def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/ask":
        return httpx.Response(200, json={"answer": "ok", "citations": [], "query_id": "q1"})
    return httpx.Response(404)

dash = Dashboard("http://testserver", 5.0, transport=httpx.MockTransport(handler))

health = await dash.health()
assert health == {"status": "ok"}
result = await dash.ask("hi")
assert result["answer"] == "ok"
```

### Advanced — secret masking and config validation

```python
from rag_aio.config import RAGConfig
from rag_ui.config_view import mask_secret, validate_config, dict_to_toml, validate_save_path

cfg = RAGConfig()
data = cfg.model_dump()

# Edit a field
data["generation"]["api_key_ref"] = "MY_OPENAI_KEY"
data["generation"]["temperature"] = 0.7

# Validate through the same model the backend uses
validated, error = validate_config(data)
if error:
    print(f"invalid: {error}")
else:
    print(f"valid: {validated.generation.temperature}")

# Mask for display
print(mask_secret("MY_OPENAI_KEY"))   # "•••••_KEY"

# Save path is safe (no traversal, .toml only, within CWD or home)
path = validate_save_path("./configs/run-1.toml")
path.write_text(dict_to_toml(data), encoding="utf-8")
```

## Configuration

The UI shares `RAGConfig` (and its sub-models) with the `rag-aio` backend. The
config has four sections, all editable through the Streamlit form:

| Section      | Fields (highlights)                                             |
|--------------|------------------------------------------------------------------|
| **pipeline** | `name`, `schema_version`, stage enables/strategies/policies, `timeout_s`, `token_budget`, `cost_budget_usd`, `concurrency`, `cache_reads`, `cache_writes`. |
| **embedder** | `backend` (`"mock"` / `"fastembed"`), `dense_model`, `sparse_model`, `dim`. |
| **storage**  | `qdrant_path`, `db_url`, `cache_backend` (`"memory"` / `"sqlite"`), `cache_path`. |
| **generation** | `provider`, `base_url`, `model`, `api_key_ref` (env-var name, masked), `temperature`, `max_tokens`. |

Load/save uses TOML (`tomllib` for reading; a minimal stdlib writer handles
saving). The save button writes `rag_config.toml` into the CWD.

Backend connection is configured via the `API_URL` environment variable or
Streamlit `st.secrets`. The `Dashboard` `timeout` (default `10.0s`, `5.0s` in
the Dashboard tab, `30.0s` in the Ask tab) controls per-request deadlines.

## Testing

```bash
uv run pytest packages/rag-ui -q
```

| Test file       | Scope | What it covers                                               |
|-----------------|-------|--------------------------------------------------------------|
| `test_smoke.py` | unit  | Import, `__version__`, **lightweight-import contract** (subprocess assertion that `import rag_ui` does not load Streamlit), lazy `rag_ui.app` attribute. |
| `test_dashboard.py` | unit | All `Dashboard` endpoints via `httpx.MockTransport`: JSON + SSE streaming, error dicts, HTTP status errors, unreachable backend. |
| `test_config.py` | unit | `RAGConfig` round-trips (Pydantic + TOML), `mask_secret` tail-reveal, `validate_save_path` rejection of traversal / non-`.toml` / outside-CWD-home. |
| `test_app.py`  | unit | Streamlit `AppTest` harness: three tabs render, config form fields present, secrets masked, dashboard reports unreachable gracefully. |

## Dependencies

| Dependency | Role                                                        |
|------------|-------------------------------------------------------------|
| `rag-aio`  | Provides `RAGConfig` (shared Pydantic config models), the `RAG` facade, and `build_services`. Also re-exports `OrchestratorServices` from `rag-orchestrator`. |
| `streamlit>=1.37` | The management console runtime (imported lazily).                |
| `httpx>=0.27` | Async HTTP client for all `Dashboard` calls; `MockTransport` for tests. |

No direct dependency on `rag-core`, `rag-orchestrator`, `rag-retrieval`, etc.
— those are reached transitively through `rag-aio` and over HTTP at runtime.

## Cross-Package Relationships

- **rag-aio** — The UI imports `RAGConfig` and its sub-models directly from
  `rag_aio.config`, guaranteeing that the config form and the backend share
  the same validation schema. The `RAG` facade (`rag_aio.facade`) is what
  produces the `OrchestratorServices` bag that backs the FastAPI service.
- **rag-orchestrator** — The backend FastAPI app is created by
  `rag_orchestrator.app.app.create_app(services, pipeline_config)`. This
  exposes the endpoints the `Dashboard` client calls:
  `/health`, `/ready`, `/metrics`, `/version`, `/v1/pipelines`, `/v1/ask`
  (JSON + SSE streaming), `/v1/ingest`, and `/v1/jobs/{job_id}`.
- **rag-core** — `Citation`, `Context`, and `AskResult` (from
  `rag_orchestrator.orchestrator`) are the models serialized in the `/v1/ask`
  JSON response. The UI's Ask tab renders `result["citations"]` and
  `result["timings_ms"]` directly.
- **rag-mass-inject** — The backend's `/v1/ingest` endpoint calls
  `rag_orchestrator.ingest.ingest()` for single-file ingestion. Bulk ingestion
  via `MassIngestor` is wired through the same `OrchestratorServices` bag; a
  mass-inject job's `JobTracker` SQLite DB can be inspected independently or
  surfaced through a future Dashboard page.
- **rag-observe** — The `Observer` protocol on `OrchestratorServices` receives
  `mass_inject.*` and `ingest.*` events; the backend's `/metrics` endpoint
  surfaces aggregate counters (vector points, cache hit rate) that the
  Dashboard tab renders as `st.metric` widgets.

See also [architecture.md](../../architecture.md) and the
[FastAPI service contract](../../packages/rag-orchestrator/src/rag_orchestrator/app/app.py).
