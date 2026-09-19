# rag-aio

The composition surface of the rag-aio platform: the `RAG` facade
(`RAG.from_config("config.toml")` → `await rag.ingest(...)` → `await rag.ask(...)`), a
Typer CLI (`rag-aio`), and the FastAPI service application (`/health`, `/ready`,
`/metrics`, `/version`, typed domain endpoints, job endpoints).
