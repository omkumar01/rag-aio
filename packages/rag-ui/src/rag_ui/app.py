"""Streamlit management console for rag-aio.

Run with::

    streamlit run src/rag_ui/app.py

Three tabs:

* **Config** — load/save a TOML config, edit every non-secret ``RAGConfig``
  field, validate through Pydantic ``model_validate`` (errors shown inline), and
  persist with the save button. Secret references (``api_key_ref``) are masked.
* **Dashboard** — read-only health/ingestion diagnostics fetched from the
  FastAPI service via :class:`~rag_ui.dashboard.Dashboard`. An unreachable
  backend is reported gracefully (never crashes the session).
* **Ask** — post a query to ``/v1/ask`` and render the answer + citations.

Streamlit is imported at module top-level here (this module is only loaded on
demand, never by ``import rag_ui``).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import streamlit as st
from rag_aio.config import RAGConfig

from rag_ui.config_view import (
    default_config_path,
    dict_to_toml,
    render_config_form,
    validate_config,
)
from rag_ui.dashboard import Dashboard

__all__ = ["API_URL", "main"]

# Default service root; overridable via the ``API_URL`` env var / ``st.secrets``.
API_URL: str = "http://localhost:8000"


def get_api_url() -> str:
    """Resolve the backend base URL from env or Streamlit secrets."""
    env_url = os.environ.get("API_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("API_URL")
    except Exception:
        secret_url = None
    if secret_url:
        return str(secret_url).rstrip("/")
    return API_URL


def _run_async(coro: Any) -> Any:
    """Run *coro* to completion on a fresh event loop.

    Wrapped so a stray running loop (or a backend hang) never crashes the
    Streamlit session: failures come back as an error dict instead.
    """
    try:
        return asyncio.run(coro)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "endpoint": "client"}


async def _dashboard_summary(dash: Dashboard) -> dict[str, Any]:
    """Fetch health first; only fan out the remaining endpoints when healthy."""
    health = await dash.health()
    if "error" in health:
        return {"healthy": False, "health": health}
    ready, metrics, pipelines = await asyncio.gather(
        dash.ready(), dash.metrics(), dash.list_pipelines()
    )
    return {
        "healthy": True,
        "health": health,
        "ready": ready,
        "metrics": metrics,
        "pipelines": pipelines,
    }


def _load_config(path: str | None) -> RAGConfig:
    if path:
        try:
            return RAGConfig.from_file(path)
        except FileNotFoundError:
            st.warning(f"Config file not found at {path!r}; using defaults.")
        except Exception as exc:
            st.warning(f"Could not load config from {path!r}: {exc}. Using defaults.")
    return RAGConfig()


def _config_tab(api_url: str) -> None:
    path = st.text_input(
        "Config file path", value=st.session_state.get("config_path", default_config_path())
    )
    if st.button("Load config"):
        st.session_state["config_path"] = str(path)
        st.rerun()

    config = _load_config(st.session_state.get("config_path"))

    data = render_config_form(config)
    cfg, error = validate_config(data)
    if error is not None:
        st.error(f"Invalid configuration:\n{error}")
        st.session_state["_validated_config"] = None
    else:
        st.session_state["_validated_config"] = cfg

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save config to TOML"):
            if error is None:
                try:
                    save_file = Path.cwd() / "rag_config.toml"
                    save_file.parent.mkdir(parents=True, exist_ok=True)
                    save_file.write_text(dict_to_toml(data), encoding="utf-8")
                    st.success(f"Saved config to {save_file}")
                except Exception as exc:
                    st.error(f"Could not save config: {exc}")
            else:
                st.warning("Fix validation errors before saving.")
    with col2:
        if st.button("Reset to defaults"):
            st.session_state.pop("config_path", None)
            st.rerun()


def _dashboard_tab(api_url: str) -> None:
    st.caption(f"Backend: `{api_url}`")
    dash = Dashboard(api_url, timeout=5.0)
    summary = _run_async(_dashboard_summary(dash))
    if not isinstance(summary, dict) or "healthy" not in summary:
        st.warning("Unexpected dashboard state; retrying may help.")
        return
    if not summary["healthy"]:
        st.warning(f"Backend unreachable at {api_url}.")
        st.json(summary.get("health", {}))
        return
    st.success("Backend healthy")
    health = summary.get("health", {})
    st.json({"health": health, "ready": summary.get("ready", {})})
    metrics = summary.get("metrics", {})
    if isinstance(metrics, dict):
        if "vector_points" in metrics:
            st.metric("Vector points", metrics.get("vector_points"))
        if isinstance(metrics.get("cache"), dict):
            st.metric("Cache hit rate", metrics["cache"].get("hit_rate"))
    pipelines = summary.get("pipelines", {})
    if isinstance(pipelines, dict) and pipelines.get("pipelines"):
        rows = []
        for p in pipelines["pipelines"]:
            cfg = p.get("config", {}) if isinstance(p, dict) else {}
            rows.append(
                {
                    "id": p.get("id") if isinstance(p, dict) else None,
                    "name": cfg.get("name"),
                    "schema_version": cfg.get("schema_version"),
                }
            )
        st.table(rows)
    else:
        st.caption("No pipelines registered.")

    st.divider()
    st.subheader("Job status")
    job_id = st.text_input("Job ID", value="", placeholder="lookup a job by id")
    if st.button("Lookup job") and job_id:
        result = _run_async(dash.list_jobs(job_id))
        if "error" in result:
            st.warning(f"Job {job_id!r}: {result.get('error')}")
        else:
            st.json(result)


def _ask_tab(api_url: str) -> None:
    dash = Dashboard(api_url, timeout=30.0)
    query = st.text_area("Query", value="", placeholder="Enter your question …")
    stream = st.checkbox("Stream response", value=False)
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        with st.spinner("Generating answer…"):
            if stream:
                result = _run_async(_collect_stream(dash, query, None))
                _render_stream(result)
            else:
                result = _run_async(dash.ask(query, stream=False))
                _render_answer(result)
    else:
        st.caption("Enter a query and click **Ask** to query the backend.")


def _render_answer(result: Any) -> None:
    if not isinstance(result, dict) or "error" in result:
        st.warning("Backend unreachable or request failed.")
        if isinstance(result, dict):
            st.json(result)
        return
    answer = result.get("answer", "")
    st.markdown(answer)
    citations = result.get("citations") or []
    if citations:
        st.subheader("Citations")
        st.write(citations)
    timings = result.get("timings_ms") or {}
    metrics = result.get("metrics") or {}
    st.caption(
        f"query_id={result.get('query_id')} | total_ms={timings.get('total_ms')} | "
        f"cached={metrics.get('cached')}"
    )


async def _collect_stream(
    dash: Dashboard, query: str, overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Consume a streaming ``ask`` on a single event loop and gather the text."""
    result = await dash.ask(query, stream=True, overrides=overrides)
    if not hasattr(result, "__aiter__"):
        return {"parts": [], "error": "unexpected non-streaming response"}
    parts: list[str] = []
    error: str | None = None
    async for delta in result:
        if isinstance(delta, dict):
            if "error" in delta:
                error = str(delta.get("error"))
                break
            parts.append(str(delta.get("delta", "")))
        else:
            parts.append(str(delta))
    return {"parts": parts, "error": error}


def _render_stream(result: Any) -> None:
    if not isinstance(result, dict):
        st.warning("Unexpected response; streaming may be unavailable.")
        return
    if result.get("error"):
        st.warning(f"Streaming failed: {result['error']}")
        return
    text = "".join(result.get("parts", []))
    st.markdown(text)
    if text:
        st.success("Stream complete.")


def main() -> None:
    api_url = get_api_url()
    st.set_page_config(page_title="rag-ui", page_icon="🤖", layout="wide")
    st.title("rag-aio console")
    st.caption(f"Backend: `{api_url}`")

    config_tab, dashboard_tab, ask_tab = st.tabs(["Config", "Dashboard", "Ask"])
    with config_tab:
        _config_tab(api_url)
    with dashboard_tab:
        _dashboard_tab(api_url)
    with ask_tab:
        _ask_tab(api_url)


if __name__ == "__main__":
    main()
