"""Streamlit helpers for editing :class:`~rag_aio.config.RAGConfig`.

This module is the *view* half of the config console. It knows how to render
editable widgets for every non-secret ``RAGConfig`` field, mask secret
references for display, validate a widget-derived dict back through Pydantic
(``model_validate``), and serialize a config dict to TOML for persistence.

Streamlit is imported lazily (inside :func:`render_config_form`) so that
``import rag_ui.config_view`` itself stays lightweight and never pulls in the
Streamlit runtime — mirroring the contract of :mod:`rag_ui` itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_aio.config import RAGConfig

__all__ = [
    "dict_to_toml",
    "mask_secret",
    "render_config_form",
    "validate_config",
    "validate_save_path",
]


def mask_secret(value: str | None, keep: int = 4) -> str:
    """Render a secret-by-reference for display, revealing only the tail.

    ``api_key_ref`` holds an *environment-variable name*, not a credential
    value, but it is still masked on screen so the reference is never shown in
    full by accident. Mirrors the ``mask_secret`` style used in
    :mod:`rag_llm_provider.secrets` but uses a bullet gutter.

    Examples
    --------
    >>> mask_secret("MY_API_KEY")
    '•••••_KEY'
    >>> mask_secret("ab")
    '••'
    >>> mask_secret(None)
    '<unset>'
    """
    if not value:
        return "<unset>"
    if keep <= 0:
        return "•" * 4
    if len(value) <= keep:
        return "•" * len(value)
    return "•••••" + value[-keep:]


# --------------------------------------------------------------------------- #
# TOML serialization (stdlib tomllib only reads; we need a minimal writer)
# --------------------------------------------------------------------------- #


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return '""'
    return json.dumps(str(value))


def _dict_to_toml(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Serialize a plain ``dict`` (from ``model_dump(mode="json")``) to TOML lines."""
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, val in data.items():
        if isinstance(val, dict):
            if val:
                tables.append((key, val))
        elif val is not None:
            scalars.append((key, val))
    for key, val in scalars:
        lines.append(f"{key} = {_toml_scalar(val)}")
    for key, val in tables:
        full_path = f"{prefix}.{key}" if prefix else key
        lines.append("")
        lines.append(f"[{full_path}]")
        lines.extend(_dict_to_toml(val, full_path))
    return lines


def dict_to_toml(data: dict[str, Any]) -> str:
    """Render a config dict as a TOML string."""
    return "\n".join(_dict_to_toml(data))


# --------------------------------------------------------------------------- #
# Form helpers
# --------------------------------------------------------------------------- #

_OPT_PLACEHOLDER = ""


def _select_index(value: str, options: list[str]) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _opt_str(label: str, value: str | None, **kwargs: Any) -> str | None:
    import streamlit as st

    current = value if value is not None else _OPT_PLACEHOLDER
    raw = st.text_input(label, value=current, **kwargs)
    return raw or None


def _opt_int(label: str, value: int | None, **kwargs: Any) -> int | None:
    import streamlit as st

    raw = st.text_input(
        label, value=str(value) if value is not None else _OPT_PLACEHOLDER, **kwargs
    )
    text = raw.strip()
    return int(text) if text else None


def _opt_float(label: str, value: float | None, **kwargs: Any) -> float | None:
    import streamlit as st

    raw = st.text_input(
        label, value=repr(value) if value is not None else _OPT_PLACEHOLDER, **kwargs
    )
    text = raw.strip()
    return float(text) if text else None


def render_config_form(config: RAGConfig, *, section: str = "") -> dict[str, Any]:
    """Render editable widgets for every non-secret ``RAGConfig`` field.

    Returns a *complete* nested dict (with defaults filled in from ``config``)
    that is ready to be handed to :func:`validate_config` /
    :meth:`RAGConfig.model_validate`. Widgets are rendered with stable,
    field-derived labels so the Streamlit testing harness and users can locate
    them.
    """
    import streamlit as st

    data: dict[str, Any] = config.model_dump()
    # model_dump() returns plain dicts of primitives; safe to mutate.
    pipeline: dict[str, Any] = data["pipeline"]
    embedder: dict[str, Any] = data["embedder"]
    storage: dict[str, Any] = data["storage"]
    generation: dict[str, Any] = data["generation"]

    st.subheader(section or "Pipeline")
    pipeline["name"] = st.text_input("Pipeline name", value=str(pipeline["name"]))
    pipeline["description"] = _opt_str("Pipeline description", pipeline.get("description"))
    pipeline["timeout_s"] = _opt_float("Timeout (s)", pipeline.get("timeout_s"))
    pipeline["token_budget"] = _opt_int("Token budget", pipeline.get("token_budget"))
    pipeline["cost_budget_usd"] = _opt_float("Cost budget (USD)", pipeline.get("cost_budget_usd"))
    pipeline["concurrency"] = st.number_input(
        "Concurrency", min_value=1, value=int(pipeline["concurrency"])
    )
    pipeline["cache_reads"] = st.checkbox("Cache reads", value=bool(pipeline["cache_reads"]))
    pipeline["cache_writes"] = st.checkbox("Cache writes", value=bool(pipeline["cache_writes"]))

    st.subheader("Embedder")
    backend = st.selectbox(
        "Embedder backend",
        ["mock", "fastembed"],
        index=_select_index(embedder["backend"], ["mock", "fastembed"]),
    )
    embedder["backend"] = backend
    embedder["dense_model"] = _opt_str("Dense model", embedder.get("dense_model"))
    embedder["sparse_model"] = _opt_str("Sparse model", embedder.get("sparse_model"))
    embedder["dim"] = st.number_input("Embedding dim", min_value=1, value=int(embedder["dim"]))

    st.subheader("Storage")
    storage["qdrant_path"] = st.text_input("Qdrant path", value=str(storage["qdrant_path"]))
    storage["db_url"] = st.text_input("DB URL", value=str(storage["db_url"]))
    storage["cache_backend"] = st.selectbox(
        "Cache backend",
        ["memory", "sqlite"],
        index=_select_index(storage["cache_backend"], ["memory", "sqlite"]),
    )
    storage["cache_path"] = _opt_str("Cache path", storage.get("cache_path"))

    st.subheader("Generation")
    generation["provider"] = st.text_input("Provider", value=str(generation["provider"]))
    generation["base_url"] = st.text_input("Base URL", value=str(generation["base_url"]))
    generation["model"] = _opt_str("Model", generation.get("model"))
    current_ref = generation.get("api_key_ref")
    st.caption(f"API key env var ref (masked): {mask_secret(current_ref)}")
    override_ref = st.text_input(
        "API key env var ref", value=_OPT_PLACEHOLDER, placeholder="enter env var name to override"
    )
    generation["api_key_ref"] = override_ref or current_ref
    generation["temperature"] = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(generation["temperature"]),
        step=0.05,
    )
    generation["max_tokens"] = _opt_int("Max tokens", generation.get("max_tokens"))

    return data


def validate_config(data: dict[str, Any]) -> tuple[RAGConfig | None, str | None]:
    """Validate a widget-derived dict via :meth:`RAGConfig.model_validate`.

    Returns ``(config, None)`` on success or ``(None, error_message)`` on
    validation failure. A broad ``except`` is intentional here: the console
    should surface *any* problem as a message rather than crash the Streamlit
    session.
    """
    try:
        cfg = RAGConfig.model_validate(data)
    except Exception as exc:
        return None, str(exc)
    return cfg, None


def default_config_path() -> str:
    """A sane default on-disk location for a user-authored config file."""
    return str(Path.cwd() / "rag_config.toml")


def validate_save_path(path: str) -> Path:
    """Validate that *path* is safe for writing (no path traversal).

    The resolved path must be within the current working directory or the
    user's home directory and must end with a ``.toml`` extension. This
    prevents the Streamlit config console from writing to arbitrary locations
    (e.g. ``../../etc/cron.d/x`` or ``/etc/passwd``).

    Returns the resolved :class:`~pathlib.Path` on success.

    Raises :class:`ValueError` if the path is empty, lacks a ``.toml``
    suffix, or escapes both the CWD and home directories.
    """
    if not path or not path.strip():
        msg = "path is empty"
        raise ValueError(msg)
    # Reject explicit traversal attempts before resolution — a resolved
    # path that lands inside an allowed dir (e.g. home) via ../ is still
    # suspect and not what the console should permit.
    if ".." in Path(path).parts:
        msg = "path traversal is not allowed"
        raise ValueError(msg)
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".toml":
        msg = f"file must end with .toml (got {resolved.suffix!r})"
        raise ValueError(msg)
    for base in (Path.cwd(), Path.home()):
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue
    msg = f"path must be within {Path.cwd()} or {Path.home()}"
    raise ValueError(msg)
