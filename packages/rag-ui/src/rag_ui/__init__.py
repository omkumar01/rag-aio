"""rag-ui: Streamlit management and experimentation console for rag-aio.

Importing :mod:`rag_ui` is intentionally lightweight — it does **not** import
Streamlit (or any heavy backend). Streamlit is only pulled in when the
console's :mod:`rag_ui.app` module is actually used (e.g. via ``streamlit run``
or the lazy ``rag_ui.app`` attribute).
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.1"


def __getattr__(name: str) -> Any:
    """Lazily expose submodules that carry optional/heavy dependencies."""
    if name == "app":
        import importlib

        return importlib.import_module("rag_ui.app")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["__version__"]
