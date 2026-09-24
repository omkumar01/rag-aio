"""Smoke tests for rag-ui: import, version, and the lightweight-import contract.

``import rag_ui`` must be cheap and must **not** pull in Streamlit (or any heavy
backend). Streamlit is only loaded on demand through the lazy ``rag_ui.app``
attribute. The Streamlit-laziness assertion runs in a fresh subprocess so the
test session's own Streamlit usage cannot contaminate ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import rag_ui


def test_version() -> None:
    assert rag_ui.__version__ == "0.1.0"


def test_import_does_not_pull_streamlit() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rag_ui; assert 'streamlit' not in sys.modules, "
            "'streamlit imported by import rag_ui'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_app_attribute_lazily_exposes_main() -> None:
    # Accessing the lazy ``app`` attribute loads the Streamlit module.
    module = rag_ui.app
    assert module.__name__ == "rag_ui.app"
    assert hasattr(module, "main")
    assert "streamlit" in sys.modules
