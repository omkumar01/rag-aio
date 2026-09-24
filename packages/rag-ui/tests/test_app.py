"""Streamlit AppTest suite for the rag-ui console.

Runs ``src/rag_ui/app.py`` through :class:`streamlit.testing.v1.AppTest`
(no browser). The backend is pointed at a dead port via ``API_URL`` so the
dashboard fetch fails fast (connection refused) and deterministically renders
the "unreachable" path without touching the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "rag_ui" / "app.py")


@pytest.fixture()
def app_test(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run the console once, isolated from any real backend."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:59999")
    at = AppTest.from_file(APP_PATH)
    at.run()
    return at


def test_app_runs_without_exception(app_test: AppTest) -> None:
    assert not app_test.exception
    assert bool(app_test.exception) is False


def test_app_has_three_tabs(app_test: AppTest) -> None:
    assert [t.label for t in app_test.tabs] == ["Config", "Dashboard", "Ask"]


def test_app_renders_title(app_test: AppTest) -> None:
    assert any("rag-aio console" in t.value for t in app_test.title)


def test_config_tab_has_form_fields(app_test: AppTest) -> None:
    tab = app_test.tabs[0]
    labels = (
        [w.label for w in tab.text_input]
        + [w.label for w in tab.number_input]
        + [w.label for w in tab.selectbox]
        + [w.label for w in tab.checkbox]
    )
    expected = [
        "Pipeline name",
        "Embedder backend",
        "Embedding dim",
        "Qdrant path",
        "DB URL",
        "Cache backend",
        "Provider",
        "Base URL",
        "Model",
        "Temperature",
        "Concurrency",
        "API key env var ref",
    ]
    for name in expected:
        assert name in labels, f"missing config field: {name}"
    # secret reference must be masked in the caption, never by value
    captions = " ".join(c.value for c in tab.caption)
    assert "masked" in captions.lower()


def test_dashboard_tab_reports_unreachable(app_test: AppTest) -> None:
    tab = app_test.tabs[1]
    warnings = [w.value for w in tab.warning]
    assert any("unreachable" in w.lower() for w in warnings)


def test_ask_tab_renders_query_widgets(app_test: AppTest) -> None:
    tab = app_test.tabs[2]
    assert [w.label for w in tab.text_area] == ["Query"]
    assert [b.label for b in tab.button]  # "Ask" button present
