from __future__ import annotations

import subprocess
import sys

import rag_aio


def test_import() -> None:
    assert rag_aio.__version__ == "0.1.0"


def test_exports() -> None:
    assert hasattr(rag_aio, "RAG")
    assert hasattr(rag_aio, "RAGConfig")


def test_light_import_no_heavy_backends() -> None:
    """``import rag_aio`` must not pull in fastembed or qdrant_client.

    Runs in a fresh subprocess so test-session contamination from later
    ``RAG.from_config(...)`` calls does not affect the assertion.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rag_aio; "
            "bad = [m for m in ('fastembed', 'qdrant_client') if m in sys.modules]; "
            "assert not bad, f'heavy backends loaded by import rag_aio: {bad}'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import rag_aio pulled in heavy backends: {result.stderr.strip()}"
    )
