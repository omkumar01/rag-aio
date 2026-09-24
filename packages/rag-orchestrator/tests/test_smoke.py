import rag_orchestrator


def test_import() -> None:
    assert rag_orchestrator.__version__ == "0.1.1"


def test_import_does_not_pull_heavy_backends() -> None:
    """``import rag_orchestrator`` must stay dependency-light.

    Heavy backends (fastembed, qdrant_client, fastapi) are imported lazily inside
    ``load_local_services`` or the ``app`` submodule, so a fresh subprocess import
    must not pull them into ``sys.modules``.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rag_orchestrator; "
            "heavy = ['fastembed', 'qdrant_client', 'fastapi', 'httpx']; "
            "loaded = [m for m in heavy if m in sys.modules]; "
            "assert not loaded, f'heavy backends loaded: {loaded}'; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
