"""Smoke tests for rag-mass-inject package importability."""

from __future__ import annotations

import rag_mass_inject
from rag_mass_inject import (
    JobTracker,
    MassIngestor,
    MassInjectConfig,
    StageWorker,
    discover_directory,
    discover_file_list,
    discover_sitemap,
    is_supported,
)


def test_import() -> None:
    assert rag_mass_inject.__version__ == "0.1.0"


def test_exports() -> None:
    assert MassIngestor is not None
    assert MassInjectConfig is not None
    assert JobTracker is not None
    assert StageWorker is not None
    assert callable(discover_directory)
    assert callable(discover_file_list)
    assert callable(discover_sitemap)
    assert callable(is_supported)
