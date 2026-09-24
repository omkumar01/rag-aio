"""Smoke test: import and check key public API surface."""

import rag_eval


def test_version() -> None:
    assert rag_eval.__version__ == "0.1.1"


def test_exports() -> None:
    from rag_eval import (
        METRICS,
        compute_metric,
        hit_rate,
        recall_at_k,
    )

    assert callable(compute_metric)
    assert "recall" in METRICS
    assert callable(recall_at_k)
    assert callable(hit_rate)
