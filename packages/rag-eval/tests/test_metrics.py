"""Tests for ranking metrics."""

from __future__ import annotations

import math

import pytest
from rag_core.errors import EvaluationError
from rag_eval.metrics import (
    METRICS,
    average_precision,
    compute_metric,
    hit_rate,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)

# ground truth and predicted use single-char ids for readability
GT = ["a", "b"]
PRED = ["c", "a", "d", "b"]


# --- recall ----------------------------------------------------------------


def test_recall_at_k_partial() -> None:
    # top-3: [c, a, d] -> 1 hit out of 2 relevant
    assert recall_at_k(GT, PRED, 3) == pytest.approx(0.5)


def test_recall_at_k_full() -> None:
    # top-4: [c, a, d, b] -> 2 hits out of 2 relevant
    assert recall_at_k(GT, PRED, 4) == pytest.approx(1.0)


def test_recall_k_clamps_to_predicted_length() -> None:
    # k=10 but only 4 preds; recall@10 == recall@4
    assert recall_at_k(GT, PRED, 10) == pytest.approx(1.0)


def test_recall_default_k() -> None:
    # default k=10 > len(pred), so same as recall@4
    assert recall_at_k(GT, PRED) == pytest.approx(1.0)


def test_recall_empty_predicted() -> None:
    assert recall_at_k(GT, [], 5) == 0.0


def test_recall_empty_ground_truth() -> None:
    assert recall_at_k([], PRED, 5) == 0.0


def test_recall_no_hits() -> None:
    assert recall_at_k(["a", "b"], ["x", "y", "z"], 3) == pytest.approx(0.0)


# --- precision -------------------------------------------------------------


def test_precision_at_k() -> None:
    # top-3: [c, a, d] -> 1 correct out of 3
    assert precision_at_k(GT, PRED, 3) == pytest.approx(1 / 3)


def test_precision_at_k_4() -> None:
    # top-4: [c, a, d, b] -> 2 correct out of 4
    assert precision_at_k(GT, PRED, 4) == pytest.approx(0.5)


def test_precision_empty() -> None:
    assert precision_at_k(GT, [], 5) == 0.0
    assert precision_at_k([], PRED, 5) == 0.0


# --- hit rate ----------------------------------------------------------------


def test_hit_rate_hit() -> None:
    # top-2: [c, a] -> a is relevant
    assert hit_rate(GT, PRED, 2) == 1.0


def test_hit_rate_miss() -> None:
    # top-1: [c] -> c not relevant
    assert hit_rate(GT, PRED, 1) == 0.0


def test_hit_rate_empty() -> None:
    assert hit_rate(GT, [], 5) == 0.0
    assert hit_rate([], PRED, 5) == 0.0


# --- MRR -------------------------------------------------------------------


def test_mrr_first_hit_at_rank_2() -> None:
    # a appears at index 1 (rank 2) -> 1/2
    assert mrr(GT, PRED) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert mrr(GT, ["x", "y", "z"]) == 0.0


def test_mrr_empty() -> None:
    assert mrr([], PRED) == 0.0
    assert mrr(GT, []) == 0.0


def test_mrr_first_position() -> None:
    assert mrr(["a"], ["a", "b", "c"]) == 1.0


# --- average precision ------------------------------------------------------


def test_average_precision() -> None:
    # AP = (1/2 + 2/4) / 2 = (0.5 + 0.5) / 2 = 0.5
    assert average_precision(GT, PRED) == pytest.approx(0.5)


def test_average_precision_no_hits() -> None:
    assert average_precision(GT, ["x", "y", "z"]) == 0.0


def test_average_precision_empty() -> None:
    assert average_precision([], PRED) == 0.0
    assert average_precision(GT, []) == 0.0


# --- nDCG -------------------------------------------------------------------


def test_ndcg_basic() -> None:
    """gt=[a,b], pred=[c,a,d,b], default k=10.

    DCG = 1/log2(3) + 1/log2(5)
    IDCG = 1/log2(2) + 1/log2(3)
    nDCG ~ 0.6509
    """
    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = expected_dcg / expected_idcg
    assert ndcg(GT, PRED) == pytest.approx(expected, abs=1e-6)


def test_ndcg_at_3() -> None:
    # top-3: [c, a, d] -> only a at rank 2
    # DCG = 1/log2(3), IDCG = 1/log2(2) + 1/log2(3)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = expected_dcg / expected_idcg
    assert ndcg(GT, PRED, 3) == pytest.approx(expected, abs=1e-6)


def test_ndcg_perfect() -> None:
    # all relevant docs at the top -> nDCG = 1.0
    assert ndcg(["a", "b"], ["a", "b", "c"], 3) == pytest.approx(1.0)


def test_ndcg_empty() -> None:
    assert ndcg(GT, [], 5) == 0.0
    assert ndcg([], PRED, 5) == 0.0


def test_ndcg_k_clamps() -> None:
    # k larger than predicted -> uses all predictions
    assert ndcg(GT, PRED, 100) == ndcg(GT, PRED, 10)


# --- registry & compute_metric ---------------------------------------------


def test_metrics_registry_contains_all() -> None:
    for name in ("recall", "precision", "hit_rate", "mrr", "average_precision", "ndcg"):
        assert name in METRICS


def test_compute_metric_with_at_syntax() -> None:
    assert compute_metric("recall@3", GT, PRED) == pytest.approx(0.5)
    assert compute_metric("mrr", GT, PRED) == pytest.approx(0.5)
    assert compute_metric("hit_rate@2", GT, PRED) == 1.0


def test_compute_metric_unknown_raises() -> None:
    with pytest.raises(EvaluationError, match="Unknown metric"):
        compute_metric("nonexistent", GT, PRED)


def test_compute_metric_invalid_k() -> None:
    with pytest.raises(EvaluationError, match="Invalid k"):
        compute_metric("recall@abc", GT, PRED)


def test_compute_metric_passes_k() -> None:
    # bare "recall" with explicit k=3 should match "recall@3"
    assert compute_metric("recall", GT, PRED, k=3) == pytest.approx(0.5)
    # bare "recall" without k defaults to 10
    assert compute_metric("recall", GT, PRED) == pytest.approx(1.0)
