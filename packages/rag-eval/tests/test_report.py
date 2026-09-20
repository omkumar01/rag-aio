"""Tests for report formatting."""

from __future__ import annotations

import pytest
from rag_core.evaluation import EvaluationResult, MetricResult, QueryEvaluation
from rag_eval.report import format_report

pytestmark = pytest.mark.unit


def _make_result(
    metrics: list[MetricResult],
    per_query: list[QueryEvaluation] | None = None,
    dataset: str = "test-dataset",
    config_hash: str | None = "abc123",
) -> EvaluationResult:
    return EvaluationResult(
        dataset=dataset,
        config_hash=config_hash,
        metrics=metrics,
        per_query=per_query or [],
    )


def test_report_contains_metric_names() -> None:
    result = _make_result(
        [
            MetricResult(name="recall@10", value=0.75),
            MetricResult(name="mrr", value=0.5),
            MetricResult(name="ndcg@10", value=0.65),
        ]
    )
    report = format_report(result)
    assert "recall@10" in report
    assert "mrr" in report
    assert "ndcg@10" in report
    assert "0.7500" in report
    assert "0.6500" in report


def test_report_contains_run_metadata() -> None:
    result = _make_result(
        [MetricResult(name="recall@10", value=0.5)],
        dataset="my-dataset",
        config_hash="deadbeef",
    )
    report = format_report(result)
    assert "my-dataset" in report
    assert "deadbeef" in report
    assert result.run_id in report


def test_report_no_nan() -> None:
    result = _make_result(
        [
            MetricResult(name="recall@10", value=float("nan")),
            MetricResult(name="mrr", value=0.5),
        ]
    )
    report = format_report(result)
    assert "nan" not in report.lower()
    assert "N/A" in report


def test_report_failure_counts() -> None:
    per_query = [
        QueryEvaluation(query_id="q1", metrics=[], failure_class="retrieval_error"),
        QueryEvaluation(query_id="q2", metrics=[], failure_class="retrieval_error"),
        QueryEvaluation(query_id="q3", metrics=[MetricResult(name="mrr", value=0.5)]),
    ]
    result = _make_result([MetricResult(name="mrr", value=0.5)], per_query=per_query)
    report = format_report(result)
    assert "retrieval_error" in report
    # count of failures (2) and total (3)
    assert "2" in report
    assert "3" in report


def test_report_no_failures() -> None:
    per_query = [
        QueryEvaluation(query_id="q1", metrics=[MetricResult(name="mrr", value=0.5)]),
    ]
    result = _make_result([MetricResult(name="mrr", value=0.5)], per_query=per_query)
    report = format_report(result)
    assert "Failed | 0" in report


def test_report_value_4dp() -> None:
    result = _make_result([MetricResult(name="recall@10", value=0.123456789)])
    report = format_report(result)
    assert "0.1235" in report  # rounded to 4dp


def test_report_inf_handled() -> None:
    result = _make_result([MetricResult(name="recall@10", value=float("inf"))])
    report = format_report(result)
    assert "nan" not in report.lower()
    assert "inf" not in report.lower()
