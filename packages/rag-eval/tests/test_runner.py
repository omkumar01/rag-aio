"""Tests for EvalRunner and compare."""

from __future__ import annotations

from typing import Any

import pytest
from rag_core.errors import EvaluationError
from rag_core.evaluation import EvaluationResult, MetricResult
from rag_eval.datasets import EvalCase
from rag_eval.runner import EvalRunner, compare

pytestmark = pytest.mark.unit


# --- fakes ------------------------------------------------------------------


class _FakeHit:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id


class _FakeRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self.hits: list[_FakeHit] = [_FakeHit(cid) for cid in ranking]


def _make_factory(ranking: list[str]):
    async def factory(case: EvalCase) -> _FakeRetriever:
        return _FakeRetriever(ranking)

    return factory


async def _failing_factory(case: EvalCase) -> Any:
    raise RuntimeError("simulated retrieval failure")


# --- fixture data -----------------------------------------------------------

_CASES = [
    EvalCase(
        query_id="q1",
        query="question one",
        relevant_ids=["a", "b"],
    ),
    EvalCase(
        query_id="q2",
        query="question two",
        relevant_ids=["c"],
    ),
]


# --- run with fixed rankings ------------------------------------------------


async def test_runner_basic_metrics_match_direct() -> None:
    ranking = ["c", "a", "d", "b"]
    runner = EvalRunner(retriever_factory=_make_factory(ranking))
    result = await runner.run(_CASES)

    assert len(result.per_query) == 2
    # q1: gt=[a,b], pred=[c,a,d,b]
    # recall@10 = 2/2 = 1.0, mrr = 0.5, hit_rate@10 = 1.0
    q1 = next(qe for qe in result.per_query if qe.query_id == "q1")
    m1 = {mr.name: mr.value for mr in q1.metrics}
    assert m1["recall@10"] == pytest.approx(1.0)
    assert m1["mrr"] == pytest.approx(0.5)
    assert m1["hit_rate@10"] == 1.0
    assert m1["ndcg@10"] == pytest.approx(0.6509, abs=1e-3)

    # q2: gt=[c], pred=[c,a,d,b] -> recall@10=1.0, mrr=1.0
    q2 = next(qe for qe in result.per_query if qe.query_id == "q2")
    m2 = {mr.name: mr.value for mr in q2.metrics}
    assert m2["recall@10"] == pytest.approx(1.0)
    assert m2["mrr"] == pytest.approx(1.0)

    # aggregated
    agg = {mr.name: mr.value for mr in result.metrics}
    assert agg["recall@10"] == pytest.approx(1.0)
    avg_mrr = (m1["mrr"] + m2["mrr"]) / 2
    assert agg["mrr"] == pytest.approx(avg_mrr)


async def test_runner_latency_recorded() -> None:
    runner = EvalRunner(retriever_factory=_make_factory(["a"]))
    result = await runner.run(_CASES)
    # latency_ms is stored in details of each per-case metric
    q1 = result.per_query[0]
    for mr in q1.metrics:
        assert "latency_ms" in mr.details
        assert mr.details["latency_ms"] >= 0.0


async def test_runner_dataset_and_config() -> None:
    runner = EvalRunner(retriever_factory=_make_factory(["a"]), dataset="my_dataset")
    result = await runner.run(_CASES)
    assert result.dataset == "my_dataset"
    assert result.config_hash is not None
    assert result.run_id is not None
    assert result.finished_at is not None


# --- failure handling -------------------------------------------------------


async def test_runner_retrieval_error_classified() -> None:
    """One failing case, one successful -> failure_class recorded on failed."""

    async def mixed_factory(case: EvalCase) -> Any:
        if case.query_id == "q1":
            raise RuntimeError("simulated failure")
        return _FakeRetriever(["c"])

    runner = EvalRunner(
        retriever_factory=mixed_factory,
        metrics=["recall@10", "mrr"],
    )
    result = await runner.run(_CASES)

    q1 = next(qe for qe in result.per_query if qe.query_id == "q1")
    q2 = next(qe for qe in result.per_query if qe.query_id == "q2")
    assert q1.failure_class == "retrieval_error"
    assert len(q1.metrics) == 1
    assert q1.metrics[0].name == "latency_ms"
    assert q2.failure_class is None


async def test_runner_all_failed_raises() -> None:
    runner = EvalRunner(
        retriever_factory=_failing_factory,
        metrics=["recall@10"],
    )
    with pytest.raises(EvaluationError, match="All cases failed"):
        await runner.run(_CASES)


async def test_runner_partial_failure_still_evaluates() -> None:
    """One failing case, one successful case -> metrics from successful only."""

    async def mixed_factory(case: EvalCase) -> Any:
        if case.query_id == "q1":
            raise RuntimeError("fail")
        return _FakeRetriever(["c"])

    runner = EvalRunner(
        retriever_factory=mixed_factory,
        metrics=["recall@10", "mrr"],
    )
    result = await runner.run(_CASES)

    q1 = next(qe for qe in result.per_query if qe.query_id == "q1")
    q2 = next(qe for qe in result.per_query if qe.query_id == "q2")
    assert q1.failure_class == "retrieval_error"
    assert q2.failure_class is None
    # q2: gt=[c], pred=[c] -> recall@10=1.0
    q2_metrics = {mr.name: mr.value for mr in q2.metrics}
    assert q2_metrics["recall@10"] == pytest.approx(1.0)
    # aggregated metrics based on 1 successful case
    agg = {mr.name: mr.value for mr in result.metrics}
    assert agg["recall@10"] == pytest.approx(1.0)


# --- compare ----------------------------------------------------------------


def test_compare_deltas() -> None:
    result_a = EvaluationResult(
        dataset="test",
        metrics=[
            MetricResult(name="recall@10", value=0.5),
            MetricResult(name="mrr", value=0.3),
        ],
        per_query=[],
    )
    result_b = EvaluationResult(
        dataset="test",
        metrics=[
            MetricResult(name="recall@10", value=0.75),
            MetricResult(name="mrr", value=0.4),
        ],
        per_query=[],
    )
    deltas = compare(result_a, result_b)
    assert deltas["recall@10"] == pytest.approx(0.25)
    assert deltas["mrr"] == pytest.approx(0.1)


def test_compare_missing_metric() -> None:
    result_a = EvaluationResult(
        dataset="test",
        metrics=[MetricResult(name="recall@10", value=0.5)],
        per_query=[],
    )
    result_b = EvaluationResult(
        dataset="test",
        metrics=[
            MetricResult(name="recall@10", value=0.8),
            MetricResult(name="mrr", value=0.6),
        ],
        per_query=[],
    )
    deltas = compare(result_a, result_b)
    assert deltas["recall@10"] == pytest.approx(0.3)
    assert deltas["mrr"] == pytest.approx(0.6)  # a has 0.0 for mrr


# --- default metrics --------------------------------------------------------


async def test_runner_default_metrics() -> None:
    runner = EvalRunner(retriever_factory=_make_factory(["a"]))
    result = await runner.run(_CASES)
    expected_names = {"recall@10", "precision@10", "mrr", "ndcg@10", "hit_rate@10"}
    actual_names = {mr.name for mr in result.metrics}
    assert expected_names == actual_names
