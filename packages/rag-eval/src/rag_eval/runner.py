"""Evaluation runner: retrieve, compute metrics, aggregate results."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from rag_core.errors import EvaluationError
from rag_core.evaluation import EvaluationResult, MetricResult, QueryEvaluation
from rag_core.ids import config_hash

from .datasets import EvalCase
from .metrics import compute_metric

__all__ = ["EvalRunner", "compare"]

DEFAULT_METRICS: list[str] = [
    "recall@10",
    "precision@10",
    "mrr",
    "ndcg@10",
    "hit_rate@10",
]


def _parse_k(name: str, default: int | None = None) -> int | None:
    """Extract the ``@k`` suffix from a metric name, falling back to *default*."""
    _, _, k_str = name.partition("@")
    if not k_str:
        return default
    try:
        return int(k_str)
    except ValueError:
        return default


class EvalRunner:
    """Run retrieval evaluation over a set of :class:`EvalCase` objects.

    Parameters
    ----------
    retriever_factory
        Callable that takes an :class:`EvalCase`` and returns an awaitable
        resolving to an object with a ``hits`` attribute (each hit having a
        ``chunk_id``).
    metrics
        List of metric names.  ``base@k`` notation is supported (e.g.
        ``"recall@10"``); bare names fall back to the runner's ``k``.
    k
        Default cutoff used when a metric name has no explicit ``@k``.
    dataset
        Label recorded in :attr:`EvaluationResult.dataset`.
    """

    def __init__(
        self,
        retriever_factory: Callable[[EvalCase], Awaitable[Any]],
        metrics: list[str] | None = None,
        k: int = 10,
        dataset: str = "rag-eval",
    ) -> None:
        self._factory = retriever_factory
        self._metrics: list[str] = metrics if metrics is not None else DEFAULT_METRICS
        self._k = k
        self._dataset = dataset

    async def run(self, cases: list[EvalCase]) -> EvaluationResult:
        """Execute the evaluation suite and return an :class:`EvaluationResult`."""
        started = datetime.now(UTC)
        cfg_hash = config_hash({"metrics": self._metrics, "k": self._k})

        per_query: list[QueryEvaluation] = []
        successful: list[QueryEvaluation] = []
        failure_counts: dict[str, int] = {}

        for case in cases:
            t0 = time.monotonic()
            try:
                retriever: Any = await self._factory(case)
                hits: Any = getattr(retriever, "hits", []) if retriever is not None else []
                ranked_ids: list[str] = [str(getattr(h, "chunk_id", h)) for h in hits]
            except Exception:
                latency_ms = (time.monotonic() - t0) * 1000.0
                failure_counts["retrieval_error"] = failure_counts.get("retrieval_error", 0) + 1
                per_query.append(
                    QueryEvaluation(
                        query_id=case.query_id,
                        failure_class="retrieval_error",
                        metrics=[
                            MetricResult(
                                name="latency_ms",
                                value=round(latency_ms, 4),
                            )
                        ],
                    )
                )
                continue

            latency_ms = (time.monotonic() - t0) * 1000.0
            case_metrics: list[MetricResult] = []
            for name in self._metrics:
                value = compute_metric(name, case.relevant_ids, ranked_ids)
                k_val = _parse_k(name, self._k)
                case_metrics.append(
                    MetricResult(
                        name=name,
                        value=value,
                        k=k_val,
                        details={"latency_ms": round(latency_ms, 4)},
                    )
                )
            qe = QueryEvaluation(query_id=case.query_id, metrics=case_metrics)
            successful.append(qe)
            per_query.append(qe)

        if not successful:
            raise EvaluationError(
                "All cases failed during retrieval",
                code="all_retrieval_failed",
                details={"failure_counts": failure_counts, "total": len(cases)},
            )

        # Aggregate over successful cases only
        aggregated: list[MetricResult] = []
        for name in self._metrics:
            values: list[float] = []
            for qe in successful:
                for mr in qe.metrics:
                    if mr.name == name:
                        values.append(mr.value)
            avg = sum(values) / len(values) if values else 0.0
            aggregated.append(
                MetricResult(
                    name=name,
                    value=avg,
                    k=_parse_k(name, self._k),
                    details={"cases": len(values), "failures": dict(failure_counts)},
                )
            )

        return EvaluationResult(
            dataset=self._dataset,
            config_hash=cfg_hash,
            metrics=aggregated,
            per_query=per_query,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


def compare(result_a: EvaluationResult, result_b: EvaluationResult) -> dict[str, float]:
    """Compute per-metric deltas (result_b - result_a).

    Metrics present in only one result use 0.0 for the missing side.
    """
    a_map: dict[str, float] = {mr.name: mr.value for mr in result_a.metrics}
    b_map: dict[str, float] = {mr.name: mr.value for mr in result_b.metrics}
    all_names: set[str] = set(a_map) | set(b_map)
    return {name: b_map.get(name, 0.0) - a_map.get(name, 0.0) for name in all_names}
