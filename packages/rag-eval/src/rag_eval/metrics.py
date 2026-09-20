"""Pure ranking metrics for retrieval evaluation.

Every function follows the signature ``(ground_truth, predicted, k) -> float``
where ``ground_truth`` is the list of relevant document ids and ``predicted``
is the ranked list of retrieved document ids.

Conventions
-----------
- Empty ``predicted``  -> 0.0  (no retrieval, no credit)
- Empty ``ground_truth`` -> 0.0  (nothing to retrieve, no credit)
- ``k`` defaults to 10 for recall/precision/hit_rate/ndcg.
- ``k`` is accepted but ignored for mrr and average_precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from rag_core.errors import EvaluationError

__all__ = [
    "METRICS",
    "MetricFunc",
    "average_precision",
    "compute_metric",
    "hit_rate",
    "mrr",
    "ndcg",
    "precision_at_k",
    "recall_at_k",
]

#: Type alias for a ranking metric function.
MetricFunc = Callable[[list[str], list[str], "int | None"], float]

_DEFAULT_K: int = 10


def recall_at_k(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """Recall@k: fraction of relevant docs found in the top-*k* predictions."""
    if k is None:
        k = _DEFAULT_K
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for p in top_k if p in relevant)
    return hits / len(relevant)


def precision_at_k(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """Precision@k: fraction of top-*k* predictions that are relevant."""
    if k is None:
        k = _DEFAULT_K
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for p in top_k if p in relevant)
    return hits / len(top_k)


def hit_rate(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """Hit rate@k: 1.0 if any relevant doc appears in the top-*k*, else 0.0."""
    if k is None:
        k = _DEFAULT_K
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    top_k = predicted[:k]
    return 1.0 if any(p in relevant for p in top_k) else 0.0


def mrr(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first relevant doc.

    ``k`` is accepted for registry compatibility but ignored.
    """
    _ = k  # explicitly unused
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    for i, p in enumerate(predicted):
        if p in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """Average Precision: mean of precision values at each relevant hit.

    ``k`` is accepted for registry compatibility but ignored.
    """
    _ = k  # explicitly unused
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    hits = 0
    sum_prec = 0.0
    for i, p in enumerate(predicted):
        if p in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant)


def ndcg(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float:
    """nDCG with binary relevance gains and idealised DCG.

    DCG uses the standard ``1 / log2(rank + 1)`` discount (0-indexed: ``1/log2(i+2)``).
    IDCG is computed over ``min(|relevant|, k)`` positions.
    """
    if k is None:
        k = _DEFAULT_K
    if not ground_truth or not predicted:
        return 0.0
    relevant = set(ground_truth)
    top_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(top_k):
        if p in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


#: Registry mapping metric names to their implementations.
METRICS: dict[str, MetricFunc] = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate,
    "mrr": mrr,
    "average_precision": average_precision,
    "ndcg": ndcg,
}


def compute_metric(
    name: str,
    ground_truth: list[str],
    predicted: list[str],
    k: int | None = None,
) -> float:
    """Compute a metric by name, parsing ``"base@k"`` notation.

    Examples
    --------
    >>> compute_metric("recall@10", ["a"], ["a", "b"])
    1.0
    >>> compute_metric("mrr", ["a"], ["b", "a"])
    0.5
    """
    base, sep, k_str = name.partition("@")
    func = METRICS.get(base)
    if func is None:
        raise EvaluationError(
            f"Unknown metric: {base!r}",
            code="unknown_metric",
            details={"name": name, "available": list(METRICS.keys())},
        )
    if sep and k_str:
        try:
            k = int(k_str)
        except ValueError as exc:
            raise EvaluationError(
                f"Invalid k value in metric name: {name!r}",
                code="invalid_k",
            ) from exc
    return func(ground_truth, predicted, k)
