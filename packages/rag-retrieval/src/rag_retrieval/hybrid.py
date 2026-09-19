"""Hybrid retrieval: concurrent strategy execution with pluggable fusion."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from rag_core.errors import RetrievalError
from rag_core.protocols import Retriever
from rag_core.queries import Query
from rag_core.retrieval import RetrievalResult

from .config import RetrievalConfig
from .fusion import FusionStrategy

logger = logging.getLogger(__name__)


def _strategy_name(retriever: Retriever) -> str:
    """Best-effort strategy label for a retriever (for error/failure reporting)."""
    name = getattr(retriever, "strategy", None)
    if isinstance(name, str):
        return name
    return type(retriever).__name__.lower()


class HybridRetriever:
    """Coordinates multiple retrievers concurrently and fuses their results.

    Each retriever is executed inside an :class:`asyncio.TaskGroup`; a failing
    retriever is recorded (rather than aborting the whole run) as long as at
    least one strategy succeeds. If every retriever fails, a
    :class:`RetrievalError` is raised.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fusion: FusionStrategy,
        config: RetrievalConfig,
    ) -> None:
        self._retrievers = list(retrievers)
        self._fusion = fusion
        self._config = config

    def _weight_for(self, result: RetrievalResult) -> float:
        strategy = result.strategies[0] if result.strategies else "unknown"
        if strategy == "dense":
            return self._config.dense_weight
        if strategy == "sparse":
            return self._config.sparse_weight
        return 1.0

    def _limits_for(self, result: RetrievalResult) -> int | None:
        strategy = result.strategies[0] if result.strategies else "unknown"
        return self._config.per_source_limits.get(strategy)

    async def retrieve(self, query: Query) -> RetrievalResult:
        t0 = time.perf_counter()
        bucket: dict[int, RetrievalResult | Exception] = {}

        async def _run(idx: int, retriever: Retriever) -> None:
            try:
                bucket[idx] = await retriever.retrieve(query)
            except Exception as exc:
                strat = _strategy_name(retriever)
                wrapped = RetrievalError(
                    f"retriever '{strat}' failed: {exc}",
                    code="retrieval_error",
                    details={"strategy": strat, "cause": repr(exc)},
                )
                logger.warning("retriever %s failed: %r", strat, exc)
                bucket[idx] = wrapped

        async with asyncio.TaskGroup() as tg:
            for idx, retriever in enumerate(self._retrievers):
                tg.create_task(_run(idx, retriever))

        results: list[RetrievalResult] = []
        failed: list[str] = []
        for idx, retriever in enumerate(self._retrievers):
            outcome = bucket.get(idx)
            if outcome is None:
                failed.append(_strategy_name(retriever))
                continue
            if isinstance(outcome, Exception):
                failed.append(_strategy_name(retriever))
                continue
            results.append(outcome)

        if not results:
            raise RetrievalError(
                "all retrieval strategies failed",
                code="retrieval_error",
                details={"failed_strategies": failed},
            )

        # Apply per-strategy candidate limits before fusion.
        trimmed: list[RetrievalResult] = []
        for r in results:
            limit = self._limits_for(r)
            if limit is not None:
                r.hits = r.hits[: max(limit, 0)]
            trimmed.append(r)
        results = trimmed

        weights = [self._weight_for(r) for r in results]
        top_k = query.top_k
        fused = self._fusion.fuse(results, top_k, weights=weights, dedup=self._config.dedup)

        merged: dict[str, float] = dict(fused.timings_ms)
        for r in results:
            merged.update(r.timings_ms)
        for strat in failed:
            merged[f"{strat}_failed"] = 1.0
        merged["hybrid"] = (time.perf_counter() - t0) * 1000.0
        fused.timings_ms = merged
        return fused
