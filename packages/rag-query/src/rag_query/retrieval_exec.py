"""Parallel retrieval execution across query variants and retrievers."""

from __future__ import annotations

import asyncio
import logging

from rag_core.errors import RetrievalError
from rag_core.protocols import Retriever
from rag_core.queries import Query, QueryVariant
from rag_core.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class ParallelRetrievalExecutor:
    """Runs one or more retrievers against the original query and its variants.

    Retrieval is fanned out across ``(query form, retriever)`` pairs using an
    :class:`asyncio.TaskGroup` bounded by a semaphore so that at most
    ``max_concurrency`` retrievals run at once. Retrievers are duck-typed: any
    object exposing ``async def retrieve(query: Query) -> RetrievalResult`` is
    accepted.

    Each variant (including a ``kind="original"`` variant when present) becomes
    one query form; results are tagged with the producing variant's id. A single
    failing pair is logged and skipped so partial results are still returned. If
    *every* pair fails, :class:`~rag_core.errors.RetrievalError` is raised.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._retrievers = list(retrievers)
        self._max_concurrency = max_concurrency

    async def execute(
        self,
        query: Query,
        variants: list[QueryVariant],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Retrieve for the original query and every non-original variant.

        ``variants`` is expected to include a ``kind="original"`` variant
        (as produced by :class:`~rag_query.engine.QueryEngine`); if it does not,
        the original ``query`` itself is retrieved first.
        """
        if not self._retrievers:
            raise RetrievalError("no retrievers configured")

        # One query form per variant, inheriting the base query's filters/tenant.
        forms: list[tuple[str, Query]] = []
        has_original = False
        for variant in variants:
            if variant.kind == "original" and has_original:
                continue
            if variant.kind == "original":
                has_original = True
            form = query.model_copy(
                update={"text": variant.text, "id": variant.id, "variants": [], "top_k": top_k}
            )
            forms.append((variant.id, form))

        # Ensure the original query text is always retrieved at least once.
        if not has_original:
            forms.insert(0, (query.id, query))

        results: list[RetrievalResult] = []
        semaphore = asyncio.Semaphore(self._max_concurrency)
        total = len(forms) * len(self._retrievers)
        failures = 0

        async def _retrieve(form_id: str, form_query: Query, retriever: Retriever) -> None:
            nonlocal failures
            async with semaphore:
                try:
                    result = await retriever.retrieve(form_query)
                except Exception:
                    failures += 1
                    logger.exception(
                        "retrieval failed for query_id=%s via %s",
                        form_id,
                        type(retriever).__name__,
                    )
                    return
            result.query_id = form_id
            results.append(result)

        async with asyncio.TaskGroup() as task_group:
            for form_id, form_query in forms:
                for retriever in self._retrievers:
                    task_group.create_task(_retrieve(form_id, form_query, retriever))

        if total > 0 and failures == total:
            raise RetrievalError(
                f"all {total} retrieval pairs failed",
                details={"total": total},
            )

        return results
