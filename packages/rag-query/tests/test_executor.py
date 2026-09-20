"""Offline tests for :class:`ParallelRetrievalExecutor`."""

from __future__ import annotations

import pytest
from rag_core.errors import RetrievalError
from rag_core.queries import Query, QueryVariant
from rag_query.retrieval_exec import ParallelRetrievalExecutor


def _make_variants(query: Query) -> list[QueryVariant]:
    original = QueryVariant(
        query_id=query.id,
        text=query.text,
        kind="original",
        strategy="original",
    )
    variant = QueryVariant(
        query_id=query.id,
        text="expansion variant text",
        kind="expansion",
        strategy="expansion",
    )
    return [original, variant]


async def test_two_retrievers_two_variants_four_results(recording_retriever) -> None:
    r1 = recording_retriever("a")
    r2 = recording_retriever("b")
    executor = ParallelRetrievalExecutor([r1, r2], max_concurrency=4)

    query = Query(text="hello world")
    variants = _make_variants(query)
    results = await executor.execute(query, variants, top_k=5)

    assert len(results) == 4
    assert r1.calls == 2
    assert r2.calls == 2
    ids = {r.query_id for r in results}
    assert {v.id for v in variants} == ids


async def test_one_failing_retriever_tolerated(recording_retriever, failing_retriever) -> None:
    good = recording_retriever("good")
    bad = failing_retriever()
    executor = ParallelRetrievalExecutor([good, bad], max_concurrency=4)

    query = Query(text="hello world foo")
    results = await executor.execute(query, _make_variants(query), top_k=5)

    assert len(results) == 2
    assert good.calls == 2
    assert bad.calls == 2


async def test_all_fail_raises_retrieval_error(failing_retriever) -> None:
    bad1 = failing_retriever()
    bad2 = failing_retriever()
    executor = ParallelRetrievalExecutor([bad1, bad2], max_concurrency=4)

    query = Query(text="hello world foo")
    with pytest.raises(RetrievalError):
        await executor.execute(query, _make_variants(query), top_k=5)


async def test_no_retrievers_raises() -> None:
    with pytest.raises(RetrievalError):
        await ParallelRetrievalExecutor([]).execute(Query(text="x"), [], top_k=5)


async def test_semaphore_bounds_concurrency(counting_retriever) -> None:
    retriever = counting_retriever()
    executor = ParallelRetrievalExecutor([retriever], max_concurrency=2)

    query = Query(text="hello world foo bar")
    variants = [
        QueryVariant(
            query_id=query.id,
            text=f"variant-{i}",
            kind="expansion",
            strategy="expansion",
        )
        for i in range(3)
    ]
    results = await executor.execute(query, variants, top_k=5)

    assert retriever.peak <= 2
    assert retriever.peak == 2
    assert retriever.calls == 4
    assert len(results) == 4


async def test_semaphore_serial_when_one(counting_retriever) -> None:
    retriever = counting_retriever()
    executor = ParallelRetrievalExecutor([retriever], max_concurrency=1)

    query = Query(text="hello world foo bar")
    variants = _make_variants(query)  # 1 original + 1 variant => 2 forms
    results = await executor.execute(query, variants, top_k=5)

    assert retriever.peak <= 1
    assert retriever.calls == 2
    assert len(results) == 2


def test_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        ParallelRetrievalExecutor([object()], max_concurrency=0)  # type: ignore[arg-type]
