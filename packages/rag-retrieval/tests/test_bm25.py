"""Tests for BM25Retriever (in-process bm25s)."""

from __future__ import annotations

from collections.abc import Callable

from rag_core.protocols import Retriever
from rag_core.queries import Query
from rag_retrieval import BM25Retriever, RetrievalConfig

Corpus = list[tuple[str, str, str]]


def _corpus_provider(corpus: Corpus) -> Callable[[], Corpus]:
    def _provider() -> Corpus:
        return list(corpus)

    return _provider


_CORPUS: Corpus = [
    ("c1", "d1", "hello world foo"),
    ("c2", "d1", "bar baz qux"),
    ("c3", "d2", "hello bar"),
]


async def test_bm25_relevant_ranks_first() -> None:
    retriever = BM25Retriever(_corpus_provider(_CORPUS), RetrievalConfig())
    query = Query(text="hello bar")
    result = await retriever.retrieve(query)
    assert result.strategies == ["bm25"]
    assert len(result.hits) == 3
    assert result.hits[0].chunk_id == "c3"
    assert result.hits[0].strategy == "bm25"
    assert result.hits[0].model == "bm25s"
    assert result.timings_ms["bm25"] >= 0.0


async def test_bm25_empty_corpus() -> None:
    retriever = BM25Retriever(_corpus_provider([]), RetrievalConfig())
    result = await retriever.retrieve(Query(text="anything"))
    assert result.hits == []
    assert result.strategies == ["bm25"]


async def test_bm25_is_retriever() -> None:
    retriever = BM25Retriever(_corpus_provider(_CORPUS), RetrievalConfig())
    assert isinstance(retriever, Retriever)


async def test_bm25_candidate_limit_and_refresh() -> None:
    """Index honors candidate_k and refreshes when the corpus changes."""
    base: Corpus = list(_CORPUS)
    provider = _corpus_provider(base)
    retriever = BM25Retriever(provider, RetrievalConfig(candidate_k=2))
    result = await retriever.retrieve(Query(text="hello bar"))
    assert len(result.hits) == 2

    base.clear()
    base.append(("c9", "d9", "unique token zzz"))
    result2 = await retriever.retrieve(Query(text="zzz"))
    assert result2.hits[0].chunk_id == "c9"
