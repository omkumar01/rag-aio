"""Offline tests for query strategies (deterministic + fake-LLM branches)."""

from __future__ import annotations

import pytest
from rag_core.errors import ConfigError
from rag_core.queries import Query
from rag_query.strategies.decompose import DecomposeStrategy
from rag_query.strategies.expansion import ExpansionStrategy
from rag_query.strategies.hyde import HyDEStrategy
from rag_query.strategies.rewrite import RewriteStrategy


def _q(text: str) -> Query:
    return Query(text=text)


async def test_expansion_deterministic_keyword_variant() -> None:
    query = _q("what is LLM retrieval")
    strategy = ExpansionStrategy(llm=None, count=2)
    variants = await strategy.transform(query)
    assert len(variants) == 1
    variant = variants[0]
    assert variant.kind == "expansion"
    assert variant.strategy == "expansion"
    assert variant.text == "LLM retrieval"
    assert variant.query_id == query.id


async def test_expansion_with_llm_parses_lines(fake_llm) -> None:
    llm = fake_llm("phrase one\nphrase two\nphrase three")
    strategy = ExpansionStrategy(llm=llm, count=2)
    variants = await strategy.transform(_q("what is RAG"))
    assert [v.text for v in variants] == ["phrase one", "phrase two"]
    assert all(v.kind == "expansion" for v in variants)
    assert llm.requests  # the LLM was actually called


async def test_rewrite_deterministic_passthrough() -> None:
    strategy = RewriteStrategy(llm=None)
    variants = await strategy.transform(_q("double   spaced   text"))
    assert len(variants) == 1
    assert variants[0].kind == "rewrite"
    assert variants[0].text == "double spaced text"


async def test_rewrite_with_llm(fake_llm) -> None:
    llm = fake_llm("standalone search query")
    strategy = RewriteStrategy(llm=llm)
    variants = await strategy.transform(_q("and then it was mentioned before"))
    assert len(variants) == 1
    assert variants[0].text == "standalone search query"


async def test_hyde_with_llm_returns_passage(fake_llm) -> None:
    llm = fake_llm("A hypothetical answer passage.")
    strategy = HyDEStrategy(llm=llm)
    variants = await strategy.transform(_q("what is RAG"))
    assert len(variants) == 1
    assert variants[0].kind == "hyde"
    assert variants[0].strategy == "hyde"
    assert variants[0].text == "A hypothetical answer passage."


def test_hyde_requires_llm() -> None:
    with pytest.raises(ConfigError):
        HyDEStrategy(llm=None)


async def test_decompose_deterministic_splits_conjunctions() -> None:
    strategy = DecomposeStrategy(llm=None)
    variants = await strategy.transform(_q("foo and bar and then baz"))
    assert [v.text for v in variants] == ["foo", "bar", "baz"]
    assert all(v.kind == "subquery" for v in variants)


async def test_decompose_deterministic_semicolon() -> None:
    strategy = DecomposeStrategy(llm=None)
    variants = await strategy.transform(_q("alpha; beta ; gamma"))
    assert [v.text for v in variants] == ["alpha", "beta", "gamma"]


async def test_decompose_llm_json_array(fake_llm) -> None:
    llm = fake_llm('["sub one", "sub two", "sub three"]')
    strategy = DecomposeStrategy(llm=llm, max_subqueries=3)
    variants = await strategy.transform(_q("multi part query"))
    assert [v.text for v in variants] == ["sub one", "sub two", "sub three"]
    assert all(v.kind == "subquery" for v in variants)


async def test_decompose_llm_json_embedded_in_prose(fake_llm) -> None:
    llm = fake_llm('Here are the subs:\n["a", "b", "c"]\nThanks!')
    strategy = DecomposeStrategy(llm=llm, max_subqueries=5)
    variants = await strategy.transform(_q("multi part query"))
    assert [v.text for v in variants] == ["a", "b", "c"]


async def test_decompose_llm_garbage_fallback(fake_llm) -> None:
    llm = fake_llm("sub one\nsub two\n")
    strategy = DecomposeStrategy(llm=llm, max_subqueries=5)
    variants = await strategy.transform(_q("multi part query"))
    assert [v.text for v in variants] == ["sub one", "sub two"]


@pytest.mark.parametrize(
    "strategy",
    [ExpansionStrategy(), RewriteStrategy(), DecomposeStrategy()],
)
async def test_strategies_short_input_returns_empty(strategy) -> None:
    assert await strategy.transform(_q("hi")) == []
    assert await strategy.transform(_q("")) == []


async def test_expansion_capped_at_count(fake_llm) -> None:
    llm = fake_llm("one\ntwo\nthree\nfour")
    strategy = ExpansionStrategy(llm=llm, count=2)
    variants = await strategy.transform(_q("what is RAG"))
    assert len(variants) == 2
