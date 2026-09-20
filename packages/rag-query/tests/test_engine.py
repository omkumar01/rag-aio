"""Offline tests for :class:`QueryEngine`."""

from __future__ import annotations

import pytest
from rag_core.errors import ConfigError
from rag_core.queries import Query
from rag_query.config import QueryConfig
from rag_query.engine import QueryEngine


async def test_fast_path_returns_only_original() -> None:
    config = QueryConfig()  # all transformation flags off
    engine = QueryEngine(config)
    result = await engine.process(Query(text="hello world"))
    assert len(result.variants) == 1
    assert result.variants[0].kind == "original"
    assert result.variants[0].text == "hello world"
    assert result.query.variants == result.variants
    assert result.query_class == "keyword"
    assert "process_ms" in result.timings_ms


async def test_normalization_in_pipeline() -> None:
    config = QueryConfig()
    engine = QueryEngine(config)
    result = await engine.process(Query(text="Hello   World"))
    assert result.variants[0].text == "Hello World"


async def test_dedupes_variant_texts(variant_strategy) -> None:
    config = QueryConfig()
    engine = QueryEngine(config, strategies=[variant_strategy(["hello world"])])
    result = await engine.process(Query(text="hello world"))
    assert len(result.variants) == 1
    assert result.variants[0].kind == "original"


async def test_caps_variants_at_max(variant_strategy) -> None:
    config = QueryConfig(max_variants=3)
    engine = QueryEngine(config, strategies=[variant_strategy([f"var-{i}" for i in range(5)])])
    result = await engine.process(Query(text="some multi word query"))
    assert len(result.variants) == 3
    assert result.variants[0].kind == "original"


async def test_original_variant_is_first(variant_strategy) -> None:
    config = QueryConfig(max_variants=10)
    engine = QueryEngine(
        config,
        strategies=[variant_strategy(["a", "b"], kind="expansion", strategy="expansion")],
    )
    result = await engine.process(Query(text="hello world foo"))
    assert result.variants[0].kind == "original"
    kinds = [v.kind for v in result.variants]
    assert kinds[1:] == ["expansion", "expansion"]


def test_deterministic_only_blocks_hyde() -> None:
    config = QueryConfig(hyde=True, deterministic_only=True)
    with pytest.raises(ConfigError):
        QueryEngine(config)


def test_hyde_requires_llm_when_not_deterministic() -> None:
    config = QueryConfig(hyde=True, deterministic_only=False)
    with pytest.raises(ConfigError):
        QueryEngine(config)


async def test_concurrent_strategies_produce_combined_results() -> None:
    config = QueryConfig(
        expand=True,
        decompose=True,
        deterministic_only=True,
        max_variants=10,
    )
    engine = QueryEngine(config)
    result = await engine.process(Query(text="foo and bar baz"))

    kinds = {v.kind for v in result.variants}
    assert "original" in kinds
    assert "expansion" in kinds
    assert "subquery" in kinds
    assert result.variants[0].kind == "original"
    assert len(result.variants) == 4
    assert result.query_class == "conversational"


async def test_default_config_has_no_strategies() -> None:
    engine = QueryEngine(QueryConfig())
    assert engine._strategies == []
