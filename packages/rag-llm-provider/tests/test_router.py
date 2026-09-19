"""Tests for model routing, fallbacks, and capability checks."""

from __future__ import annotations

import pytest
from rag_core import ProviderError
from rag_llm_provider import (
    ModelConfig,
    ModelRouter,
    ProviderConfig,
    ProviderRegistry,
    ProvidersConfig,
    RouteDecision,
    SecretRef,
)


def _m(
    model_id: str,
    roles: list[str] | None = None,
    caps: list[str] | None = None,
    aliases: list[str] | None = None,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, roles=roles or [], capabilities=caps or [], aliases=aliases or []
    )


def _p(
    name: str, models: list[ModelConfig], kind: str = "openai_compatible", enabled: bool = True
) -> ProviderConfig:
    return ProviderConfig(
        name=name, kind=kind, models=models, enabled=enabled, secret=SecretRef(kind="none")
    )


def _cfg(*providers: ProviderConfig) -> ProvidersConfig:
    return ProvidersConfig(providers=list(providers))


def _reg(*providers: ProviderConfig) -> ProviderRegistry:
    return ProviderRegistry(_cfg(*providers))


# --- prefer wins ---


async def test_prefer_wins_even_without_role() -> None:
    reg = _reg(_p("a", [_m("m-preferred")]), _p("b", [_m("m-other")]))
    router = ModelRouter(reg)
    dec = await router.route("generate", prefer="m-preferred")
    assert isinstance(dec, RouteDecision)
    assert dec.provider == "a"
    assert dec.model_id == "m-preferred"
    assert dec.fallback_used is False


async def test_prefer_alias_resolves_to_model_id() -> None:
    reg = _reg(_p("a", [_m("m1", aliases=["alias-1"])]))
    router = ModelRouter(reg)
    dec = await router.route("generate", prefer="alias-1")
    assert dec.model_id == "m1"
    assert dec.fallback_used is False


# --- role resolution ---


async def test_role_resolution_finds_role_in_config_order() -> None:
    reg = _reg(
        _p("a", [_m("a1", roles=["embed"])]),
        _p("b", [_m("b1", roles=["embed"])]),
    )
    router = ModelRouter(reg)
    dec = await router.route("embed")
    assert dec.provider == "a"
    assert dec.model_id == "a1"
    assert dec.fallback_used is False


async def test_prefer_failure_falls_through_to_role() -> None:
    reg = _reg(_p("a", [_m("m1", roles=["generate"])]))
    router = ModelRouter(reg)
    # prefer unknown -> fall through to role resolution, which succeeds
    dec = await router.route("generate", prefer="does-not-exist")
    assert dec.model_id == "m1"
    assert dec.fallback_used is False


# --- fallback chain ---


async def test_fallback_chain_order_and_fallback_used() -> None:
    reg = _reg(_p("fb", [_m("m1"), _m("m2"), _m("m3")]))
    router = ModelRouter(reg, fallbacks={"generate": ["m2", "m3"]})
    dec = await router.route("generate")
    # no prefer, no role match -> first fallback that exists wins
    assert dec.model_id == "m2"
    assert dec.fallback_used is True


async def test_fallback_chain_first_match_wins() -> None:
    reg = _reg(_p("fb", [_m("m1"), _m("m2"), _m("m3")]))
    router = ModelRouter(reg, fallbacks={"generate": ["m3", "m2"]})
    dec = await router.route("generate")
    assert dec.model_id == "m3"
    assert dec.fallback_used is True


async def test_fallback_recorded_in_attempted() -> None:
    reg = _reg(_p("fb", [_m("m1"), _m("m2")]))
    router = ModelRouter(reg, fallbacks={"generate": ["m2", "m1"]})
    dec = await router.route("generate")
    assert "m2" in dec.attempted


async def test_prefer_recorded_in_attempted_when_fails() -> None:
    reg = _reg(_p("fb", [_m("m1")]))
    router = ModelRouter(reg, fallbacks={"generate": ["m1"]})
    dec = await router.route("generate", prefer="nope")
    assert "nope" in dec.attempted
    assert dec.model_id == "m1"
    assert dec.fallback_used is True


# --- no model for role ---


async def test_no_model_for_role_raises() -> None:
    reg = _reg(_p("a", [_m("m1", roles=["generate"])]))
    router = ModelRouter(reg)
    with pytest.raises(ProviderError, match="no model for role"):
        await router.route("rerank")


async def test_no_model_for_role_with_empty_fallback_raises() -> None:
    reg = _reg(_p("a", [_m("m1", roles=["generate"])]))
    router = ModelRouter(reg)
    with pytest.raises(ProviderError, match="no model for role"):
        await router.route("embed", prefer="missing", required_capabilities=[])


# --- capability filtering ---


async def test_required_capabilities_filter_skips_model() -> None:
    # m1 lacks "rerank" cap, m2 has it. role "generate" -> m1 first.
    reg = _reg(
        _p(
            "a",
            [
                _m("m1", roles=["generate"], caps=["json"]),
                _m("m2", roles=["generate"], caps=["rerank"]),
            ],
        )
    )
    router = ModelRouter(reg, fallbacks={"generate": ["m2"]})
    dec = await router.route("generate", required_capabilities=["rerank"])
    assert dec.model_id == "m2"
    assert dec.fallback_used is True


async def test_required_capabilities_prefer_skipped_if_missing() -> None:
    reg = _reg(
        _p(
            "a",
            [_m("m1", roles=["generate"], caps=[]), _m("m2", roles=["generate"], caps=["rerank"])],
        )
    )
    router = ModelRouter(reg)
    # m1 (prefer) lacks "rerank"; role resolution also picks m1 first (config order)
    # which lacks the cap; no fallback chain -> should raise.
    with pytest.raises(ProviderError):
        await router.route("generate", prefer="m1", required_capabilities=["rerank"])


async def test_route_is_coroutine_function() -> None:
    import inspect

    assert inspect.iscoroutinefunction(ModelRouter.route)
