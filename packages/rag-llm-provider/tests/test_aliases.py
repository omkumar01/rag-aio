"""Tests for the alias expansion helper."""

from __future__ import annotations

from rag_llm_provider import (
    ModelConfig,
    ProviderConfig,
    ProviderRegistry,
    ProvidersConfig,
    SecretRef,
    expand_alias,
)


def _p(name: str, models: list[ModelConfig]) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compatible",
        models=models,
        secret=SecretRef(kind="none"),
    )


def test_expand_alias_returns_canonical_model_id() -> None:
    reg = ProviderRegistry(
        ProvidersConfig(providers=[_p("a", [ModelConfig(model_id="m1", aliases=["g1"])])])
    )
    assert expand_alias(reg, "g1") == "m1"


def test_expand_alias_model_id_returns_itself() -> None:
    reg = ProviderRegistry(ProvidersConfig(providers=[_p("a", [ModelConfig(model_id="m1")])]))
    assert expand_alias(reg, "m1") == "m1"


def test_expand_alias_unknown_returns_none() -> None:
    reg = ProviderRegistry(ProvidersConfig(providers=[_p("a", [ModelConfig(model_id="m1")])]))
    assert expand_alias(reg, "nope") is None


def test_expand_alias_skips_disabled_provider() -> None:
    reg = ProviderRegistry(
        ProvidersConfig(
            providers=[
                _p("disabled", [ModelConfig(model_id="m1", aliases=["g1"])]).model_copy(
                    update={"enabled": False}
                ),
                _p("enabled", [ModelConfig(model_id="m2", aliases=["g1"])]),
            ]
        )
    )
    # disabled provider's alias must not resolve
    assert expand_alias(reg, "g1") == "m2"
