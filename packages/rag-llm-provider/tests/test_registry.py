"""Tests for the provider registry and provider-info mapping."""

from __future__ import annotations

import pytest
from rag_core import ProviderInfo
from rag_llm_provider import (
    ConfigError,
    ModelConfig,
    ProviderConfig,
    ProviderRegistry,
    ProvidersConfig,
    SecretRef,
)


def _m(
    model_id: str,
    roles: list[str] | None = None,
    aliases: list[str] | None = None,
    caps: list[str] | None = None,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        roles=roles or [],
        aliases=aliases or [],
        capabilities=caps or [],
    )


def _p(
    name: str,
    kind: str = "openai_compatible",
    models: list[ModelConfig] | None = None,
    enabled: bool = True,
    secret: SecretRef | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        models=models or [],
        enabled=enabled,
        secret=secret or SecretRef(kind="none"),
    )


def _cfg(*providers: ProviderConfig) -> ProvidersConfig:
    return ProvidersConfig(providers=list(providers))


# --- get / names / enabled ---


def test_get_known_provider() -> None:
    cfg = _cfg(_p("a"), _p("b"))
    reg = ProviderRegistry(cfg)
    assert reg.get("a").name == "a"
    assert reg.get("b").name == "b"


def test_get_unknown_provider_raises() -> None:
    reg = ProviderRegistry(_cfg(_p("a")))
    with pytest.raises(ConfigError, match="unknown provider"):
        reg.get("zzz")


def test_names_lists_all_providers_in_order() -> None:
    reg = ProviderRegistry(_cfg(_p("a"), _p("b"), _p("c")))
    assert reg.names() == ["a", "b", "c"]


def test_enabled_lists_only_enabled() -> None:
    reg = ProviderRegistry(
        _cfg(_p("a", enabled=True), _p("b", enabled=False), _p("c", enabled=True))
    )
    assert reg.enabled() == ["a", "c"]


# --- find_model ---


def test_find_model_by_id() -> None:
    reg = ProviderRegistry(_cfg(_p("a", models=[_m("gpt-4")])))
    found = reg.find_model("gpt-4")
    assert found is not None
    assert found[0].name == "a"
    assert found[1].model_id == "gpt-4"


def test_find_model_by_alias() -> None:
    reg = ProviderRegistry(_cfg(_p("a", models=[_m("gpt-4", aliases=["g4"])])))
    found = reg.find_model("g4")
    assert found is not None
    assert found[1].model_id == "gpt-4"


def test_find_model_unknown_returns_none() -> None:
    reg = ProviderRegistry(_cfg(_p("a", models=[_m("gpt-4")])))
    assert reg.find_model("nope") is None


def test_find_model_skips_disabled_providers() -> None:
    reg = ProviderRegistry(
        _cfg(
            _p("disabled", models=[_m("m1")], enabled=False),
            _p("enabled", models=[_m("m1")], enabled=True),
        )
    )
    found = reg.find_model("m1")
    assert found is not None
    assert found[0].name == "enabled"


def test_find_model_id_takes_priority_over_alias() -> None:
    # "x" is a model_id in provider a; provider b also has "x" as an alias.
    reg = ProviderRegistry(
        _cfg(
            _p("a", models=[_m("x")]),
            _p("b", models=[_m("y", aliases=["x"])]),
        )
    )
    found = reg.find_model("x")
    assert found is not None
    assert found[0].name == "a"


# --- resolve_role ---


def test_resolve_role_returns_first_config_order_match() -> None:
    reg = ProviderRegistry(
        _cfg(
            _p("a", models=[_m("a1", roles=["generate"])]),
            _p("b", models=[_m("b1", roles=["generate"])]),
        )
    )
    found = reg.resolve_role("generate")
    assert found is not None
    assert found[1].model_id == "a1"


def test_resolve_role_skips_disabled() -> None:
    reg = ProviderRegistry(
        _cfg(
            _p("a", models=[_m("a1", roles=["generate"])], enabled=False),
            _p("b", models=[_m("b1", roles=["generate"])]),
        )
    )
    found = reg.resolve_role("generate")
    assert found is not None
    assert found[1].model_id == "b1"


def test_resolve_role_unknown_returns_none() -> None:
    reg = ProviderRegistry(_cfg(_p("a", models=[_m("a1", roles=["generate"])])))
    assert reg.resolve_role("rerank") is None


# --- to_provider_info (no secret leakage) ---


def test_to_provider_info_has_auth_ref_no_secret_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_REGISTRY_ACTUAL_KEY", "sk-super-secret-leaked-if-present")
    p = _p(
        "lm",
        secret=SecretRef(kind="env", ref="RAG_REGISTRY_ACTUAL_KEY"),
        models=[_m("m1", roles=["generate"])],
    )
    reg = ProviderRegistry(_cfg(p))
    info = reg.to_provider_info(p)
    assert isinstance(info, ProviderInfo)
    assert info.auth_ref == "RAG_REGISTRY_ACTUAL_KEY"
    # The actual secret value must never appear anywhere in the serialized info.
    dumped = info.model_dump()
    blob = repr(dumped)
    assert "sk-super-secret-leaked-if-present" not in blob
    assert "sk-super-secret-leaked-if-present" not in str(info)
    assert info.health == "unknown"
    assert info.models[0].model_id == "m1"


def test_to_provider_info_none_auth_ref_when_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _p("local", models=[_m("m1")])
    reg = ProviderRegistry(_cfg(p))
    info = reg.to_provider_info(p)
    assert info.auth_ref is None
