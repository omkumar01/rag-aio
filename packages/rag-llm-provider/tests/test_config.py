"""Tests for provider/model configuration and validation."""

from __future__ import annotations

import pytest
from rag_llm_provider import ConfigError, ModelConfig, ProviderConfig, ProvidersConfig


def _model(model_id: str, roles: list[str] | None = None) -> ModelConfig:
    return ModelConfig(model_id=model_id, roles=roles or [])


# --- base_url defaults per kind ---


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("anthropic", "https://api.anthropic.com"),
        ("gemini", "https://generativelanguage.googleapis.com"),
        ("openai_compatible", "http://localhost:1234/v1"),
        ("ollama", "http://localhost:11434"),
        ("vllm", "http://localhost:8000/v1"),
        ("llamacpp", "http://localhost:8080"),
    ],
)
def test_base_url_default_per_kind(kind: str, expected: str) -> None:
    cfg = ProviderConfig(name="p", kind=kind, models=[])  # type: ignore[arg-type]
    assert cfg.base_url == expected


def test_base_url_explicit_overrides_default() -> None:
    cfg = ProviderConfig(
        name="p",
        kind="openai",
        base_url="https://custom.example/v1",
        models=[],
    )
    assert cfg.base_url == "https://custom.example/v1"


# --- validation: duplicate model ids ---


def test_duplicate_model_ids_rejected() -> None:
    models = [_model("m1"), _model("m1")]
    with pytest.raises(ConfigError, match="duplicate model_id"):
        ProviderConfig(name="p", kind="openai_compatible", models=models)


# --- validation: default_models must reference known model ---


def test_default_models_unknown_model_rejected() -> None:
    models = [_model("m1")]
    with pytest.raises(ConfigError, match="default_models"):
        ProviderConfig(
            name="p",
            kind="openai_compatible",
            models=models,
            default_models={"generate": "nonexistent"},
        )


def test_default_models_valid_references_ok() -> None:
    models = [_model("m1"), _model("m2")]
    cfg = ProviderConfig(
        name="p",
        kind="openai_compatible",
        models=models,
        default_models={"generate": "m1", "embed": "m2"},
    )
    assert cfg.default_models == {"generate": "m1", "embed": "m2"}


# --- local default profile (ADR-0004) ---


def test_local_default_is_lm_studio_compatible() -> None:
    cfg = ProvidersConfig.local_default()
    assert len(cfg.providers) == 1
    p = cfg.providers[0]
    assert p.kind == "openai_compatible"
    assert p.base_url == "http://localhost:1234/v1"
    assert p.secret.kind == "none"
    assert p.enabled is True


def test_default_models_in_local_default_reference_existing_model() -> None:
    cfg = ProvidersConfig.local_default()
    ids = {m.model_id for m in cfg.providers[0].models}
    for role, mid in cfg.providers[0].default_models.items():
        assert mid in ids, f"default_models[{role}]={mid} not in models"


def test_providers_empty_by_default() -> None:
    cfg = ProvidersConfig()
    assert cfg.providers == []


def test_model_config_defaults() -> None:
    m = ModelConfig(model_id="x")
    assert m.roles == []
    assert m.capabilities == []
    assert m.aliases == []
    assert m.temperature is None
    assert m.context_window is None
    assert m.max_output_tokens is None
    assert m.embedding_dim is None
