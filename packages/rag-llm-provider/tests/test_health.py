"""Tests for the health-probe protocol and registry health aggregation."""

from __future__ import annotations

from rag_llm_provider import (
    HealthProbe,
    HealthState,
    ModelConfig,
    ProviderConfig,
    ProviderRegistry,
    ProvidersConfig,
    RegistryHealth,
    SecretRef,
)


def _p(name: str, models: list[ModelConfig], kind: str = "openai_compatible") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind=kind,
        models=models,
        secret=SecretRef(kind="none"),
    )


class FakeProbe:
    """Concrete HealthProbe injected by the consumer (no httpx in this package)."""

    def __init__(self, states: dict[str, HealthState]) -> None:
        self._states = states
        self.calls: list[str] = []

    async def probe(self, provider: ProviderConfig) -> HealthState:
        self.calls.append(provider.name)
        return self._states[provider.name]


async def test_check_all_updates_cached_state() -> None:
    reg = ProviderRegistry(
        ProvidersConfig(
            providers=[
                _p("a", [ModelConfig(model_id="ma")]),
                _p("b", [ModelConfig(model_id="mb")]),
            ]
        )
    )
    probe = FakeProbe({"a": "healthy", "b": "down"})
    health = RegistryHealth(reg)
    results = await health.check_all(probe)
    assert results == {"a": "healthy", "b": "down"}
    # cached state inside the registry must be updated
    assert reg.to_provider_info(reg.get("a")).health == "healthy"
    assert reg.to_provider_info(reg.get("b")).health == "down"
    assert sorted(probe.calls) == ["a", "b"]


async def test_health_probe_protocol_matches() -> None:
    probe = FakeProbe({"a": "degraded"})
    assert isinstance(probe, HealthProbe)


async def test_unknown_default_health_is_unknown() -> None:
    reg = ProviderRegistry(ProvidersConfig(providers=[_p("a", [ModelConfig(model_id="m")])]))
    info = reg.to_provider_info(reg.get("a"))
    assert info.health == "unknown"
