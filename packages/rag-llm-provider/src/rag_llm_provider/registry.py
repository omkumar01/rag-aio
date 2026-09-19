"""Provider registry, provider-info projection, and health probing.

The registry indexes provider configs and resolves (provider, model) lookups
for routing. Health state is cached *inside the registry* (never on the
immutable config object) and updated by an injected :class:`HealthProbe`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_core import ConfigError
from rag_core.models_info import HealthState, ModelInfo, ProviderInfo

from .config import ModelConfig, ProviderConfig, ProvidersConfig


class ProviderRegistry:
    """In-memory index of provider configs with a per-provider health cache."""

    def __init__(self, config: ProvidersConfig) -> None:
        self._config = config
        self._by_name: dict[str, ProviderConfig] = {}
        for provider in config.providers:
            if provider.name in self._by_name:
                raise ConfigError(f"duplicate provider name {provider.name!r} in ProvidersConfig")
            self._by_name[provider.name] = provider
        # Health cache lives here, not on the (immutable) config object.
        self._health: dict[str, HealthState] = {
            provider.name: "unknown" for provider in config.providers
        }

    def get(self, name: str) -> ProviderConfig:
        """Return the provider config for ``name`` or raise :class:`ConfigError`."""
        try:
            return self._by_name[name]
        except KeyError:
            raise ConfigError(f"unknown provider {name!r}") from None

    def names(self) -> list[str]:
        """Provider names in configuration order."""
        return [provider.name for provider in self._config.providers]

    def enabled(self) -> list[str]:
        """Names of enabled providers, in configuration order."""
        return [provider.name for provider in self._config.providers if provider.enabled]

    def find_model(self, model_or_alias: str) -> tuple[ProviderConfig, ModelConfig] | None:
        """Resolve a model by id then alias across enabled providers.

        Enabled providers are scanned in configuration order. Model ids take
        priority over aliases: the first matching model id wins; if none match,
        aliases are scanned in the same order. Disabled providers are skipped.
        """
        # Pass 1: match by model_id.
        for provider in self._config.providers:
            if not provider.enabled:
                continue
            for model in provider.models:
                if model.model_id == model_or_alias:
                    return provider, model
        # Pass 2: match by alias.
        for provider in self._config.providers:
            if not provider.enabled:
                continue
            for model in provider.models:
                if model_or_alias in model.aliases:
                    return provider, model
        return None

    def resolve_role(self, role: str) -> tuple[ProviderConfig, ModelConfig] | None:
        """Find the first model (config order) whose ``roles`` include ``role``.

        Only enabled providers are considered. Returns ``None`` when no model
        declares the role.
        """
        for provider in self._config.providers:
            if not provider.enabled:
                continue
            for model in provider.models:
                if role in model.roles:
                    return provider, model
        return None

    def set_health(self, name: str, state: HealthState) -> None:
        """Update the cached health state for a provider."""
        if name not in self._by_name:
            raise ConfigError(f"unknown provider {name!r}")
        self._health[name] = state

    @property
    def config(self) -> ProvidersConfig:
        """The underlying (immutable) providers configuration."""
        return self._config

    def to_provider_info(self, provider: ProviderConfig) -> ProviderInfo:
        """Project a :class:`ProviderConfig` onto the public :class:`ProviderInfo`.

        ``auth_ref`` is the secret *reference* (env var name / file path), or
        ``None`` -- never the resolved credential value. The current cached
        health state is included.
        """
        auth_ref = provider.secret.describe() if provider.secret.kind != "none" else None
        return ProviderInfo(
            name=provider.name,
            kind=provider.kind,
            base_url=provider.base_url,
            auth_ref=auth_ref,
            models=[
                ModelInfo(
                    model_id=model.model_id,
                    provider=provider.name,
                    display_name=None,
                    capabilities=model.capabilities,
                    context_window=model.context_window,
                    max_output_tokens=model.max_output_tokens,
                    embedding_dim=model.embedding_dim,
                    pricing={},
                )
                for model in provider.models
            ],
            default_models=provider.default_models,
            enabled=provider.enabled,
            health=self._health.get(provider.name, "unknown"),
        )


@runtime_checkable
class HealthProbe(Protocol):
    """Abstract health probe (concrete implementation injected by consumers).

    Implementations may use httpx or any transport; this package declares only
    the contract so it stays dependency-free.
    """

    async def probe(self, provider: ProviderConfig) -> HealthState:
        """Return the current :class:`HealthState` for ``provider``."""
        ...


class RegistryHealth:
    """Aggregator that runs a :class:`HealthProbe` over every provider.

    On completion, each provider's cached health state inside the registry is
    updated to match the probe result.
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    async def check_all(self, probe: HealthProbe) -> dict[str, HealthState]:
        results: dict[str, HealthState] = {}
        for name in self._registry.names():
            provider = self._registry.get(name)
            state = await probe.probe(provider)
            self._registry.set_health(name, state)
            results[name] = state
        return results


__all__ = [
    "HealthProbe",
    "ProviderRegistry",
    "RegistryHealth",
]
