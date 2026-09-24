"""rag-llm-provider: provider and model management control layer.

This package owns the *control-plane* description of LLM providers and models,
their routing, and secret-by-reference handling. It depends only on
:mod:`rag_core` (no httpx): concrete health probes and API clients are injected
by consumers.

Public API
----------
Configuration:
    :class:`ProviderConfig`, :class:`ModelConfig`, :class:`ProvidersConfig`

Secrets (by reference only, ADR-0004/0006):
    :class:`SecretRef`, :func:`resolve_secret`, :func:`mask_secret`

Routing & resolution:
    :class:`ProviderRegistry`, :func:`expand_alias`, :class:`ModelRouter`,
    :class:`RouteDecision`

Health (abstract probe + aggregator):
    :class:`HealthProbe`, :class:`RegistryHealth`
"""

from __future__ import annotations

from rag_core import ConfigError, ProviderError
from rag_core.models_info import HealthState, ProviderKind

from .aliases import expand_alias
from .config import ModelConfig, ProviderConfig, ProvidersConfig
from .registry import HealthProbe, ProviderRegistry, RegistryHealth
from .routing import ModelRouter, RouteDecision
from .secrets import SecretRef, mask_secret, resolve_secret

__version__ = "0.1.1"

__all__ = [
    "ConfigError",
    "HealthProbe",
    "HealthState",
    "ModelConfig",
    "ModelRouter",
    "ProviderConfig",
    "ProviderError",
    "ProviderKind",
    "ProviderRegistry",
    "ProvidersConfig",
    "RegistryHealth",
    "RouteDecision",
    "SecretRef",
    "__version__",
    "expand_alias",
    "mask_secret",
    "resolve_secret",
]
