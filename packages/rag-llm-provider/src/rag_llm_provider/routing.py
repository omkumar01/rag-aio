"""Model routing with prefer / role / fallback resolution and capability gating."""

from __future__ import annotations

from rag_core import ProviderError, RagBaseModel

from .aliases import expand_alias
from .config import ModelConfig, ProviderConfig
from .registry import ProviderRegistry


class RouteDecision(RagBaseModel):
    """Outcome of a routing decision."""

    provider: str
    model_id: str
    fallback_used: bool
    attempted: list[str]


class ModelRouter:
    """Routes a logical role (and optional preferred model) to a concrete model.

    Resolution order:
      1. ``prefer`` (explicit model id or alias) when it satisfies any required
         capabilities.
      2. Role resolution: the first model (config order) declaring ``role``.
      3. Fallback chain: the named candidates under ``fallbacks[role]`` in order.
      4. :class:`~rag_core.ProviderError` when nothing matches.

    ``route`` is async because resolution is currently synchronous over the
    registry's in-memory state, but the contract is async to permit future
    dynamic capability checks (e.g. live cardinality/capability probes per
    provider) without changing callers.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        fallbacks: dict[str, list[str]] | None = None,
    ) -> None:
        self._registry = registry
        self._fallbacks = fallbacks or {}

    @staticmethod
    def _has_capabilities(model: ModelConfig, required: list[str] | None) -> bool:
        if not required:
            return True
        have = set(model.capabilities)
        return all(cap in have for cap in required)

    @staticmethod
    def _find_eligible(
        registry: ProviderRegistry,
        name: str,
        required_caps: list[str] | None,
    ) -> tuple[ProviderConfig, ModelConfig] | None:
        result = registry.find_model(name)
        if result is not None and ModelRouter._has_capabilities(result[1], required_caps):
            return result
        return None

    async def route(
        self,
        role: str,
        prefer: str | None = None,
        required_capabilities: list[str] | None = None,
    ) -> RouteDecision:
        """Resolve ``role`` (and optional ``prefer``) to a :class:`RouteDecision`.

        ``attempted`` records every candidate name examined, in order,
        including the one ultimately chosen, so callers can audit the path the
        router took.
        """
        attempted: list[str] = []
        caps = required_capabilities

        # 1. Explicit preference takes priority when eligible.
        if prefer is not None:
            attempted.append(expand_alias(self._registry, prefer) or prefer)
            eligible = self._find_eligible(self._registry, prefer, caps)
            if eligible is not None:
                return RouteDecision(
                    provider=eligible[0].name,
                    model_id=eligible[1].model_id,
                    fallback_used=False,
                    attempted=attempted,
                )

        # 2. Role resolution via the model's declared ``roles``.
        role_match = self._registry.resolve_role(role)
        if role_match is not None:
            attempted.append(role_match[1].model_id)
            if self._has_capabilities(role_match[1], caps):
                return RouteDecision(
                    provider=role_match[0].name,
                    model_id=role_match[1].model_id,
                    fallback_used=False,
                    attempted=attempted,
                )

        # 3. Fallback chain walk (in declared order).
        for candidate in self._fallbacks.get(role, []):
            attempted.append(expand_alias(self._registry, candidate) or candidate)
            eligible = self._find_eligible(self._registry, candidate, caps)
            if eligible is not None:
                return RouteDecision(
                    provider=eligible[0].name,
                    model_id=eligible[1].model_id,
                    fallback_used=True,
                    attempted=attempted,
                )

        # 4. Nothing resolved.
        raise ProviderError(f"no model for role {role!r}")


__all__ = ["ModelRouter", "RouteDecision"]
