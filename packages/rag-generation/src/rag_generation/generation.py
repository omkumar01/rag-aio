"""Module-level facade: provider registry, factory, and routing service.

Wires concrete adapters to the ``ProviderKind`` tags defined in rag-core and adds
operational policy on top: provider resolution, model override, retry with
exponential backoff (only on transient errors), a fallback provider chain, and a
usage/cost tracking hook.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from rag_core.errors import ProviderError, ProviderUnavailableError, RateLimitError
from rag_core.generation import GenerationRequest, GenerationResult
from rag_core.models_info import ProviderKind
from rag_core.protocols import LLMProvider

from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider

# Map every ProviderKind the platform knows to a concrete adapter class.
# `vllm` and `llamacpp` reuse the OpenAI-compatible wire format.
# Typed as ``Callable[..., LLMProvider]`` so the factory can forward arbitrary
# constructor kwargs (timeout_s, transport, supports_json_schema, ...) without
# the static checker narrowing on a single ``__init__`` signature.
_REGISTRY: dict[str, Callable[..., LLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "vllm": OpenAICompatibleProvider,
    "llamacpp": OpenAICompatibleProvider,
}


class GeneratorRegistry:
    """Static mapping of ``ProviderKind`` -> adapter class + factory."""

    registry: dict[str, Callable[..., LLMProvider]] = _REGISTRY

    @classmethod
    def provider_class(cls, kind: ProviderKind) -> Callable[..., LLMProvider]:
        try:
            return cls.registry[kind]
        except KeyError as e:
            raise ProviderError(f"Unknown provider kind: {kind}", code="config") from e

    @classmethod
    def create_provider(
        cls,
        kind: ProviderKind,
        base_url: str | None,
        api_key_provider: Callable[[], str | None] | None = None,
        **opts: Any,
    ) -> LLMProvider:
        provider_cls = cls.provider_class(kind)
        kwargs: dict[str, Any] = dict(opts)
        if base_url is not None:
            kwargs["base_url"] = base_url
        kwargs["api_key_provider"] = api_key_provider
        return provider_cls(**kwargs)


def create_provider(
    kind: ProviderKind,
    base_url: str | None,
    api_key_provider: Callable[[], str | None] | None = None,
    **opts: Any,
) -> LLMProvider:
    """Module-level convenience wrapping :meth:`GeneratorRegistry.create_provider`."""
    return GeneratorRegistry.create_provider(kind, base_url, api_key_provider, **opts)


class GenerationService:
    """Routes generation requests across named providers with retry + fallback.

    ``providers`` maps a name to an :class:`LLMProvider`. The first provider in the
    dict is the default when ``provider_name`` is omitted. ``router`` may return a
    preferred provider name for a request. On transient failures
    (:class:`RateLimitError`, :class:`ProviderUnavailableError`) the request is
    retried with exponential backoff (``backoff_base * 2**n``) up to ``max_retries``
    times; auth errors propagate immediately. If all retries on a provider are
    exhausted, the next name in ``fallback_names`` is tried.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        router: Callable[[GenerationRequest], str | None] | None = None,
        *,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        fallback_names: list[str] | None = None,
        on_usage: Callable[[GenerationResult], None] | None = None,
    ) -> None:
        self.providers = providers
        self.router = router
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.fallback_names = list(fallback_names or [])
        self.on_usage = on_usage

    async def generate(
        self,
        request: GenerationRequest,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> GenerationResult:
        if provider_name is None and self.router is not None:
            provider_name = self.router(request)
        if provider_name is None:
            provider_name = next(iter(self.providers))

        chain: list[str] = [provider_name]
        for name in self.fallback_names:
            if name not in chain:
                chain.append(name)

        last_exc: Exception | None = None
        for name in chain:
            provider = self.providers.get(name)
            if provider is None:
                continue
            try:
                result = await self._retry(provider, request, model)
            except RateLimitError as e:
                last_exc = e
                continue
            except ProviderUnavailableError as e:
                last_exc = e
                continue
            except ProviderError:
                # Non-transient provider error (e.g. auth): do not fall back.
                raise
            self._record_usage(result)
            return result

        if last_exc is not None:
            raise last_exc
        raise ProviderError("No provider available to handle the request")

    async def _retry(
        self, provider: LLMProvider, request: GenerationRequest, model: str | None
    ) -> GenerationResult:
        req = request.model_copy()
        if model is not None:
            req.model = model
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await provider.complete(req)
            except (RateLimitError, ProviderUnavailableError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_base * (2**attempt))
                    continue
        assert last_exc is not None
        raise last_exc

    def _record_usage(self, result: GenerationResult) -> None:
        if self.on_usage is not None and result.usage is not None:
            self.on_usage(result)


async def collect_stream(stream: AsyncIterator[str]) -> str:
    """Drain an async delta stream, concatenating the fragments in order."""
    parts: list[str] = []
    async for chunk in stream:
        parts.append(chunk)
    return "".join(parts)


__all__ = [
    "GenerationService",
    "GeneratorRegistry",
    "collect_stream",
    "create_provider",
]
