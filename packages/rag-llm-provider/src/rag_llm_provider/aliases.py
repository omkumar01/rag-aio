"""Small alias-expansion helper used by the model router."""

from __future__ import annotations

from .registry import ProviderRegistry


def expand_alias(registry: ProviderRegistry, name: str) -> str | None:
    """Resolve ``name`` (a model id or alias) to its canonical ``model_id``.

    Returns the canonical model id when ``name`` matches a model id *or* an
    alias of a model in an enabled provider; returns ``None`` when the name is
    unknown to every enabled provider. Used by :class:`~rag_llm_provider.routing.ModelRouter`
    to normalize user-supplied model references into canonical ids (and to
    populate the ``attempted`` trail on :class:`~rag_llm_provider.routing.RouteDecision`).
    """
    result = registry.find_model(name)
    if result is None:
        return None
    return result[1].model_id


__all__ = ["expand_alias"]
